//! Exact endgame solver (design doc §4.5 B8).
//!
//! Buro's endgame protocol has three phases: selective midgame -> WLD
//! (win/loss/draw) -> exact disc-difference maximisation. This module
//! implements the last two; the selective-midgame -> endgame *switch* lives
//! in the engine driver ([`crate::engine`]).
//!
//! # Why `depth = empties` reaches the terminal exactly
//!
//! [`negascout`] resolves a node in this order (design doc §4.5 B6):
//!
//! 1. `state.is_terminal()` -> exact [`terminal_score`];
//! 2. else no legal placement (must pass) -> apply `Pass`, recurse at the
//!    **same depth** with **empties unchanged**;
//! 3. else `depth == 0` -> the heuristic leaf evaluator;
//! 4. else the move loop at `depth - 1`.
//!
//! A *placement* decrements both `depth` and the empty count; a *pass*
//! changes neither. The terminal check (step 1) runs **before** the
//! `depth == 0` horizon check (step 3). Therefore, starting a search with
//! `depth = state.board.empty_count()`, every root-to-leaf path consumes one
//! unit of `depth` per placed disc, and the board fills (or both sides get
//! stuck) at or before `depth` reaches 0 — so the terminal branch is always
//! taken first and **the heuristic evaluator is never invoked**. The value
//! is then the true game-theoretic final disc differential
//! (side-to-move POV, empties awarded to the winner, design doc §4.5 B6).
//!
//! A dummy [`DiscDiffEval`] is passed only to satisfy the
//! `LeafEvaluator` type parameter; by the argument above its leaf method is
//! unreachable from `solve_exact` / `solve_wld`.

use logistello_core::Zobrist;
use logistello_eval::DiscDiffEval;
use othello_core::{GameState, Move};

use crate::alphabeta::{INF, SearchConfig, SearchContext, negascout, terminal_score};
use crate::killer::KillerTable;
use crate::tt::TranspositionTable;

/// Builds a search context configured for exact endgame play.
///
/// TT + killer ordering only reorder moves / cause value-preserving
/// cutoffs, so they are kept enabled (they only make the exact solve
/// faster, never change the value — proven by the Phase 2 invariants).
fn endgame_ctx<'a>(
    tt: &'a mut TranspositionTable,
    killers: &'a mut KillerTable,
    zobrist: &'a Zobrist,
    evaluator: &'a DiscDiffEval,
) -> SearchContext<'a, DiscDiffEval> {
    SearchContext {
        evaluator,
        tt,
        killers,
        zobrist,
        config: SearchConfig::default(),
        nodes: 0,
    }
}

/// Exact final disc differential of `root` under perfect play, from the
/// side-to-move's perspective (design doc §4.5 B8 phase 3).
///
/// Searches to the game's terminal with a full window; the returned value
/// is always even (empties go to the winner, design doc §4.5 B6).
#[must_use]
pub fn solve_exact(
    root: &GameState,
    tt: &mut TranspositionTable,
    killers: &mut KillerTable,
    zobrist: &Zobrist,
) -> i32 {
    let dummy = DiscDiffEval;
    let mut ctx = endgame_ctx(tt, killers, zobrist, &dummy);
    let depth = root.board.empty_count();
    negascout(&mut ctx, root, -INF, INF, depth, 0)
}

/// Win/loss/draw of `root` under perfect play, from the side-to-move's
/// perspective (design doc §4.5 B8 phase 2): a full search to the terminal
/// through the **narrow window** `(alpha, beta) = (-1, +1)` (Buro 2002 /
/// Edax: WLD is the same exact search, just sign-resolving).
///
/// Returns `+1` (win), `0` (draw), or `-1` (loss). With the `(-1, +1)`
/// window the negascout value is clamped to that range, so its sign is the
/// game outcome; we re-clamp defensively and return the `signum`.
#[must_use]
pub fn solve_wld(
    root: &GameState,
    tt: &mut TranspositionTable,
    killers: &mut KillerTable,
    zobrist: &Zobrist,
) -> i32 {
    let dummy = DiscDiffEval;
    let mut ctx = endgame_ctx(tt, killers, zobrist, &dummy);
    let depth = root.board.empty_count();
    // Narrow window: alpha = -1, beta = +1. A fail-low (<= -1) is a loss, a
    // fail-high (>= +1) is a win, an in-window 0 is a draw.
    let v = negascout(&mut ctx, root, -1, 1, depth, 0);
    v.clamp(-1, 1).signum()
}

/// Exact best endgame move and its value (side-to-move POV).
///
/// Mirrors the Phase 2 iterative root handling
/// ([`crate::iterative::search_with`]): terminal -> `(Pass, terminal)`;
/// must-pass -> `(Pass, -solve_exact(child))` at the **same** (= empties)
/// depth; otherwise the standard NegaScout root layer over the exact solver
/// so the chosen move is observable.
#[must_use]
pub fn best_endgame_move(
    root: &GameState,
    tt: &mut TranspositionTable,
    killers: &mut KillerTable,
    zobrist: &Zobrist,
) -> (Move, i32) {
    if root.is_terminal() {
        return (Move::Pass, terminal_score(root));
    }

    let moves = root.legal_moves();
    if moves.is_empty() {
        // Must pass: side flips, empties unchanged -> solve the child to
        // terminal and negate (design doc §4.5 B6 step 2).
        let mut passed = root.clone();
        passed
            .apply_move(Move::Pass)
            .expect("pass is legal when there is no legal placement");
        let v = -solve_exact(&passed, tt, killers, zobrist);
        return (Move::Pass, v);
    }

    let dummy = DiscDiffEval;
    let mut ctx = endgame_ctx(tt, killers, zobrist, &dummy);

    let mut best = -INF;
    let mut best_move = moves[0];
    let mut alpha = -INF;
    let beta = INF;
    for (i, &m) in moves.iter().enumerate() {
        let mut child = root.clone();
        child.apply_move(m).expect("legal move applies");
        // Child is searched to *its* terminal: a placement consumed one
        // empty, so the child's exact depth is its own empty count.
        let child_depth = child.board.empty_count();
        let b = if i == 0 { beta } else { alpha + 1 };
        let mut score = -negascout(&mut ctx, &child, -b, -alpha, child_depth, 1);
        if i != 0 && alpha < score && score < beta {
            score = -negascout(&mut ctx, &child, -beta, -alpha, child_depth, 1);
        }
        if score > best {
            best = score;
            best_move = m;
        }
        if score > alpha {
            alpha = score;
        }
    }
    (best_move, best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use othello_core::{Color, Coord};

    fn fresh() -> (TranspositionTable, KillerTable, Zobrist) {
        (
            TranspositionTable::new(),
            KillerTable::new(),
            Zobrist::new(),
        )
    }

    #[test]
    fn solve_exact_on_full_board() {
        // All-black full board, Black to move: 2P - 64 = 64.
        let mut s = GameState::standard_8x8();
        for r in 0..8u8 {
            for c in 0..8u8 {
                s.board.set(Coord::new(r, c), Some(Color::Black));
            }
        }
        s.side_to_move = Color::Black;
        s.consecutive_passes = 2;
        let (mut tt, mut k, z) = fresh();
        assert_eq!(solve_exact(&s, &mut tt, &mut k, &z), 64);
        assert_eq!(solve_wld(&s, &mut tt, &mut k, &z), 1);
    }

    #[test]
    fn solve_exact_matches_reference_near_end() {
        // Deterministic walk down to a few empties, then check the solver
        // equals the perfect-play differential (reference at depth =
        // empties also reaches the terminal).
        use crate::iterative::reference_negamax;
        let mut s = GameState::standard_8x8();
        while s.board.empty_count() > 10 && !s.is_terminal() {
            let mv = s.legal_moves();
            if mv.is_empty() {
                s.apply_move(Move::Pass).unwrap();
            } else {
                s.apply_move(mv[0]).unwrap();
            }
        }
        let empties = s.board.empty_count();
        let want = reference_negamax(&s, empties, &DiscDiffEval);
        let (mut tt, mut k, z) = fresh();
        assert_eq!(solve_exact(&s, &mut tt, &mut k, &z), want);
        assert_eq!(solve_exact(&s, &mut tt, &mut k, &z) % 2, 0);
    }

    #[test]
    fn best_endgame_move_value_equals_solve_exact() {
        let mut s = GameState::standard_8x8();
        while s.board.empty_count() > 9 && !s.is_terminal() {
            let mv = s.legal_moves();
            if mv.is_empty() {
                s.apply_move(Move::Pass).unwrap();
            } else {
                s.apply_move(mv[0]).unwrap();
            }
        }
        let (mut tt, mut k, z) = fresh();
        let (mv, val) = best_endgame_move(&s, &mut tt, &mut k, &z);
        assert_eq!(val, solve_exact(&s, &mut tt, &mut k, &z));
        if !s.legal_moves().is_empty() {
            assert!(s.legal_moves().contains(&mv));
        }
    }

    #[test]
    fn solve_equals_reference_on_crafted_terminal() {
        use crate::iterative::reference_negamax;
        let mut t = GameState::standard_8x8();
        for r in 0..8u8 {
            for c in 0..8u8 {
                t.board.set(Coord::new(r, c), None);
            }
        }
        // 2 black, 2 white, rest empty but both sides stuck -> terminal.
        t.board.set(Coord::new(0, 0), Some(Color::Black));
        t.board.set(Coord::new(0, 1), Some(Color::Black));
        t.board.set(Coord::new(7, 6), Some(Color::White));
        t.board.set(Coord::new(7, 7), Some(Color::White));
        t.side_to_move = Color::Black;
        t.consecutive_passes = 2;
        let (mut tt, mut k, z) = fresh();
        let e = t.board.empty_count();
        assert_eq!(
            solve_exact(&t, &mut tt, &mut k, &z),
            reference_negamax(&t, e, &DiscDiffEval)
        );
    }
}
