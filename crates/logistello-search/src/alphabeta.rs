//! NegaScout / PVS alpha-beta search core (design doc §4.3.2 + §4.5 B6).
//!
//! This is a faithful implementation of the corrected
//! *NegaScout-with-Transposition-Table* pseudocode in design doc §4.3.2:
//!
//! - TT probe with EXACT / LOWER / UPPER window narrowing and an early
//!   return when the narrowed window collapses (`alpha >= beta`).
//! - Move ordering: TT best move first, then killer moves, then the
//!   remaining legal moves.
//! - Null-window scout for every move after the first, with a re-search of
//!   any move `m != moves[0]` whose scout score lands strictly inside
//!   `(alpha, beta)` (exactly the pseudocode's
//!   `if alpha < score < beta and m != moves[0]`).
//! - TT store of the final `best` value with the bound deduced from the
//!   original window.
//!
//! Pass / terminal handling is **owned by the search** and follows §4.5 B6
//! exactly (the search does not rely on `GameState` auto-pass):
//!
//! 1. `state.is_terminal()` -> exact [`terminal_score`].
//! 2. else no legal placement (must pass, opponent can move) -> apply
//!    `Move::Pass`, recurse at the **same depth** (do not decrement, empties
//!    unchanged), negate, best move = `Pass`.
//! 3. else `depth == 0` -> the leaf evaluator (side-to-move POV).
//! 4. else the normal NegaScout move loop at `depth - 1`.

use logistello_eval::LeafEvaluator;
use othello_core::{GameState, Move};

use crate::killer::KillerTable;
use crate::probcut::{ProbCutConfig, probcut_bounds};
use crate::tt::{Bound, Entry, TranspositionTable};

/// Score sentinel used for ±infinity. Far outside the legal score range
/// `±(64 + 64)` so terminal scores and real evaluations always dominate.
pub const INF: i32 = 1_000_000;

/// Tunable search behaviour. Defaults enable every (value-preserving)
/// ordering heuristic and keep ProbCut **off**, so a default config
/// reproduces the Phase 2/3 NegaScout exactly (every soundness invariant
/// holds); tests flip individual flags to assert invariance.
#[derive(Debug, Clone, Copy)]
pub struct SearchConfig {
    /// Use the transposition table for cutoffs / move ordering.
    pub use_tt: bool,
    /// Use killer-move ordering.
    pub use_killers: bool,
    /// Single-ProbCut configuration (design doc §4.3.4 / §4.5 B7).
    ///
    /// [`ProbCutConfig::default`] is **disabled**; when disabled the search
    /// is byte-identical to the Phase 2 NegaScout. ProbCut is an *unsound*
    /// forward prune (by design): with it enabled the search trades exact
    /// minimax equality for speed, so the correct invariant becomes
    /// statistical agreement, never byte equality.
    pub probcut: ProbCutConfig,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            use_tt: true,
            use_killers: true,
            probcut: ProbCutConfig::default(),
        }
    }
}

/// Exact terminal score from the side-to-move's perspective (design doc
/// §4.5 B6): disc differential with the empty squares awarded to the winner
/// (tournament rule).
///
/// `P` = side-to-move discs, `O` = opponent discs, `E` = empty squares,
/// `diff = P - O`:
/// - `diff > 0` -> `diff + E`
/// - `diff < 0` -> `diff - E`
/// - `diff == 0` -> `0`
///
/// A full board (`E == 0`) gives `2P - 64`; solver values are always even.
#[must_use]
pub fn terminal_score(state: &GameState) -> i32 {
    let me = state.side_to_move;
    let opp = me.opponent();
    let p = state.board.count(me) as i32;
    let o = state.board.count(opp) as i32;
    let e = state.board.empty_count() as i32;
    let diff = p - o;
    if diff > 0 {
        diff + e
    } else if diff < 0 {
        diff - e
    } else {
        0
    }
}

/// Mutable search context threaded through the recursion.
pub struct SearchContext<'a, E: LeafEvaluator> {
    /// Leaf evaluator (depth-0, side-to-move POV).
    pub evaluator: &'a E,
    /// Shared transposition table.
    pub tt: &'a mut TranspositionTable,
    /// Per-ply killer table.
    pub killers: &'a mut KillerTable,
    /// Zobrist hasher.
    pub zobrist: &'a logistello_core::Zobrist,
    /// Behaviour flags.
    pub config: SearchConfig,
    /// Visited-node counter (includes pass and terminal nodes).
    pub nodes: u64,
}

/// Orders the legal `moves` of `state` in place: TT best move first, then
/// killer moves (most recent first), then the rest in generation order.
/// Pure reordering — never drops or adds a move.
fn order_moves(
    moves: &mut [Move],
    tt_move: Option<Move>,
    killers: [Option<Move>; 2],
    use_killers: bool,
) {
    // Rank: 0 = TT move, 1 = killer 0, 2 = killer 1, 3 = everything else.
    let rank = |m: Move| -> u8 {
        if Some(m) == tt_move {
            0
        } else if use_killers && Some(m) == killers[0] {
            1
        } else if use_killers && Some(m) == killers[1] {
            2
        } else {
            3
        }
    };
    // Stable sort keeps the generator order within the "rest" bucket, which
    // keeps the search deterministic.
    moves.sort_by_key(|m| rank(*m));
}

/// NegaScout / PVS with transposition table (design doc §4.3.2).
///
/// Returns the negamax value of `state` from the side-to-move's perspective
/// within the window `(alpha, beta)`. `depth` is the remaining placement
/// plies; `ply` is the distance from the root (used to index killers).
pub fn negascout<E: LeafEvaluator>(
    ctx: &mut SearchContext<'_, E>,
    state: &GameState,
    mut alpha: i32,
    beta: i32,
    depth: u32,
    ply: usize,
) -> i32 {
    ctx.nodes += 1;

    // --- B6 step 1: terminal (exact, not heuristic) -----------------------
    if state.is_terminal() {
        return terminal_score(state);
    }

    let moves_vec = state.legal_moves();

    // --- B6 step 2: must pass (no placement, opponent can move) -----------
    // Depth is preserved and empties are unchanged; only the side flips.
    if moves_vec.is_empty() {
        let mut passed = state.clone();
        // `apply_move(Pass)` is legal exactly when there is no placement.
        passed
            .apply_move(Move::Pass)
            .expect("pass is legal when there is no legal placement");
        return -negascout(ctx, &passed, -beta, -alpha, depth, ply + 1);
    }

    // --- B6 step 3: horizon leaf ------------------------------------------
    if depth == 0 {
        return ctx.evaluator.eval(state);
    }

    // --- TT probe + window narrowing (design doc §4.3.2) ------------------
    let key = ctx.zobrist.hash(state);
    let alpha_orig = alpha;
    let mut beta = beta;
    let mut tt_move: Option<Move> = None;

    if ctx.config.use_tt
        && let Some(e) = ctx.tt.probe(key)
    {
        tt_move = e.best_move;
        if e.depth >= depth {
            match e.flag {
                Bound::Exact => return e.value,
                Bound::Lower => alpha = alpha.max(e.value),
                Bound::Upper => beta = beta.min(e.value),
            }
            if alpha >= beta {
                return e.value;
            }
        }
    }

    // --- Single ProbCut (design doc §4.3.4 / §4.5 B7) ---------------------
    //
    // Attempted at an *interior* node whose remaining height is exactly
    // `h` (the canonical 8), when ProbCut is enabled and a usable
    // coefficient cell exists for this node's disc-count phase. By
    // construction this is reached only **after** the terminal (step 1),
    // must-pass (step 2) and horizon (step 3) returns, so ProbCut never
    // fires on a terminal or must-pass node. It also runs after the TT
    // narrowing, so it composes with the TT (the probes recurse through
    // the full `negascout`, sharing the same TT/killers). The exact
    // endgame solver runs with a default `SearchConfig` (ProbCut OFF), so
    // ProbCut can never bypass the Phase-3 exact endgame search.
    //
    // ProbCut is intentionally unsound: the depth-`d` probe predicts the
    // depth-`h` value only up to the regression's residual σ, so on a
    // fraction of nodes the returned `beta`/`alpha` differs from the true
    // minimax value. This is the deliberate speed/accuracy trade-off; the
    // correct on-invariant is statistical agreement, never equality.
    {
        let pc = &ctx.config.probcut;
        if pc.enabled && depth == pc.h && pc.d < depth {
            let discs = 64 - state.board.empty_count();
            let params = *pc.params_for_discs(discs);
            if params.is_usable() {
                let (bound_low, bound_high) = probcut_bounds(&params, pc.t, alpha, beta);
                let probe_depth = pc.d;

                // Probe 1: probable fail-high. Null window just below
                // `bound_high`. `bound_high - 1` is safe: `probcut_bounds`
                // clamps to ±INF, so the synthesised window stays in i32.
                if bound_high < INF {
                    let v = negascout(ctx, state, bound_high - 1, bound_high, probe_depth, ply + 1);
                    if v >= bound_high {
                        // Probable β-cut; do not pollute the TT with this
                        // unsound bound (keep the table sound for the
                        // ProbCut-OFF invariants).
                        return beta;
                    }
                }

                // Probe 2: probable fail-low. Null window just above
                // `bound_low`.
                if bound_low > -INF {
                    let v = negascout(ctx, state, bound_low, bound_low + 1, probe_depth, ply + 1);
                    if v <= bound_low {
                        return alpha;
                    }
                }
                // Fall through: normal full-depth-`h` NegaScout below.
            }
        }
    }

    // --- Move ordering: TT move, killers, rest ----------------------------
    let mut moves = moves_vec;
    let killers = if ctx.config.use_killers {
        ctx.killers.killers(ply)
    } else {
        [None, None]
    };
    order_moves(&mut moves, tt_move, killers, ctx.config.use_killers);

    // --- NegaScout move loop ----------------------------------------------
    let mut best = -INF;
    let mut best_move = moves[0];
    let mut b = beta; // scout window upper bound (full window for move 0)

    for (i, &m) in moves.iter().enumerate() {
        let mut child = state.clone();
        child
            .apply_move(m)
            .expect("generated legal move must apply");

        let mut score = -negascout(ctx, &child, -b, -alpha, depth - 1, ply + 1);

        // Re-search with the full window for any non-first move whose scout
        // result lands strictly inside (alpha, beta).
        if alpha < score && score < beta && i != 0 {
            score = -negascout(ctx, &child, -beta, -alpha, depth - 1, ply + 1);
        }

        if score > best {
            best = score;
            best_move = m;
        }
        if score > alpha {
            alpha = score;
        }
        if alpha >= beta {
            // Beta cutoff: the move that fails high becomes a killer.
            if ctx.config.use_killers {
                ctx.killers.record(ply, m);
            }
            break;
        }
        b = alpha + 1; // null window for the remaining moves
    }

    // --- TT store ----------------------------------------------------------
    if ctx.config.use_tt {
        let flag = if best <= alpha_orig {
            Bound::Upper
        } else if best >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        ctx.tt.store(Entry {
            key,
            depth,
            value: best,
            flag,
            best_move: Some(best_move),
        });
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use logistello_eval::DiscDiffEval;
    use othello_core::{Color, Coord};

    #[test]
    fn terminal_score_full_board_black() {
        // All-black full board, Black to move: 2P - 64 = 64.
        let mut s = GameState::standard_8x8();
        for r in 0..8u8 {
            for c in 0..8u8 {
                s.board.set(Coord::new(r, c), Some(Color::Black));
            }
        }
        s.side_to_move = Color::Black;
        s.consecutive_passes = 2;
        assert!(s.is_terminal());
        assert_eq!(terminal_score(&s), 64);
        // From White's POV the same board is -64.
        s.side_to_move = Color::White;
        assert_eq!(terminal_score(&s), -64);
    }

    #[test]
    fn terminal_score_awards_empties_to_winner() {
        // Black 10, White 4, 50 empty -> diff = +6, score = 6 + 50 = 56.
        let mut s = GameState::standard_8x8();
        for r in 0..8u8 {
            for c in 0..8u8 {
                s.board.set(Coord::new(r, c), None);
            }
        }
        for i in 0..10u8 {
            s.board.set(Coord::new(i / 8, i % 8), Some(Color::Black));
        }
        for i in 0..4u8 {
            s.board.set(Coord::new(2, i), Some(Color::White));
        }
        s.side_to_move = Color::Black;
        assert_eq!(terminal_score(&s), 56);
        s.side_to_move = Color::White;
        assert_eq!(terminal_score(&s), -56);
    }

    #[test]
    fn terminal_score_draw_is_zero() {
        let mut s = GameState::standard_8x8();
        for r in 0..8u8 {
            for c in 0..8u8 {
                let col = if c < 4 { Color::Black } else { Color::White };
                s.board.set(Coord::new(r, c), Some(col));
            }
        }
        assert_eq!(terminal_score(&s), 0);
    }

    #[test]
    fn root_search_runs() {
        let s = GameState::standard_8x8();
        let mut tt = TranspositionTable::new();
        let mut killers = KillerTable::new();
        let z = logistello_core::Zobrist::new();
        let mut ctx = SearchContext {
            evaluator: &DiscDiffEval,
            tt: &mut tt,
            killers: &mut killers,
            zobrist: &z,
            config: SearchConfig::default(),
            nodes: 0,
        };
        let v = negascout(&mut ctx, &s, -INF, INF, 3, 0);
        assert!(v.abs() <= 128, "value {v} within disc scale");
        assert!(ctx.nodes > 0);
    }
}
