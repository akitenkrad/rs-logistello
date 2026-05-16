//! Iterative-deepening driver (design doc §4.3.2).
//!
//! Searches depths `1..=max_depth`, reusing the transposition table across
//! iterations so each iteration's TT best moves order the next, deeper one.
//! The killer table is also kept across iterations (it only reorders moves
//! and never changes a value).
//!
//! This module also provides a deliberately dead-simple [`reference_negamax`]
//! (no pruning, no TT, no ordering) that shares the **exact** §4.5 B6
//! pass/terminal semantics and the same `LeafEvaluator`. It is the gold
//! standard the optimised NegaScout is property-tested against.

use logistello_eval::LeafEvaluator;
use othello_core::{GameState, Move};

use crate::alphabeta::{INF, SearchConfig, SearchContext, negascout, terminal_score};
use crate::killer::KillerTable;
use crate::tt::TranspositionTable;

/// Result of an iterative-deepening search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchResult {
    /// Best move at the root (`Move::Pass` if the root must pass).
    pub best_move: Move,
    /// Negamax value from the root side-to-move's perspective.
    pub value: i32,
    /// Depth actually completed (== `max_depth` unless `max_depth == 0`).
    pub depth: u32,
    /// Total nodes visited across **all** deepening iterations.
    pub nodes: u64,
}

/// Picks the best root move at a fixed `depth` by scoring every legal move
/// with [`negascout`] (the root layer is broken out so the chosen move is
/// observable; child scoring uses the full optimised search + shared TT).
///
/// Implements §4.5 B6 at the root: terminal -> exact terminal score with
/// `best_move = Pass`; must-pass -> recurse through a pass at the **same
/// depth**; depth 0 -> static leaf value.
fn search_root<E: LeafEvaluator>(
    ctx: &mut SearchContext<'_, E>,
    state: &GameState,
    depth: u32,
) -> (Move, i32) {
    if state.is_terminal() {
        return (Move::Pass, terminal_score(state));
    }
    let moves = state.legal_moves();
    if moves.is_empty() {
        // Must pass: same depth, empties unchanged, negate child value.
        let mut passed = state.clone();
        passed
            .apply_move(Move::Pass)
            .expect("pass is legal when there is no legal placement");
        let v = -negascout(ctx, &passed, -INF, INF, depth, 1);
        return (Move::Pass, v);
    }
    if depth == 0 {
        return (moves[0], ctx.evaluator.eval(state));
    }

    let mut best = -INF;
    let mut best_move = moves[0];
    let mut alpha = -INF;
    let beta = INF;
    for (i, &m) in moves.iter().enumerate() {
        let mut child = state.clone();
        child.apply_move(m).expect("legal move applies");
        let b = if i == 0 { beta } else { alpha + 1 };
        let mut score = -negascout(ctx, &child, -b, -alpha, depth - 1, 1);
        if i != 0 && alpha < score && score < beta {
            score = -negascout(ctx, &child, -beta, -alpha, depth - 1, 1);
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

/// Iterative-deepening entry point.
///
/// Runs depths `1..=max_depth` (or just depth 0 when `max_depth == 0`),
/// reusing `tt` across iterations. Returns the deepest completed result.
#[allow(clippy::too_many_arguments)]
pub fn search_with<E: LeafEvaluator>(
    root: &GameState,
    max_depth: u32,
    evaluator: &E,
    tt: &mut TranspositionTable,
    killers: &mut KillerTable,
    zobrist: &logistello_core::Zobrist,
    config: SearchConfig,
) -> SearchResult {
    let mut ctx = SearchContext {
        evaluator,
        tt,
        killers,
        zobrist,
        config,
        nodes: 0,
    };

    if max_depth == 0 {
        let (best_move, value) = search_root(&mut ctx, root, 0);
        return SearchResult {
            best_move,
            value,
            depth: 0,
            nodes: ctx.nodes,
        };
    }

    let mut last = SearchResult {
        best_move: Move::Pass,
        value: 0,
        depth: 0,
        nodes: 0,
    };
    for depth in 1..=max_depth {
        let (best_move, value) = search_root(&mut ctx, root, depth);
        last = SearchResult {
            best_move,
            value,
            depth,
            nodes: ctx.nodes,
        };
    }
    last
}

/// Convenience wrapper that allocates a fresh default TT + killer table and a
/// Zobrist hasher, then runs iterative deepening with default config.
#[must_use]
pub fn search<E: LeafEvaluator>(root: &GameState, max_depth: u32, evaluator: &E) -> SearchResult {
    let mut tt = TranspositionTable::new();
    let mut killers = KillerTable::new();
    let zobrist = logistello_core::Zobrist::new();
    search_with(
        root,
        max_depth,
        evaluator,
        &mut tt,
        &mut killers,
        &zobrist,
        SearchConfig::default(),
    )
}

/// Dead-simple reference negamax: no pruning, no TT, no move ordering.
///
/// Shares the **exact** §4.5 B6 pass/terminal semantics with [`negascout`]:
/// 1. terminal -> [`terminal_score`];
/// 2. no legal placement -> pass at the **same depth**, negate;
/// 3. `depth == 0` -> static leaf value;
/// 4. else `max(-negamax(child, depth - 1))` over legal moves.
///
/// This is the gold standard the optimised search is property-tested
/// against; it must never be weakened to make a test pass.
pub fn reference_negamax<E: LeafEvaluator>(state: &GameState, depth: u32, evaluator: &E) -> i32 {
    if state.is_terminal() {
        return terminal_score(state);
    }
    let moves = state.legal_moves();
    if moves.is_empty() {
        let mut passed = state.clone();
        passed
            .apply_move(Move::Pass)
            .expect("pass is legal when there is no legal placement");
        return -reference_negamax(&passed, depth, evaluator);
    }
    if depth == 0 {
        return evaluator.eval(state);
    }
    let mut best = -INF;
    for m in moves {
        let mut child = state.clone();
        child.apply_move(m).expect("legal move applies");
        let v = -reference_negamax(&child, depth - 1, evaluator);
        if v > best {
            best = v;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use logistello_eval::DiscDiffEval;

    #[test]
    fn search_returns_a_legal_root_move() {
        let s = GameState::standard_8x8();
        let r = search(&s, 4, &DiscDiffEval);
        assert_eq!(r.depth, 4);
        assert!(r.nodes > 0);
        assert!(s.legal_moves().contains(&r.best_move));
    }

    #[test]
    fn depth_zero_returns_static_value() {
        let s = GameState::standard_8x8();
        let r = search(&s, 0, &DiscDiffEval);
        assert_eq!(r.depth, 0);
        assert_eq!(r.value, reference_negamax(&s, 0, &DiscDiffEval));
    }

    #[test]
    fn iterative_value_matches_reference_from_start() {
        let s = GameState::standard_8x8();
        for d in 1..=6 {
            let r = search(&s, d, &DiscDiffEval);
            let want = reference_negamax(&s, d, &DiscDiffEval);
            assert_eq!(r.value, want, "depth {d}: {} != {want}", r.value);
        }
    }
}
