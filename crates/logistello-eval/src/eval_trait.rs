//! Leaf-evaluator abstraction shared between the search engine and the
//! (Phase 3/4) pattern evaluator.
//!
//! The search engine (`logistello-search`) calls the leaf evaluator at
//! horizon nodes (`depth == 0`). The contract is deliberately minimal so the
//! Phase 2 search can be exercised with a trivial disc-difference evaluator
//! while the real pattern/regression evaluator is built in Phase 3/4.

use othello_core::{Color, GameState};

/// A static (depth-0) position evaluator.
///
/// # Contract
///
/// - The returned score is from the **side-to-move's** point of view: a
///   larger value means the position is better for `state.side_to_move`.
/// - The scale is **disc-difference compatible**: it must stay well inside
///   `±(64 + 64)` so that exact terminal scores and mate-distance sentinels
///   produced by the search dominate heuristic leaf values
///   (design doc §4.5 B6).
/// - It must be a pure function of the position (no interior mutability that
///   would make repeated calls disagree); the search relies on this for its
///   minimax-invariance guarantees.
pub trait LeafEvaluator {
    /// Scores `state` from the side-to-move's perspective.
    fn eval(&self, state: &GameState) -> i32;
}

/// Trivial leaf evaluator: signed disc differential `P - O` from the
/// side-to-move's perspective.
///
/// This is for **testing and Phase 2 only**. The real Logistello-2 evaluator
/// (Edax-style pattern features + linear regression on the final disc
/// difference) arrives in Phase 3/4 (design doc §4.3.3 / §4.4 B1-B3).
#[derive(Debug, Clone, Copy, Default)]
pub struct DiscDiffEval;

impl LeafEvaluator for DiscDiffEval {
    #[inline]
    fn eval(&self, state: &GameState) -> i32 {
        let me = state.side_to_move;
        let opp = me.opponent();
        debug_assert!(matches!(me, Color::Black | Color::White));
        state.board.count(me) as i32 - state.board.count(opp) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_position_is_even() {
        let s = GameState::standard_8x8();
        // 2 vs 2 at the start, whoever is to move.
        assert_eq!(DiscDiffEval.eval(&s), 0);
    }

    #[test]
    fn perspective_is_side_to_move() {
        let mut s = GameState::standard_8x8();
        // Black plays the first move; afterwards White is to move and is
        // behind on discs, so the score (White POV) must be negative.
        s.apply_move(s.legal_moves()[0]).unwrap();
        assert_eq!(s.side_to_move, Color::White);
        assert!(DiscDiffEval.eval(&s) < 0);
        // Flip the side to move: same board, Black POV => positive.
        s.side_to_move = Color::Black;
        assert!(DiscDiffEval.eval(&s) > 0);
    }
}
