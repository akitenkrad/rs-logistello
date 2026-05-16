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

/// Basic Othello evaluator (design doc §4.3.3, IAGO / Buro 1994 lineage).
///
/// This is the classic *disc-count + mobility* linear evaluator that
/// preceded Logistello's learned pattern evaluator. It exists so the
/// Phase 3 search/engine pipeline can play full games before the Phase 4
/// regression-trained pattern evaluator replaces it.
///
/// # Definition
///
/// From the **side-to-move's** point of view (`self` = side to move,
/// `opp` = the other side):
///
/// ```text
/// score = w_disc * (P - O) + w_mobility * (mob_self - mob_opp)
/// ```
///
/// where `P` / `O` are the disc counts and `mob_x` is the number of legal
/// *placement* moves available to side `x` on the current board (computed
/// with `Board::legal_moves`, which never mutates shared state — the
/// opponent's mobility is queried directly for the opposite colour, no
/// `side_to_move` flip and no `clone`).
///
/// The construction is **antisymmetric**: swapping the side to move negates
/// both `(P - O)` and `(mob_self - mob_opp)`, hence negates the score, which
/// the search relies on for its minimax-invariance guarantees.
///
/// # Weights
///
/// `w_disc = 1`, `w_mobility = 3`. Rationale: in classic Othello evaluators
/// (IAGO, BILL, early Logistello) raw disc count is a famously poor mid-game
/// signal — the eventual winner is frequently *behind* on discs for most of
/// the game — whereas **mobility** (keeping more moves than the opponent)
/// correlates strongly with winning. Weighting mobility above the disc
/// differential reproduces that classical insight. The exact constants are
/// not load-bearing: this whole evaluator is replaced in Phase 4 by the
/// learned pattern model (design doc §4.4 B1-B3); the values are chosen only
/// to (a) give mobility the dominant mid-game voice and (b) keep the score
/// well inside `±64` on realistic boards so it can never rival an exact
/// terminal/win value produced by the search.
///
/// Scale bound: `|P - O| <= 64` and `|mob_self - mob_opp|` is small in
/// practice; the disc term alone could reach ±64 only on a finished board,
/// where the search already uses the exact terminal score instead of this
/// evaluator, so heuristic leaves stay comfortably dominated by the
/// `terminal_score` / mate sentinels (design doc §4.5 B6).
#[derive(Debug, Clone, Copy)]
pub struct BasicEval {
    /// Weight on the disc differential `P - O`.
    pub w_disc: i32,
    /// Weight on the mobility differential `mob_self - mob_opp`.
    pub w_mobility: i32,
}

impl Default for BasicEval {
    #[inline]
    fn default() -> Self {
        Self {
            w_disc: 1,
            w_mobility: 3,
        }
    }
}

impl LeafEvaluator for BasicEval {
    #[inline]
    fn eval(&self, state: &GameState) -> i32 {
        let me = state.side_to_move;
        let opp = me.opponent();
        debug_assert!(matches!(me, Color::Black | Color::White));

        let p = state.board.count(me) as i32;
        let o = state.board.count(opp) as i32;
        let disc_diff = p - o;

        // `Board::legal_moves(side)` takes the side explicitly and does not
        // mutate; querying it for both colours avoids cloning / flipping the
        // shared `GameState`.
        let mob_self = state.board.legal_moves(me).len() as i32;
        let mob_opp = state.board.legal_moves(opp).len() as i32;
        let mob_diff = mob_self - mob_opp;

        self.w_disc * disc_diff + self.w_mobility * mob_diff
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use othello_core::Coord;

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

    #[test]
    fn basic_eval_default_weights() {
        let b = BasicEval::default();
        assert_eq!(b.w_disc, 1);
        assert_eq!(b.w_mobility, 3);
    }

    #[test]
    fn basic_eval_start_is_zero() {
        // 2 vs 2 discs, 4 vs 4 mobility -> perfectly symmetric -> 0.
        let s = GameState::standard_8x8();
        assert_eq!(BasicEval::default().eval(&s), 0);
    }

    #[test]
    fn basic_eval_antisymmetric() {
        let mut s = GameState::standard_8x8();
        s.apply_move(s.legal_moves()[0]).unwrap();
        let b = BasicEval::default();
        let white_pov = b.eval(&s);
        s.side_to_move = Color::Black;
        let black_pov = b.eval(&s);
        assert_eq!(white_pov, -black_pov);
    }

    #[test]
    fn basic_eval_mobility_dominates_disc_in_midgame() {
        // Constructed: disc diff 0, Black has a mobility edge -> score > 0
        // and equal to the mobility term (disc term is 0).
        let mut t = GameState::standard_8x8();
        for r in 0..8u8 {
            for c in 0..8u8 {
                t.board.set(Coord::new(r, c), None);
            }
        }
        t.board.set(Coord::new(3, 2), Some(Color::Black));
        t.board.set(Coord::new(3, 3), Some(Color::White));
        t.board.set(Coord::new(4, 4), Some(Color::White));
        t.board.set(Coord::new(4, 5), Some(Color::Black));
        t.side_to_move = Color::Black;

        let b = BasicEval::default();
        let p = t.board.count(Color::Black) as i32;
        let o = t.board.count(Color::White) as i32;
        assert_eq!(p - o, 0);
        let mob_self = t.board.legal_moves(Color::Black).len() as i32;
        let mob_opp = t.board.legal_moves(Color::White).len() as i32;
        assert_eq!(b.eval(&t), b.w_mobility * (mob_self - mob_opp));
    }
}
