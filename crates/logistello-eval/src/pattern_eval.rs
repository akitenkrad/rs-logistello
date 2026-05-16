//! [`PatternEval`] — the Edax-準拠 pattern/regression leaf evaluator
//! (design doc §4.3.3, §4.4 B1/B2/B4).
//!
//! For a position it (1) picks the [`stage`](crate::stage::stage) from the
//! disc count (B2), (2) computes the 47 raw feature keys from the
//! side-to-move's point of view (B1), (3) resolves each key to its canonical
//! shared weight slot via the Edax pack table (B4), and (4) sums the per-stage
//! canonical weights. Weights are in **1/128-disc units**, so the leaf score
//! is `round(sum / 128)` with Edax's bias rounding (`midgame.c:36-44`):
//! `s = (sum>0 ? sum+64 : sum-64) / 128`, then clamped to `±63`
//! (`SCORE_MIN+1 .. SCORE_MAX-1`, `const.h:61-64`) so a heuristic leaf can
//! never rival an exact terminal/mate value (design doc §4.5 B6, matching the
//! [`LeafEvaluator`] scale contract).

use crate::eval_trait::LeafEvaluator;
use crate::pattern::{FEATURES, feature_keys};
use crate::stage::stage_for_state;
use crate::weights::EvalWeights;
use othello_core::GameState;

/// Pattern-feature linear evaluator (Logistello-2 / Edax style).
#[derive(Debug, Clone)]
pub struct PatternEval {
    weights: EvalWeights,
}

impl PatternEval {
    /// Builds an evaluator from a learned weight set.
    #[must_use]
    pub fn new(weights: EvalWeights) -> Self {
        Self { weights }
    }

    /// All-zero evaluator: every position evaluates to exactly `0`
    /// (testable identity element).
    #[must_use]
    pub fn zeros() -> Self {
        Self {
            weights: EvalWeights::zeros(),
        }
    }

    /// Borrow the underlying weights.
    #[must_use]
    pub fn weights(&self) -> &EvalWeights {
        &self.weights
    }

    /// Raw 1/128-disc-unit weight sum (before rounding) for `state`, from the
    /// side-to-move's point of view. Exposed for tests / diagnostics.
    #[must_use]
    pub fn raw_sum(&self, state: &GameState) -> i64 {
        let stage = stage_for_state(state);
        let keys = feature_keys(&state.board, state.side_to_move);
        let mut sum = 0i64;
        for (i, def) in FEATURES.iter().enumerate() {
            sum += i64::from(self.weights.lookup(stage, def.ty, keys[i]));
        }
        sum
    }
}

impl LeafEvaluator for PatternEval {
    #[inline]
    fn eval(&self, state: &GameState) -> i32 {
        let sum = self.raw_sum(state);
        // Edax midgame.c:36-44 bias rounding toward larger magnitude.
        let biased = if sum > 0 { sum + 64 } else { sum - 64 };
        let score = (biased / 128) as i32;
        score.clamp(-63, 63)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use othello_core::{Color, Move};
    use rand::seq::SliceRandom;
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    fn random_position(rng: &mut ChaCha20Rng, plies: usize) -> GameState {
        let mut s = GameState::standard_8x8();
        let mut made = 0;
        while made < plies {
            if s.is_terminal() {
                break;
            }
            let moves = s.legal_moves();
            if moves.is_empty() {
                s.apply_move(Move::Pass).expect("pass legal");
                continue;
            }
            let m = *moves.choose(rng).expect("non-empty");
            s.apply_move(m).expect("legal");
            made += 1;
        }
        s
    }

    #[test]
    fn zeros_evaluates_to_zero_everywhere() {
        let ev = PatternEval::zeros();
        let mut rng = ChaCha20Rng::seed_from_u64(0xC0FFEE);
        for _ in 0..200 {
            let plies = (rng.next_u32() % 55) as usize;
            let s = random_position(&mut rng, plies);
            assert_eq!(ev.eval(&s), 0, "zeros must eval to 0");
            assert_eq!(ev.raw_sum(&s), 0);
        }
    }

    #[test]
    fn const_term_is_side_independent() {
        // The constant feature key is always 0 regardless of side, so a
        // pure-constant weight set yields the same raw sum after a side flip.
        let mut w = EvalWeights::zeros();
        for s in 0..crate::stage::N_STAGES {
            w.weights_mut(s, crate::pattern::PatternType::Const)[0] = 500;
        }
        let ev = PatternEval::new(w);
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        for _ in 0..50 {
            let plies = (rng.next_u32() % 50) as usize;
            let mut s = random_position(&mut rng, plies);
            let a = ev.raw_sum(&s);
            s.side_to_move = match s.side_to_move {
                Color::Black => Color::White,
                Color::White => Color::Black,
            };
            let b = ev.raw_sum(&s);
            assert_eq!(a, b, "const term is side-independent (key always 0)");
        }
    }
}
