//! Elo estimation vs Edax (design doc `Logistello.md` §4.3.8
//! `elo_vs_edax_level_N`, §5 objective 4; Phase 9b).
//!
//! Pure scoring/aggregation logic, kept separate from the (heavy, real)
//! Edax-driving game loop so it is fully unit-tested with synthetic
//! results.
//!
//! ## Match scoring
//!
//! Each game contributes a *score for our engine*: win = 1, draw = 0.5,
//! loss = 0. We always play **both colours** an equal number of times at a
//! level (colour-balanced) so first-move advantage cancels. `p` = our mean
//! score over the games at that level.
//!
//! ## Elo delta
//!
//! `Δ = -400 · log10(1/p − 1)` (the inverse of the logistic Elo
//! expectation `p = 1 / (1 + 10^(−Δ/400))`). `p` is clamped to the open
//! interval `(0, 1)` so `Δ` stays finite at a clean sweep (a 0/all result
//! gives a large but finite bound, not `±∞`). The sign convention: `Δ > 0`
//! means our engine is estimated *stronger* than this Edax level.
//!
//! NOTE (honest caveat, design doc §5): an Edax `-level N` maps to playing
//! strength only *qualitatively*; the absolute Elo of a level is not
//! published, so `Δ` is "our engine relative to *this Edax level*", a
//! within-experiment comparison, not an absolute FFO/portable Elo.

/// Outcome of one game from our engine's point of view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Our engine won.
    Win,
    /// The game was a draw.
    Draw,
    /// Our engine lost.
    Loss,
}

impl Outcome {
    /// Our score for this game (win 1, draw 0.5, loss 0).
    #[must_use]
    pub fn score(self) -> f64 {
        match self {
            Outcome::Win => 1.0,
            Outcome::Draw => 0.5,
            Outcome::Loss => 0.0,
        }
    }

    /// Decides the outcome from final disc counts (our discs vs Edax's).
    #[must_use]
    pub fn from_discs(our_discs: u32, edax_discs: u32) -> Self {
        match our_discs.cmp(&edax_discs) {
            std::cmp::Ordering::Greater => Outcome::Win,
            std::cmp::Ordering::Less => Outcome::Loss,
            std::cmp::Ordering::Equal => Outcome::Draw,
        }
    }
}

/// Smallest/largest score probability used to keep `Δ` finite when a level
/// is a clean sweep. With `n` games the natural resolution is `1/(2n)`; we
/// use a fixed small epsilon that is safely below any realistic
/// per-level resolution and keeps the bound interpretable.
pub const P_EPS: f64 = 1.0e-4;

/// Estimated Elo delta from our mean score `p` against an opponent:
/// `Δ = -400·log10(1/p − 1)`, with `p` clamped into `(0, 1)`.
///
/// `p = 0.5` → `0`; `p > 0.5` → positive (we are stronger); a clean sweep
/// (`p = 1` / `p = 0`) yields a large but finite ±bound (≈ ±1600 at the
/// default `P_EPS`), never `±∞`.
#[must_use]
pub fn elo_delta(p: f64) -> f64 {
    let pc = p.clamp(P_EPS, 1.0 - P_EPS);
    -400.0 * (1.0 / pc - 1.0).log10()
}

/// Aggregated result for one Edax level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LevelResult {
    /// Edax `-level N`.
    pub level: u32,
    /// Games played at this level (colour-balanced).
    pub games: u32,
    /// Our wins.
    pub wins: u32,
    /// Draws.
    pub draws: u32,
    /// Our losses.
    pub losses: u32,
}

impl LevelResult {
    /// Empty accumulator for `level`.
    #[must_use]
    pub fn new(level: u32) -> Self {
        Self {
            level,
            games: 0,
            wins: 0,
            draws: 0,
            losses: 0,
        }
    }

    /// Folds one game outcome in.
    pub fn record(&mut self, o: Outcome) {
        self.games += 1;
        match o {
            Outcome::Win => self.wins += 1,
            Outcome::Draw => self.draws += 1,
            Outcome::Loss => self.losses += 1,
        }
    }

    /// Our mean score `p ∈ [0, 1]` (0 when no games — caller guards).
    #[must_use]
    pub fn score_rate(&self) -> f64 {
        if self.games == 0 {
            0.0
        } else {
            (f64::from(self.wins) + 0.5 * f64::from(self.draws)) / f64::from(self.games)
        }
    }

    /// Win rate `wins / games` (draws excluded), in `[0, 1]`.
    #[must_use]
    pub fn win_rate(&self) -> f64 {
        if self.games == 0 {
            0.0
        } else {
            f64::from(self.wins) / f64::from(self.games)
        }
    }

    /// Estimated Elo delta vs this Edax level (see [`elo_delta`]).
    #[must_use]
    pub fn elo_delta(&self) -> f64 {
        elo_delta(self.score_rate())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elo_delta_is_zero_at_even_and_symmetric() {
        assert!((elo_delta(0.5)).abs() < 1e-9);
        // p and 1-p give equal-magnitude opposite deltas.
        for &p in &[0.6, 0.75, 0.9, 0.99] {
            let a = elo_delta(p);
            let b = elo_delta(1.0 - p);
            assert!((a + b).abs() < 1e-6, "antisymmetry at p={p}: {a} vs {b}");
            assert!(a > 0.0, "p>0.5 must be positive");
        }
    }

    #[test]
    fn elo_delta_matches_known_logistic_values() {
        // p = 1/(1+10^(-D/400)) inverted: D=200 -> p≈0.7597.
        let p = 1.0 / (1.0 + 10f64.powf(-200.0 / 400.0));
        assert!((elo_delta(p) - 200.0).abs() < 1e-6);
        // D = -100 -> p ≈ 0.3599.
        let p2 = 1.0 / (1.0 + 10f64.powf(100.0 / 400.0));
        assert!((elo_delta(p2) + 100.0).abs() < 1e-6);
    }

    #[test]
    fn elo_delta_is_clamped_finite_at_clean_sweeps() {
        let hi = elo_delta(1.0);
        let lo = elo_delta(0.0);
        assert!(hi.is_finite() && lo.is_finite(), "must be finite, not ±∞");
        assert!(hi > 0.0 && lo < 0.0);
        assert!((hi + lo).abs() < 1e-6, "clamp is symmetric");
        // Out-of-range inputs are clamped, never NaN/inf.
        assert!(elo_delta(2.0).is_finite());
        assert!(elo_delta(-1.0).is_finite());
        // At P_EPS the bound is ~1600 Elo.
        assert!(
            (hi - 1600.0).abs() < 50.0,
            "approx ±1600 at P_EPS, got {hi}"
        );
    }

    #[test]
    fn outcome_score_and_from_discs() {
        assert_eq!(Outcome::Win.score(), 1.0);
        assert_eq!(Outcome::Draw.score(), 0.5);
        assert_eq!(Outcome::Loss.score(), 0.0);
        assert_eq!(Outcome::from_discs(40, 24), Outcome::Win);
        assert_eq!(Outcome::from_discs(20, 44), Outcome::Loss);
        assert_eq!(Outcome::from_discs(32, 32), Outcome::Draw);
    }

    #[test]
    fn level_result_aggregates_both_colors_correctly() {
        let mut r = LevelResult::new(3);
        // 4 games: 3 wins, 1 draw, 0 losses (both colours folded the same).
        r.record(Outcome::Win);
        r.record(Outcome::Win);
        r.record(Outcome::Draw);
        r.record(Outcome::Win);
        assert_eq!(r.games, 4);
        assert_eq!(r.wins, 3);
        assert_eq!(r.draws, 1);
        assert_eq!(r.losses, 0);
        // score = (3 + 0.5*1)/4 = 0.875
        assert!((r.score_rate() - 0.875).abs() < 1e-12);
        assert!((r.win_rate() - 0.75).abs() < 1e-12);
        assert!(r.elo_delta() > 0.0);

        // A losing level -> negative delta.
        let mut l = LevelResult::new(20);
        l.record(Outcome::Loss);
        l.record(Outcome::Loss);
        l.record(Outcome::Win);
        l.record(Outcome::Loss);
        assert!((l.score_rate() - 0.25).abs() < 1e-12);
        assert!(l.elo_delta() < 0.0);

        // Empty -> guarded (0 rate, finite delta).
        let e = LevelResult::new(1);
        assert_eq!(e.score_rate(), 0.0);
        assert!(e.elo_delta().is_finite());
    }
}
