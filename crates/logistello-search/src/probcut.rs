//! Single ProbCut selective forward pruning (Buro 1995 ICCA; design doc
//! §4.3.4 + §4.5 B7).
//!
//! ProbCut models the relation between a shallow depth-`d` search value
//! `v_d` and the deep depth-`h` value `v_h` as a linear regression
//!
//! ```text
//! v_h = a · v_d + b + ε,   ε ~ N(0, σ²)
//! ```
//!
//! (design doc §4.5 B7; for Othello `R² > 0.96`). Given the search window
//! `(α, β)` and a per-σ confidence `T = Φ⁻¹(p)`, a deep fail-high
//! `v_h ≥ β` is *probable* (probability `≥ p`) iff
//!
//! ```text
//! v_d ≥ (T·σ + β − b) / a       =: bound_high
//! ```
//!
//! and a deep fail-low `v_h ≤ α` is probable iff
//!
//! ```text
//! v_d ≤ (−T·σ + α − b) / a      =: bound_low
//! ```
//!
//! [`probcut_bounds`] returns `(bound_low, bound_high)` as integers (the
//! search values are integer disc-scale scores). The cut itself is
//! performed inline in [`negascout`](crate::negascout): at an interior
//! node whose remaining height equals `h`, it runs the two null-window
//! depth-`d` probes of the design doc §4.3.4 pseudocode and, on
//! fall-through, continues with the normal full-depth-`h` search.
//!
//! # Why this is *unsound by design*
//!
//! ProbCut is a **probabilistic forward prune**: it discards a subtree when
//! a shallow search makes the deep value *probably* (not certainly) outside
//! the window. The regression has residual variance `σ² > 0`, so for some
//! fraction of nodes the shallow probe mispredicts and ProbCut returns a
//! value that differs from the true depth-`h` minimax value. This is the
//! intended speed/accuracy trade-off (Buro 1995). Consequently the correct
//! invariant for a ProbCut-on search is **statistical agreement on a high
//! fraction of positions with a bounded mean error**, *not* byte-identical
//! minimax equality. With ProbCut **off** (the default) the search is
//! exactly the Phase 2 NegaScout and every prior soundness invariant holds
//! unchanged.

use serde::{Deserialize, Serialize};

/// Regression coefficients for one (disc-count phase, shallow `d`, deep `h`)
/// cell: `v_h = a · v_d + b + ε`, `ε ~ N(0, σ²)` (design doc §4.5 B7).
#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProbCutParams {
    /// Linear slope `a` (≈ 1 for Othello with a good evaluator).
    pub a: f64,
    /// Linear intercept `b`.
    pub b: f64,
    /// Residual standard deviation `σ` (> 0).
    pub sigma: f64,
}

impl ProbCutParams {
    /// Builds a parameter triple.
    #[must_use]
    pub fn new(a: f64, b: f64, sigma: f64) -> Self {
        Self { a, b, sigma }
    }

    /// `true` iff the coefficients are usable for a cut: `a` is finite and
    /// not (numerically) zero (we divide by `a`) and `σ` is a finite
    /// non-negative number. Unfitted / degenerate cells return `false` so
    /// the search simply falls through to the normal expansion.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        self.a.is_finite()
            && self.b.is_finite()
            && self.sigma.is_finite()
            && self.sigma >= 0.0
            && self.a.abs() > 1e-9
    }
}

/// Single-ProbCut configuration (design doc §4.5 B7).
///
/// The canonical single ProbCut is `(d, h) = (4, 8)`, `T = 1.5`. Production
/// disc-count phases split at **36 discs** (`< 36` vs `≥ 36`); a separate
/// `(a, b, σ)` is fitted per phase (this also generalises to the Phase 6
/// cascade). [`Default`] is **ProbCut OFF** with the canonical `(4, 8)`,
/// `T = 1.5` and zeroed (hence *unusable*, see
/// [`ProbCutParams::is_usable`]) coefficients, so a default
/// [`SearchConfig`](crate::SearchConfig) reproduces Phase 2/3 behaviour
/// exactly.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProbCutConfig {
    /// Master switch. When `false`, the search is exactly Phase 2 NegaScout.
    pub enabled: bool,
    /// Per-σ confidence `T = Φ⁻¹(p)` (design doc §4.5 B7: `T = 1.5`).
    pub t: f64,
    /// Shallow check depth `d` (design doc §4.5 B7: `4`).
    pub d: u32,
    /// Deep height `h` at which ProbCut is attempted (design doc §4.5 B7:
    /// `8`).
    pub h: u32,
    /// Coefficients for the `< 36` discs phase.
    pub params_lt36: ProbCutParams,
    /// Coefficients for the `≥ 36` discs phase.
    pub params_ge36: ProbCutParams,
}

/// Disc-count split between the two production ProbCut phases
/// (design doc §4.5 B7: split at 36 discs).
pub const PROBCUT_PHASE_SPLIT_DISCS: u32 = 36;

impl Default for ProbCutConfig {
    fn default() -> Self {
        Self {
            // OFF by default: every Phase 2/3/4 invariant holds unchanged.
            enabled: false,
            // Canonical single-ProbCut parameters (design doc §4.5 B7).
            t: 1.5,
            d: 4,
            h: 8,
            params_lt36: ProbCutParams::default(),
            params_ge36: ProbCutParams::default(),
        }
    }
}

impl ProbCutConfig {
    /// Picks the coefficient cell for a node with `discs` placed discs
    /// (design doc §4.5 B7: split at [`PROBCUT_PHASE_SPLIT_DISCS`]).
    #[must_use]
    pub fn params_for_discs(&self, discs: u32) -> &ProbCutParams {
        if discs < PROBCUT_PHASE_SPLIT_DISCS {
            &self.params_lt36
        } else {
            &self.params_ge36
        }
    }

    /// Reads a [`ProbCutConfig`] from a JSON file (the `probcut-fit`
    /// output). The loaded config has `enabled = true` forced on so a
    /// caller that asked for a params file gets a ProbCut-enabled engine
    /// even if the file was written with `enabled = false`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or is not valid JSON.
    pub fn load_json(path: &std::path::Path) -> std::io::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        let mut cfg: Self = serde_json::from_str(&s)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        cfg.enabled = true;
        Ok(cfg)
    }

    /// Serialises this config to a pretty JSON file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written or serialisation
    /// fails.
    pub fn save_json(&self, path: &std::path::Path) -> std::io::Result<()> {
        let s = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, s)
    }
}

/// The closed-form ProbCut window bounds for one parameter cell
/// (design doc §4.3.4 / §4.5 B7).
///
/// Returns `(bound_low, bound_high)` where, with `T`, `σ`, `a`, `b` and the
/// search window `(alpha, beta)`:
///
/// ```text
/// bound_high = (T·σ + β − b) / a
/// bound_low  = (−T·σ + α − b) / a
/// ```
///
/// # Rounding / clamping (documented contract)
///
/// The search operates on **integer** disc-scale scores, so the real bounds
/// are rounded to integers in the *conservative* direction — making a cut
/// strictly harder to trigger than the exact real-valued test, so rounding
/// never makes ProbCut more aggressive than its statistical model:
///
/// - `bound_high` is rounded **up** (`ceil`): a depth-`d` probe must reach
///   the rounded-up integer to fail high, i.e. `v_d ≥ ⌈bound_high⌉`.
/// - `bound_low` is rounded **down** (`floor`): a probe must drop to the
///   rounded-down integer to fail low, i.e. `v_d ≤ ⌊bound_low⌋`.
///
/// Both are then clamped to `±`[`INF`](crate::INF) so the synthesised
/// null windows are always representable `i32` values and never overflow
/// (`bound_high − 1` / `bound_low + 1` stay in range).
///
/// A negative slope `a < 0` flips the inequality directions; the division
/// by `a` already handles the sign, and the ceil/floor choice keeps the
/// *conservative* rounding meaning (it is applied to the post-division real
/// value). Callers must only invoke a cut when
/// [`ProbCutParams::is_usable`] (so `a` is never ~0).
#[must_use]
pub fn probcut_bounds(p: &ProbCutParams, t: f64, alpha: i32, beta: i32) -> (i32, i32) {
    let ts = t * p.sigma;
    let hi_real = (ts + f64::from(beta) - p.b) / p.a;
    let lo_real = (-ts + f64::from(alpha) - p.b) / p.a;

    let clamp = |x: f64| -> i32 {
        if !x.is_finite() {
            // Degenerate: push the bound out of reach so no cut fires.
            return if x.is_sign_negative() {
                -(crate::INF)
            } else {
                crate::INF
            };
        }
        x.clamp(f64::from(-crate::INF), f64::from(crate::INF)) as i32
    };

    // Conservative integer rounding (see doc comment): high rounds up,
    // low rounds down. With a<0 the divisions already swapped the roles, so
    // we still take ceil for the high bound and floor for the low bound of
    // the *post-division* reals to stay on the safe side.
    let bound_high = clamp(hi_real.ceil());
    let bound_low = clamp(lo_real.floor());
    (bound_low, bound_high)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_match_closed_form_identity_coeffs() {
        // a=1, b=0, sigma=0: bound_high = beta, bound_low = alpha exactly.
        let p = ProbCutParams::new(1.0, 0.0, 0.0);
        let (lo, hi) = probcut_bounds(&p, 1.5, -10, 20);
        assert_eq!(hi, 20);
        assert_eq!(lo, -10);
    }

    #[test]
    fn bounds_match_closed_form_hand_values() {
        // a=2, b=3, sigma=4, T=1.5, alpha=-8, beta=12.
        // T*sigma = 6.
        // hi = (6 + 12 - 3) / 2 = 15/2 = 7.5  -> ceil  -> 8
        // lo = (-6 + (-8) - 3) / 2 = -17/2 = -8.5 -> floor -> -9
        let p = ProbCutParams::new(2.0, 3.0, 4.0);
        let (lo, hi) = probcut_bounds(&p, 1.5, -8, 12);
        assert_eq!(hi, 8, "ceil((6+12-3)/2) = ceil(7.5) = 8");
        assert_eq!(lo, -9, "floor((-6-8-3)/2) = floor(-8.5) = -9");
    }

    #[test]
    fn bound_high_monotone_nondecreasing_in_t() {
        // Larger T (more confidence demanded) => harder to fail high =>
        // bound_high must be (weakly) larger; bound_low (weakly) smaller.
        let p = ProbCutParams::new(1.0, 0.0, 5.0);
        let mut prev_hi = i32::MIN;
        let mut prev_lo = i32::MAX;
        for k in 0..20 {
            let t = 0.5 + 0.25 * f64::from(k);
            let (lo, hi) = probcut_bounds(&p, t, -30, 30);
            assert!(hi >= prev_hi, "bound_high non-decreasing in T");
            assert!(lo <= prev_lo, "bound_low non-increasing in T");
            prev_hi = hi;
            prev_lo = lo;
        }
    }

    #[test]
    fn symmetric_construction_sanity() {
        // a=1, b=0, symmetric window (-W, +W): bounds are symmetric around 0
        // up to the conservative ceil/floor (which can only widen by 1).
        let p = ProbCutParams::new(1.0, 0.0, 3.0);
        let (lo, hi) = probcut_bounds(&p, 1.4, -25, 25);
        // hi = 25 + 1.4*3 = 29.2 -> 30 ; lo = -25 - 4.2 = -29.2 -> -30.
        assert_eq!(hi, 30);
        assert_eq!(lo, -30);
        assert_eq!(hi, -lo, "perfectly symmetric for a=1,b=0,(-W,+W)");
    }

    #[test]
    fn negative_slope_handled_without_panic() {
        // a < 0 must not panic and must stay clamped in i32 range.
        let p = ProbCutParams::new(-1.5, 2.0, 4.0);
        let (lo, hi) = probcut_bounds(&p, 1.5, -20, 20);
        let range = -crate::INF..=crate::INF;
        assert!(range.contains(&lo));
        assert!(range.contains(&hi));
    }

    #[test]
    fn degenerate_params_are_unusable() {
        assert!(!ProbCutParams::default().is_usable(), "all-zero unusable");
        assert!(!ProbCutParams::new(0.0, 1.0, 1.0).is_usable(), "a~0");
        assert!(!ProbCutParams::new(f64::NAN, 0.0, 1.0).is_usable(), "NaN a");
        assert!(
            ProbCutParams::new(1.0, 0.0, 2.0).is_usable(),
            "normal fitted params usable"
        );
    }

    #[test]
    fn config_default_is_off_and_canonical() {
        let c = ProbCutConfig::default();
        assert!(!c.enabled, "ProbCut OFF by default (Phase 2/3 preserved)");
        assert_eq!(c.t, 1.5);
        assert_eq!(c.d, 4);
        assert_eq!(c.h, 8);
        assert!(!c.params_lt36.is_usable());
        assert!(!c.params_ge36.is_usable());
    }

    #[test]
    fn params_for_discs_splits_at_36() {
        let c = ProbCutConfig {
            params_lt36: ProbCutParams::new(1.0, 0.0, 1.0),
            params_ge36: ProbCutParams::new(2.0, 0.0, 1.0),
            ..ProbCutConfig::default()
        };
        assert_eq!(c.params_for_discs(35).a, 1.0);
        assert_eq!(c.params_for_discs(36).a, 2.0);
        assert_eq!(c.params_for_discs(0).a, 1.0);
        assert_eq!(c.params_for_discs(64).a, 2.0);
    }

    #[test]
    fn config_json_roundtrips() {
        let mut c = ProbCutConfig {
            enabled: false,
            t: 1.3,
            d: 4,
            h: 8,
            params_lt36: ProbCutParams::new(0.97, -0.2, 4.1),
            params_ge36: ProbCutParams::new(1.02, 0.5, 2.7),
        };
        let dir = std::env::temp_dir();
        let path = dir.join(format!("pc_cfg_test_{}.json", std::process::id()));
        c.save_json(&path).unwrap();
        let back = ProbCutConfig::load_json(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        // load_json forces enabled = true; everything else round-trips.
        assert!(back.enabled);
        c.enabled = true;
        assert_eq!(back, c);
    }
}
