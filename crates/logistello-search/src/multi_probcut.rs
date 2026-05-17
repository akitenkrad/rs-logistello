//! Multi-ProbCut cascade (Buro 1997; design doc §4.3.5 + §4.5 B7).
//!
//! Multi-ProbCut generalises single ProbCut ([`crate::probcut`]) by
//! cascading **several** shallow probes per interior node. For a node whose
//! remaining height is `h`, [`MPC_CASCADE`] gives an ordered list of check
//! depths `d₁ [, d₂]` (the canonical B7 table). [`MultiProbCut`] tries each
//! stage shallow→deep: it runs the *same* single-ProbCut primitive
//! ([`crate::probcut::probcut_bounds`] + the two null-window probes) for
//! `(node, α, β, h, dᵢ, T, aᵢ, bᵢ, σᵢ)` and, the moment any stage's probe
//! predicts the deep value is out of `(α, β)`, returns `β` (probable
//! fail-high) or `α` (probable fail-low). If every stage falls through it
//! continues with the normal full-depth-`h` NegaScout (design doc §4.3.5
//! pseudocode).
//!
//! Each `(disc-phase, h, d)` cell has an independent regression triple
//! `(a, b, σ)`. The production threshold is **2-phase** (design doc §4.5 B7,
//! canonical): discs `< 36 → T = 1.0`; discs `≥ 36 → T = 1.4`.
//!
//! # Why this is *strictly looser* than single ProbCut (unsound by design)
//!
//! ProbCut is already a probabilistic forward prune (residual `σ² > 0`, so
//! some nodes mispredict). Multi-ProbCut compounds **several** such cuts per
//! node and uses the more aggressive production thresholds (`1.0` / `1.4`,
//! both below the single-ProbCut `1.5`). It therefore visits fewer nodes but
//! agrees with the true depth-`h` minimax value on a *lower* (still high)
//! fraction of positions than single ProbCut. The correct on-invariant is
//! **statistical agreement on a high fraction with a bounded mean error**,
//! never byte-identical equality. With MPC **off** (the default) the search
//! is byte-identical to the Phase 2/5 NegaScout and every prior soundness
//! invariant holds unchanged. MPC never fires at a terminal / must-pass node
//! (it is reached only after the §4.5 B6 returns) and the Phase-3 exact
//! endgame solver runs with a default config (MPC OFF), so MPC can never
//! bypass exact endgame play.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::probcut::PROBCUT_PHASE_SPLIT_DISCS;

/// The canonical Multi-ProbCut cascade (design doc §4.5 B7, Table 2),
/// verbatim. Each entry is `(h, d₁, Option<d₂>)` where `h` is the search
/// height and `d₁` (`d₂`) the first (second) shallow check depth (`–` = no
/// second stage).
///
/// | h  | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
/// |----|---|---|---|---|---|---|---|----|----|----|----|
/// | d₁ | 1 | 2 | 1 | 2 | 3 | 4 | 3 | 4  | 3  | 4  | 5  |
/// | d₂ | – | – | – | – | – | – | 5 | 6  | 5  | –  | –  |
pub const MPC_CASCADE: [(u32, u32, Option<u32>); 11] = [
    (3, 1, None),
    (4, 2, None),
    (5, 1, None),
    (6, 2, None),
    (7, 3, None),
    (8, 4, None),
    (9, 3, Some(5)),
    (10, 4, Some(6)),
    (11, 3, Some(5)),
    (12, 4, None),
    (13, 5, None),
];

// Pre-materialised shallow→deep stage slices for every cascade height, so
// `stages_for_height` can hand back a `&'static [u32]` (no allocation, the
// design doc §4.3.5 `StagesForDepth`). Indexed by `h - 3`.
static STAGES_H3: [u32; 1] = [1];
static STAGES_H4: [u32; 1] = [2];
static STAGES_H5: [u32; 1] = [1];
static STAGES_H6: [u32; 1] = [2];
static STAGES_H7: [u32; 1] = [3];
static STAGES_H8: [u32; 1] = [4];
static STAGES_H9: [u32; 2] = [3, 5];
static STAGES_H10: [u32; 2] = [4, 6];
static STAGES_H11: [u32; 2] = [3, 5];
static STAGES_H12: [u32; 1] = [4];
static STAGES_H13: [u32; 1] = [5];
static STAGES_EMPTY: [u32; 0] = [];

/// The cascade check depths for search height `h`, ordered shallow→deep
/// (`d₁` then `d₂`), per design doc §4.3.5 `StagesForDepth` / §4.5 B7.
///
/// Returns an empty slice when `h` is outside `3..=13` (no cascade defined →
/// the caller does a normal full-depth search). E.g. `h = 9 → [3, 5]`,
/// `h = 8 → [4]`, `h = 2 → []`.
#[must_use]
pub fn stages_for_height(h: u32) -> &'static [u32] {
    match h {
        3 => &STAGES_H3,
        4 => &STAGES_H4,
        5 => &STAGES_H5,
        6 => &STAGES_H6,
        7 => &STAGES_H7,
        8 => &STAGES_H8,
        9 => &STAGES_H9,
        10 => &STAGES_H10,
        11 => &STAGES_H11,
        12 => &STAGES_H12,
        13 => &STAGES_H13,
        _ => &STAGES_EMPTY,
    }
}

/// Disc-count phase for the production 2-phase MPC thresholds / coefficient
/// cells (design doc §4.5 B7: split at [`PROBCUT_PHASE_SPLIT_DISCS`] = 36).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DiscPhase {
    /// Fewer than 36 discs placed (production `T = 1.0`).
    Lt36,
    /// 36 or more discs placed (production `T = 1.4`).
    Ge36,
}

impl DiscPhase {
    /// The phase for a node with `discs` placed discs (design doc §4.5 B7).
    #[must_use]
    pub fn for_discs(discs: u32) -> Self {
        if discs < PROBCUT_PHASE_SPLIT_DISCS {
            DiscPhase::Lt36
        } else {
            DiscPhase::Ge36
        }
    }
}

/// Regression coefficients for one Multi-ProbCut `(disc-phase, h, d)` cell:
/// `v_h = a · v_d + b + ε`, `ε ~ N(0, σ²)` (design doc §4.5 B7).
#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MpcStageParams {
    /// Linear slope `a` (≈ 1 for Othello with a good evaluator).
    pub a: f64,
    /// Linear intercept `b`.
    pub b: f64,
    /// Residual standard deviation `σ` (> 0).
    pub sigma: f64,
}

impl MpcStageParams {
    /// Builds a parameter triple.
    #[must_use]
    pub fn new(a: f64, b: f64, sigma: f64) -> Self {
        Self { a, b, sigma }
    }

    /// `true` iff usable for a cut: `a` finite and not ~0 (we divide by it),
    /// `σ` finite and non-negative. Unfitted / degenerate cells return
    /// `false` so the stage is skipped (search falls through).
    #[must_use]
    pub fn is_usable(&self) -> bool {
        self.a.is_finite()
            && self.b.is_finite()
            && self.sigma.is_finite()
            && self.sigma >= 0.0
            && self.a.abs() > 1e-9
    }
}

/// JSON map key `"phase:h:d"` for a coefficient cell. A flat string key keeps
/// the serialised config a plain JSON object (a `BTreeMap` of tuple keys is
/// not representable in JSON).
fn cell_key(phase: DiscPhase, h: u32, d: u32) -> String {
    let p = match phase {
        DiscPhase::Lt36 => "lt36",
        DiscPhase::Ge36 => "ge36",
    };
    format!("{p}:{h}:{d}")
}

/// Multi-ProbCut configuration (design doc §4.3.5 / §4.5 B7).
///
/// [`Default`] is **MPC OFF** with the canonical production thresholds
/// (`t_lt36 = 1.0`, `t_ge36 = 1.4`) and an empty coefficient map, so a
/// default [`SearchConfig`](crate::SearchConfig) reproduces Phase 2/5
/// behaviour exactly. Coefficients are indexed by `(DiscPhase, h, d)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiProbCutConfig {
    /// Master switch. When `false` the search is exactly the Phase 2/5
    /// NegaScout (single ProbCut, if any, still applies independently).
    pub enabled: bool,
    /// Production threshold `T` for the discs `< 36` phase (design doc
    /// §4.5 B7: `1.0`).
    pub t_lt36: f64,
    /// Production threshold `T` for the discs `≥ 36` phase (design doc
    /// §4.5 B7: `1.4`).
    pub t_ge36: f64,
    /// Per-cell regression coefficients keyed by `"phase:h:d"`.
    params: BTreeMap<String, MpcStageParams>,
}

impl Default for MultiProbCutConfig {
    fn default() -> Self {
        Self {
            // OFF by default: every Phase 2/3/4/5 invariant holds unchanged.
            enabled: false,
            // Canonical 2-phase production thresholds (design doc §4.5 B7).
            t_lt36: 1.0,
            t_ge36: 1.4,
            params: BTreeMap::new(),
        }
    }
}

impl MultiProbCutConfig {
    /// Sets the `(a, b, σ)` cell for `(phase, h, d)`.
    pub fn set_params(&mut self, phase: DiscPhase, h: u32, d: u32, p: MpcStageParams) {
        self.params.insert(cell_key(phase, h, d), p);
    }

    /// The fitted coefficients for `(phase, h, d)`, or `None` if that cell
    /// was never fitted (→ the stage is skipped, search falls through).
    #[must_use]
    pub fn params_for(&self, phase: DiscPhase, h: u32, d: u32) -> Option<&MpcStageParams> {
        self.params.get(&cell_key(phase, h, d))
    }

    /// The production threshold `T` for a node with `discs` placed discs
    /// (design doc §4.5 B7: `< 36 → t_lt36`, `≥ 36 → t_ge36`).
    #[must_use]
    pub fn t_for_discs(&self, discs: u32) -> f64 {
        match DiscPhase::for_discs(discs) {
            DiscPhase::Lt36 => self.t_lt36,
            DiscPhase::Ge36 => self.t_ge36,
        }
    }

    /// Number of fitted `(phase, h, d)` cells (diagnostics / tests).
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.params.len()
    }

    /// Reads a [`MultiProbCutConfig`] from a JSON file (the `probcut-fit
    /// --mpc-cascade` output). The loaded config has `enabled = true` forced
    /// on so a caller that asked for a params file gets an MPC-enabled
    /// engine even if the file was written with `enabled = false`.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cascade_is_the_b7_table() {
        assert_eq!(MPC_CASCADE.len(), 11);
        // Spot-check the canonical entries.
        assert_eq!(MPC_CASCADE[0], (3, 1, None));
        assert_eq!(MPC_CASCADE[6], (9, 3, Some(5)));
        assert_eq!(MPC_CASCADE[7], (10, 4, Some(6)));
        assert_eq!(MPC_CASCADE[8], (11, 3, Some(5)));
        assert_eq!(MPC_CASCADE[10], (13, 5, None));
        // Heights are 3..=13 contiguous.
        for (i, &(h, _, _)) in MPC_CASCADE.iter().enumerate() {
            assert_eq!(h, 3 + i as u32);
        }
    }

    #[test]
    fn stages_for_height_matches_cascade() {
        for &(h, d1, d2) in &MPC_CASCADE {
            let st = stages_for_height(h);
            match d2 {
                Some(d2) => assert_eq!(st, &[d1, d2], "h={h}"),
                None => assert_eq!(st, &[d1], "h={h}"),
            }
        }
        for h in [0u32, 1, 2, 14, 50] {
            assert!(stages_for_height(h).is_empty(), "h={h} out of range");
        }
    }

    #[test]
    fn disc_phase_splits_at_36() {
        assert_eq!(DiscPhase::for_discs(0), DiscPhase::Lt36);
        assert_eq!(DiscPhase::for_discs(35), DiscPhase::Lt36);
        assert_eq!(DiscPhase::for_discs(36), DiscPhase::Ge36);
        assert_eq!(DiscPhase::for_discs(64), DiscPhase::Ge36);
    }

    #[test]
    fn default_off_canonical_thresholds() {
        let c = MultiProbCutConfig::default();
        assert!(!c.enabled);
        assert!((c.t_lt36 - 1.0).abs() < 1e-12);
        assert!((c.t_ge36 - 1.4).abs() < 1e-12);
        assert_eq!(c.cell_count(), 0);
        assert!((c.t_for_discs(35) - 1.0).abs() < 1e-12);
        assert!((c.t_for_discs(36) - 1.4).abs() < 1e-12);
    }

    #[test]
    fn params_roundtrip_and_lookup() {
        let mut c = MultiProbCutConfig::default();
        c.set_params(DiscPhase::Lt36, 9, 3, MpcStageParams::new(1.01, -0.2, 4.0));
        c.set_params(DiscPhase::Ge36, 9, 5, MpcStageParams::new(0.98, 0.3, 2.5));
        assert_eq!(c.cell_count(), 2);
        assert_eq!(c.params_for(DiscPhase::Lt36, 9, 3).unwrap().a, 1.01);
        assert_eq!(c.params_for(DiscPhase::Ge36, 9, 5).unwrap().sigma, 2.5);
        assert!(c.params_for(DiscPhase::Lt36, 9, 5).is_none());

        let dir = std::env::temp_dir();
        let path = dir.join(format!("mpc_cfg_test_{}.json", std::process::id()));
        c.save_json(&path).unwrap();
        let back = MultiProbCutConfig::load_json(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(back.enabled, "load_json forces enabled = true");
        let mut want = c.clone();
        want.enabled = true;
        assert_eq!(back, want);
    }

    #[test]
    fn stage_params_usability() {
        assert!(!MpcStageParams::default().is_usable());
        assert!(!MpcStageParams::new(0.0, 1.0, 1.0).is_usable());
        assert!(!MpcStageParams::new(f64::NAN, 0.0, 1.0).is_usable());
        assert!(MpcStageParams::new(1.0, 0.0, 2.0).is_usable());
    }
}
