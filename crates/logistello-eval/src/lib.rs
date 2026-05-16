//! # logistello-eval
//!
//! Pattern-based evaluation function for the Logistello reproduction
//! (inference side). Training of the regression weights is done in Python
//! (`tools/logistello_tools`); this crate only consumes learned weights and
//! evaluates positions during search.
//!
//! - [`eval_trait`] — minimal [`LeafEvaluator`] abstraction, the trivial
//!   [`DiscDiffEval`] used by Phase 2 search tests, and the Phase 3
//!   [`BasicEval`] (disc count + mobility, IAGO / Buro 1994 lineage).
//! - [`pattern`] — Edax-準拠 (v4.6) pattern definitions: 47 features
//!   (46 instances + 1 constant), raw key extraction (design doc §4.4 B1).
//! - [`stage`]   — 13-stage (disc-count) evaluation dispatch (B2).
//! - [`weights`] — Edax pack/unpack symmetry normalisation + the
//!   [`EvalWeights`] container / serialization contract (B4).
//! - [`pattern_eval`] — [`PatternEval`], the pattern/regression
//!   [`LeafEvaluator`] (design doc §4.3.3).
//! - [`glem`]    — GLEM basic-feature → conjunction expansion (Phase 7).

pub mod eval_trait;
pub mod glem;
pub mod pattern;
pub mod pattern_eval;
pub mod stage;
pub mod weights;

pub use eval_trait::{BasicEval, DiscDiffEval, LeafEvaluator};
pub use pattern::{
    CONST_FEATURE, FEATURES, FeatureDef, N_FEATURES, PatternType, feature_key, feature_keys,
    feature_keys_state,
};
pub use pattern_eval::PatternEval;
pub use stage::{N_STAGES, stage, stage_for_state};
pub use weights::{EVAL_PACKED_SIZE, EvalWeights, PackTable, PackTables, Weights, WeightsError};
