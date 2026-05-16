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
//! - [`pattern`] — Edax-compatible pattern encoding.
//! - [`stage`]   — 13-stage (disc-count) evaluation dispatch.
//! - [`glem`]    — GLEM basic-feature → conjunction expansion.
//! - [`weights`] — load weights produced by the Python trainer.

pub mod eval_trait;
pub mod glem;
pub mod pattern;
pub mod stage;
pub mod weights;

pub use eval_trait::{BasicEval, DiscDiffEval, LeafEvaluator};
