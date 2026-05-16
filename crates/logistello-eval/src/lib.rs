//! # logistello-eval
//!
//! Pattern-based evaluation function for the Logistello reproduction
//! (inference side). Training of the regression weights is done in Python
//! (`tools/logistello_tools`); this crate only consumes learned weights and
//! evaluates positions during search.
//!
//! - [`pattern`] — Edax-compatible pattern encoding.
//! - [`stage`]   — 13-stage (disc-count) evaluation dispatch.
//! - [`glem`]    — GLEM basic-feature → conjunction expansion.
//! - [`weights`] — load weights produced by the Python trainer.

pub mod glem;
pub mod pattern;
pub mod stage;
pub mod weights;
