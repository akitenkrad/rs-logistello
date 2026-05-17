//! # logistello-cli (library surface)
//!
//! The CLI binary (`src/main.rs`) is the user entry point; this library
//! target exists so the workspace integration tests can exercise the
//! Phase 4b position-extraction pipeline (`extract`) directly without
//! shelling out to the binary.
//!
//! - [`extract`] — training-position extraction for evaluation-weight
//!   learning (design doc §4.4 B3, §4.5 B5; Phase 4b). Owns the B4
//!   canonicalisation (reuses the exact `logistello-eval` pack tables) and
//!   emits the documented `PEX1` columnar binary (see `EXTRACT_FORMAT.md`).
//! - [`probcut_fit`] — single-ProbCut `(a, b, σ)` coefficient estimation
//!   (design doc §4.3.4 / §4.5 B7; Phase 5). Generates a sample corpus,
//!   measures `v_d`/`v_h`, fits OLS per disc-count phase, and writes a
//!   [`logistello_search::ProbCutConfig`]-compatible JSON file.
//! - [`glem_extract`] — GLEM base-literal extraction (design doc §4.3.6;
//!   Phase 7). Owns the base-literal extraction (mirrors `extract`'s "Rust
//!   owns canonicalisation") and emits the documented `GLX1` columnar
//!   binary the Python `train-glem` tool consumes.
//! - [`edax`] — shared Edax v4.6 external-engine configuration (design doc
//!   §4.5 B5; Phase 9a). The single place that knows how to drive the
//!   locally built, gitignored `.edax/edax` binary; consumed by Phase 9b's
//!   `elo-vs-edax`.

pub mod edax;
pub mod extract;
pub mod glem_extract;
pub mod probcut_fit;
