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

pub mod extract;
