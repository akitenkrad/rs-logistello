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
//! - [`edax`] — shared Edax v4.6 external-engine configuration + the
//!   Phase-9b [`edax::EdaxGtpSession`] direct-GTP full-game driver (design
//!   doc §4.5 B5). The single place that knows how to drive the locally
//!   built, gitignored `.edax/edax` binary.
//! - [`wthor_murakami`] — Murakami-1997 gold-set extraction from the raw
//!   WThor `.wtb`+`.jou` DB (design doc §4.5 B5; Phase 9b `murakami-extract`).
//! - [`match_replay`] — recorded-match replay + move-match-rate scoring
//!   (design doc §4.3.8 `move_match_rate_murakami`; Phase 9b `match-replay`).
//! - [`elo`] — Elo-delta / win-rate aggregation vs Edax (design doc §4.3.8
//!   `elo_vs_edax_level_N`; Phase 9b `elo-vs-edax`).
//! - [`eval_corr`] — Pearson correlation of our eval vs Edax's (design doc
//!   §4.3.8 `eval_correlation_edax`; Phase 9b `eval-correlation-edax`).
//! - [`record`] — how a run is recorded into runvault: the paper metadata,
//!   the datasets whose *contents* decide a result (learned weights, the
//!   opening book, the Edax install), and the shape of every event.
//! - [`sweep`] — Phase 10 sensitivity analysis: §6 parameter-grid
//!   expansion, deterministic per-`(condition, seed)` measurement, and the
//!   sweep parent / per-condition child runs it records them into.

pub mod edax;
pub mod elo;
pub mod eval_corr;
pub mod extract;
pub mod glem_extract;
pub mod match_replay;
pub mod probcut_fit;
pub mod record;
pub mod sweep;
pub mod wthor_murakami;
