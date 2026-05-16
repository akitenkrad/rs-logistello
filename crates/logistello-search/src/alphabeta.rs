//! NegaScout / PVS alpha-beta search core.
//!
//! TODO: Phase 2 — implement NegaScout (PVS) over the reused `othello-core`
//! `GameState`, using `logistello-eval` at the leaves and the transposition
//! table for move ordering and cutoffs (design doc §4.3.2).

use othello_core::GameState;

/// Searches `state` to `depth` plies and returns a centi-disc score
/// (placeholder).
///
/// TODO: Phase 2 — implement the real NegaScout recursion.
#[must_use]
pub fn search(_state: &GameState, _depth: u32) -> i32 {
    0
}
