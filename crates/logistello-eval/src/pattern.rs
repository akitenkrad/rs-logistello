//! Pattern encoding (Edax-compatible).
//!
//! TODO: Phase 4 — implement the Edax-準拠 pattern set (corner 3x3, edges,
//! diagonals, etc.) and the pack/unpack symmetry normalisation described in
//! design doc §4.4 B1 / B4.

use othello_core::GameState;

/// Identifier for a single pattern family (placeholder).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PatternId(pub u16);

/// Extracts the pattern feature indices for `state`.
///
/// TODO: Phase 4 — return the per-pattern symmetry-normalised indices.
#[must_use]
pub fn encode(_state: &GameState) -> Vec<u32> {
    Vec::new()
}
