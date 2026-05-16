//! Zobrist-keyed transposition table.
//!
//! TODO: Phase 3 — fixed-size, replace-by-depth table keyed by the Zobrist
//! hash from `logistello_core::zobrist` (design doc §4.3.2).

/// Transposition table (placeholder).
#[derive(Debug, Default)]
pub struct TranspositionTable;

impl TranspositionTable {
    /// Creates an empty transposition table.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}
