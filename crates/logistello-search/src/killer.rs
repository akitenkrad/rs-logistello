//! Killer-move heuristic for move ordering.
//!
//! TODO: Phase 2 — per-ply killer slots feeding the move orderer.

/// Killer-move table (placeholder).
#[derive(Debug, Default)]
pub struct KillerTable;

impl KillerTable {
    /// Creates an empty killer table.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}
