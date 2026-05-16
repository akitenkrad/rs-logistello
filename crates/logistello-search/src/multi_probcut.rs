//! Multi-ProbCut cascade (Buro 1997).
//!
//! TODO: Phase 6 — cascade of ProbCut cuts for heights h = 3..13
//! (design doc §4.5 B7).

use crate::probcut::ProbCutParams;

/// Multi-ProbCut cut table indexed by remaining height (placeholder).
#[derive(Debug, Default)]
pub struct MultiProbCut;

impl MultiProbCut {
    /// Creates an empty Multi-ProbCut table.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Returns the ProbCut parameters for a given height (placeholder).
    #[must_use]
    pub fn params_for_height(&self, _height: u32) -> Option<ProbCutParams> {
        None
    }
}
