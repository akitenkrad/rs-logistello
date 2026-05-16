//! Load learned evaluation weights produced by the Python trainer.
//!
//! TODO: Phase 4 — deserialize the per-stage weight vectors (bincode) emitted
//! by `logistello_tools train-eval` and expose a fast lookup for search.

/// Per-stage evaluation weights (placeholder).
#[derive(Debug, Default, Clone)]
pub struct Weights;

impl Weights {
    /// Creates an empty weight set.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Loads weights from a bincode file.
    ///
    /// TODO: Phase 4 — implement actual deserialization.
    #[must_use]
    pub fn load(_path: &str) -> Option<Self> {
        None
    }
}
