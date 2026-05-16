//! GLEM — Generalized Linear Evaluation Model.
//!
//! TODO: Phase 7 — basic-feature → conjunction expansion (Buro 1998 CG'98).
//! Feature enumeration is performed in Python; this module consumes the
//! resulting conjunction table at inference time.

/// Placeholder GLEM feature table.
#[derive(Debug, Default, Clone)]
pub struct GlemFeatures;

impl GlemFeatures {
    /// Creates an empty GLEM feature table.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}
