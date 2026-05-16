//! Opening-book learner: self-play + Negamax back-propagation.
//!
//! TODO: Phase 8 — generate self-play games via `othello-engine`, then
//! back-propagate leaf evaluations through the book tree (design doc §4.3.7).

/// Opening-book learner state (placeholder).
#[derive(Debug, Default)]
pub struct BookLearner;

impl BookLearner {
    /// Creates a new, empty book learner.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}
