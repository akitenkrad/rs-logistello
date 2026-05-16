//! Single ProbCut selective forward pruning (Buro 1995 ICCA).
//!
//! TODO: Phase 5 — ProbCut(d, D) with (a, b, sigma) fitted in Python; canonical
//! parameters are in design doc §4.5 B7.

/// ProbCut parameters for one (shallow d, deep D) pair (placeholder).
#[derive(Debug, Default, Clone, Copy)]
pub struct ProbCutParams {
    /// Linear slope.
    pub a: f64,
    /// Linear intercept.
    pub b: f64,
    /// Residual standard deviation.
    pub sigma: f64,
}
