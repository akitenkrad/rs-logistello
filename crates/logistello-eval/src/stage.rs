//! 13-stage (disc-count) evaluation dispatch.
//!
//! TODO: Phase 4 — map a position's disc count to one of the 13 evaluation
//! stages and select the corresponding weight set (design doc §4.4 B2).

use logistello_core::consts::NUM_STAGES;

/// Returns the evaluation stage index `0..NUM_STAGES` for a given disc count.
///
/// TODO: Phase 4 — replace with the real Edax-style stage boundaries.
#[must_use]
pub fn stage_for_disc_count(_disc_count: u32) -> usize {
    let _ = NUM_STAGES;
    0
}
