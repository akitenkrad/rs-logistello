//! Iterative deepening driver.
//!
//! TODO: Phase 2 — drive [`crate::alphabeta::search`] over increasing depths
//! with aspiration windows and time control.

use othello_core::{GameState, Move};

/// Returns the best move found within `max_depth` (placeholder).
///
/// TODO: Phase 2 — implement the iterative-deepening loop.
#[must_use]
pub fn best_move(_state: &GameState, _max_depth: u32) -> Option<Move> {
    None
}
