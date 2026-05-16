//! Board constants shared across the Logistello workspace.
//!
//! TODO: Phase 1 — expand with stage boundaries, pattern offsets, and
//! Edax-compatible square indices as evaluation work lands.

/// Edge length of a standard Othello board.
pub const BOARD_SIZE: usize = 8;

/// Number of cells on a standard Othello board (`BOARD_SIZE * BOARD_SIZE`).
pub const BOARD_CELLS: usize = BOARD_SIZE * BOARD_SIZE;

/// Number of evaluation stages (disc-count buckets) used by the Logistello
/// pattern evaluation function. See design doc §4.4 B2 (13 stages).
pub const NUM_STAGES: usize = 13;
