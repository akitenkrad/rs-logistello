//! # logistello-core
//!
//! Core utilities for the Logistello (Buro 1994-1999) reproduction.
//!
//! This crate intentionally **reuses** the bitboard, move generation, and
//! game-state machinery from the external `rs-othello-sim` library
//! (`othello-core`). It only adds the pieces that Logistello-style search and
//! evaluation need on top of that foundation:
//!
//! - [`zobrist`] — incremental Zobrist hashing for the transposition table.
//! - [`perft`] — search-tree size verification against known Othello values.
//! - [`consts`] — board constants shared across the workspace.

pub mod consts;
pub mod perft;
pub mod zobrist;

pub use consts::{BOARD_CELLS, BOARD_SIZE};
pub use perft::perft;
pub use zobrist::Zobrist;
