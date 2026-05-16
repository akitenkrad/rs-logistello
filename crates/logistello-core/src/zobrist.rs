//! Incremental Zobrist hashing over the reused `othello-core` board types.
//!
//! TODO: Phase 3 — wire this into the transposition table
//! (`logistello-search::tt`). For now this is a placeholder type with a
//! deterministic table so the API shape is fixed but no hashing logic is
//! committed to yet.

use othello_core::GameState;

/// Number of (square, piece-color) slots in the Zobrist key table:
/// 64 squares * 2 colors.
const TABLE_LEN: usize = 64 * 2;

/// Holds the random keys used for incremental Zobrist hashing.
///
/// The hash is computed/updated externally; this struct owns the key table
/// and the side-to-move key.
#[derive(Debug, Clone)]
pub struct Zobrist {
    /// Per-(square, color) random keys.
    table: [u64; TABLE_LEN],
    /// Key XOR-ed in when it is White's turn to move.
    side_to_move_key: u64,
}

impl Zobrist {
    /// Creates a new key table.
    ///
    /// TODO: Phase 3 — seed from a deterministic RNG (`rand_chacha`) so hashes
    /// are reproducible across runs. Currently zero-initialised.
    #[must_use]
    pub fn new() -> Self {
        Self {
            table: [0u64; TABLE_LEN],
            side_to_move_key: 0,
        }
    }

    /// Computes the full Zobrist hash of a position from scratch.
    ///
    /// TODO: Phase 3 — replace with the real folding logic over the
    /// `othello-core` bitboard population. Currently returns 0.
    #[must_use]
    pub fn hash(&self, _state: &GameState) -> u64 {
        let _ = (&self.table, self.side_to_move_key);
        0
    }
}

impl Default for Zobrist {
    fn default() -> Self {
        Self::new()
    }
}
