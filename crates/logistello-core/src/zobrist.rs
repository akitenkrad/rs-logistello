//! Zobrist hashing over the reused `othello-core` board types.
//!
//! The transposition table (`logistello-search::tt`) keys positions by a
//! 64-bit Zobrist hash (design doc §4.3.2). The key tables are filled from a
//! **fixed-seed** ChaCha20 RNG so hashes are reproducible across runs and
//! processes.
//!
//! TODO(perf, Phase 10): incremental update. The hash is currently recomputed
//! from scratch over the occupied squares on every probe/store. This is
//! correct but O(occupied squares); an incremental XOR update threaded
//! through `apply_move` is a pure performance optimisation and does not
//! change any search value.

use othello_core::{Color, Coord, GameState};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// Number of board squares on a standard 8x8 board.
const SQUARES: usize = 64;

/// Number of stone colors (Black, White).
const COLORS: usize = 2;

/// Fixed seed for the ChaCha20 RNG that fills the key tables.
///
/// Chosen arbitrarily but **frozen**: changing it changes every Zobrist key
/// and therefore invalidates any persisted transposition data. Reproducible
/// hashing across runs is a hard requirement (see module docs).
const ZOBRIST_SEED: u64 = 0x6C_6F_67_69_73_74_65_6C; // "logistel" in ASCII

/// Maps a [`Color`] to its index into the per-square key table.
#[inline]
const fn color_index(color: Color) -> usize {
    match color {
        Color::Black => 0,
        Color::White => 1,
    }
}

/// Holds the random keys used for Zobrist hashing.
///
/// - `square_color[sq][c]` — key XOR-ed in when square `sq` (bit index
///   `row*8+col`, A1 = 0) holds a stone of color index `c`
///   (`0 = Black`, `1 = White`).
/// - `side_to_move_key` — key XOR-ed in when it is White's turn to move
///   (Black-to-move contributes nothing, by convention).
#[derive(Debug, Clone)]
pub struct Zobrist {
    square_color: [[u64; COLORS]; SQUARES],
    side_to_move_key: u64,
}

impl Zobrist {
    /// Creates a key table seeded deterministically from [`ZOBRIST_SEED`].
    ///
    /// The same keys are produced on every call, in every process.
    #[must_use]
    pub fn new() -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(ZOBRIST_SEED);
        let mut square_color = [[0u64; COLORS]; SQUARES];
        for square in square_color.iter_mut() {
            for slot in square.iter_mut() {
                *slot = rng.r#gen();
            }
        }
        let side_to_move_key = rng.r#gen();
        Self {
            square_color,
            side_to_move_key,
        }
    }

    /// Computes the full Zobrist hash of `state` from scratch.
    ///
    /// Folds in one key per occupied square (by its color) plus the
    /// side-to-move key when White is to move. Two positions with the same
    /// stone configuration **and** the same side to move hash identically;
    /// the side-to-move term keeps otherwise-identical boards with different
    /// turns distinct.
    #[must_use]
    pub fn hash(&self, state: &GameState) -> u64 {
        let mut key = 0u64;
        for row in 0..8u8 {
            for col in 0..8u8 {
                let coord = Coord::new(row, col);
                if let Some(color) = state.board.cell(coord) {
                    let sq = coord.to_bit_index_8x8() as usize;
                    key ^= self.square_color[sq][color_index(color)];
                }
            }
        }
        if state.side_to_move == Color::White {
            key ^= self.side_to_move_key;
        }
        key
    }
}

impl Default for Zobrist {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience free function: hash `state` with a default key table.
///
/// Note: this builds a fresh key table on every call (cheap but not free).
/// Hot paths should hold a single [`Zobrist`] and call [`Zobrist::hash`].
#[must_use]
pub fn zobrist_key(state: &GameState) -> u64 {
    Zobrist::new().hash(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use othello_core::Move;

    #[test]
    fn deterministic_across_instances() {
        let a = Zobrist::new();
        let b = Zobrist::new();
        let s = GameState::standard_8x8();
        assert_eq!(a.hash(&s), b.hash(&s));
        assert_eq!(zobrist_key(&s), a.hash(&s));
    }

    #[test]
    fn distinct_for_different_positions() {
        let z = Zobrist::new();
        let start = GameState::standard_8x8();
        let mut moved = start.clone();
        moved.apply_move(start.legal_moves()[0]).unwrap();
        assert_ne!(z.hash(&start), z.hash(&moved));
    }

    #[test]
    fn side_to_move_changes_hash() {
        let z = Zobrist::new();
        let mut a = GameState::standard_8x8();
        let mut b = GameState::standard_8x8();
        // Same board configuration, different side to move.
        a.side_to_move = Color::Black;
        b.side_to_move = Color::White;
        assert_ne!(z.hash(&a), z.hash(&b));
    }

    #[test]
    fn empty_board_black_to_move_hashes_zero() {
        // No occupied squares, Black to move => no keys folded in.
        let z = Zobrist::new();
        let mut s = GameState::standard_8x8();
        for row in 0..8u8 {
            for col in 0..8u8 {
                s.board.set(Coord::new(row, col), None);
            }
        }
        s.side_to_move = Color::Black;
        assert_eq!(z.hash(&s), 0);
    }

    #[test]
    fn unaffected_by_irrelevant_state_fields() {
        // Hash depends only on board + side to move, not move_number etc.
        let z = Zobrist::new();
        let mut a = GameState::standard_8x8();
        let mut b = GameState::standard_8x8();
        b.move_number = 999;
        b.last_move = Some(Move::Pass);
        b.consecutive_passes = 1;
        assert_eq!(z.hash(&a), z.hash(&b));
        let _ = &mut a;
    }
}
