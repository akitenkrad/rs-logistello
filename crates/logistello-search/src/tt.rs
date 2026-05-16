//! Zobrist-keyed transposition table (design doc §4.3.2).
//!
//! Fixed-size, power-of-two number of slots, direct-mapped by
//! `index = key & mask`, with **depth-preferred** replacement: a probe that
//! collides with a deeper existing entry leaves it in place.
//!
//! The stored value is bound-qualified by [`Bound`]:
//! - `Exact` — `value` is the exact minimax value of the subtree.
//! - `Lower` — search failed high; `value` is a lower bound (`>= value`).
//! - `Upper` — search failed low; `value` is an upper bound (`<= value`).
//!
//! These map to the standard alpha-beta window narrowing in
//! [`crate::alphabeta`].

use othello_core::Move;

/// Bound type qualifying a stored value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bound {
    /// Exact minimax value of the subtree.
    Exact,
    /// Lower bound: the true value is `>= value` (fail-high / beta cutoff).
    Lower,
    /// Upper bound: the true value is `<= value` (fail-low / no move beat alpha).
    Upper,
}

/// One transposition-table record.
#[derive(Debug, Clone, Copy)]
pub struct Entry {
    /// Full Zobrist key (used to detect index collisions).
    pub key: u64,
    /// Search depth (remaining plies) this entry was produced at.
    pub depth: u32,
    /// Stored value, qualified by `flag`.
    pub value: i32,
    /// Bound type of `value`.
    pub flag: Bound,
    /// Best move found at this node (used first by the move orderer).
    pub best_move: Option<Move>,
}

/// Direct-mapped, fixed-size transposition table.
#[derive(Debug)]
pub struct TranspositionTable {
    slots: Vec<Option<Entry>>,
    /// `slots.len() - 1`; `slots.len()` is a power of two.
    mask: u64,
}

/// Default table size: `2^20` slots (design doc §4.3.2 / §6 sweep default).
pub const DEFAULT_CAPACITY_POW2: u32 = 20;

impl TranspositionTable {
    /// Creates a table with `2^capacity_pow2` slots.
    ///
    /// `capacity_pow2` is clamped to `1..=32` so the table is always a
    /// non-trivial power of two.
    #[must_use]
    pub fn with_capacity_pow2(capacity_pow2: u32) -> Self {
        let bits = capacity_pow2.clamp(1, 32);
        let len = 1usize << bits;
        Self {
            slots: vec![None; len],
            mask: (len as u64) - 1,
        }
    }

    /// Creates a table with the default capacity ([`DEFAULT_CAPACITY_POW2`]).
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity_pow2(DEFAULT_CAPACITY_POW2)
    }

    /// Number of slots in the table.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Clears all entries (used between independent searches in tests).
    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            *slot = None;
        }
    }

    #[inline]
    fn index(&self, key: u64) -> usize {
        (key & self.mask) as usize
    }

    /// Probes for an entry whose full key matches `key`.
    ///
    /// Returns `None` on an empty slot or an index collision (different key
    /// hashing to the same slot).
    #[must_use]
    pub fn probe(&self, key: u64) -> Option<&Entry> {
        let idx = self.index(key);
        match &self.slots[idx] {
            Some(e) if e.key == key => Some(e),
            _ => None,
        }
    }

    /// Stores `entry`, keeping the deeper of any colliding records.
    ///
    /// Depth-preferred replacement: if the slot already holds an entry for a
    /// **different** position that was searched at least as deep, the new
    /// (shallower) entry is dropped. An entry for the **same** key is always
    /// replaced (it is at least as informative — same or greater depth from
    /// iterative deepening).
    pub fn store(&mut self, entry: Entry) {
        let idx = self.index(entry.key);
        let replace = match &self.slots[idx] {
            None => true,
            Some(existing) => existing.key == entry.key || entry.depth >= existing.depth,
        };
        if replace {
            self.slots[idx] = Some(entry);
        }
    }
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use othello_core::Coord;

    fn entry(key: u64, depth: u32, value: i32) -> Entry {
        Entry {
            key,
            depth,
            value,
            flag: Bound::Exact,
            best_move: Some(Move::Place(Coord::new(2, 3))),
        }
    }

    #[test]
    fn capacity_is_power_of_two() {
        let tt = TranspositionTable::with_capacity_pow2(10);
        assert_eq!(tt.capacity(), 1024);
        assert!(tt.capacity().is_power_of_two());
    }

    #[test]
    fn store_then_probe_roundtrip() {
        let mut tt = TranspositionTable::with_capacity_pow2(8);
        tt.store(entry(0xDEAD_BEEF, 5, 17));
        let got = tt.probe(0xDEAD_BEEF).expect("entry present");
        assert_eq!(got.value, 17);
        assert_eq!(got.depth, 5);
    }

    #[test]
    fn probe_miss_on_absent_key() {
        let tt = TranspositionTable::with_capacity_pow2(8);
        assert!(tt.probe(123).is_none());
    }

    #[test]
    fn index_collision_does_not_alias() {
        let mut tt = TranspositionTable::with_capacity_pow2(4); // 16 slots, mask = 0xF
        // Two keys with the same low nibble collide on the same slot.
        tt.store(entry(0x10, 3, 1));
        tt.store(entry(0x20, 3, 2)); // same slot (low nibble 0), equal depth -> replaces
        // 0x10 was evicted; 0x20 present; the other key must not be aliased.
        assert!(tt.probe(0x10).is_none());
        assert_eq!(tt.probe(0x20).unwrap().value, 2);
    }

    #[test]
    fn depth_preferred_replacement() {
        let mut tt = TranspositionTable::with_capacity_pow2(4);
        tt.store(entry(0x100, 8, 1)); // deep entry
        tt.store(entry(0x200, 3, 2)); // collides (low nibble 0), shallower -> rejected
        assert_eq!(tt.probe(0x100).unwrap().value, 1, "deep entry kept");
        assert!(tt.probe(0x200).is_none(), "shallow colliding entry dropped");
    }

    #[test]
    fn same_key_always_replaced() {
        let mut tt = TranspositionTable::with_capacity_pow2(4);
        tt.store(entry(0xABC, 8, 1));
        tt.store(entry(0xABC, 2, 99)); // same key, shallower depth, still replaces
        assert_eq!(tt.probe(0xABC).unwrap().value, 99);
    }
}
