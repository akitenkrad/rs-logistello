//! Killer-move heuristic for move ordering (design doc §4.3.2).
//!
//! Two killer slots per ply. A *killer* is a quiet move that produced a
//! beta cutoff at the same ply in a sibling subtree; trying it early in
//! later siblings tends to cause earlier cutoffs. Killers only **reorder**
//! moves — they never prune — so they cannot change the minimax value
//! (verified by `pvs_killer_invariance` in the search tests).

use othello_core::Move;

/// Number of killer slots kept per ply.
const KILLERS_PER_PLY: usize = 2;

/// Maximum search ply the table tracks; deeper plies simply skip the
/// heuristic (correctness is unaffected).
const MAX_PLY: usize = 128;

/// Per-ply killer-move table (2 slots per ply).
#[derive(Debug, Clone)]
pub struct KillerTable {
    slots: Vec<[Option<Move>; KILLERS_PER_PLY]>,
}

impl KillerTable {
    /// Creates an empty killer table.
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: vec![[None; KILLERS_PER_PLY]; MAX_PLY],
        }
    }

    /// Clears every killer slot.
    pub fn clear(&mut self) {
        for s in &mut self.slots {
            *s = [None; KILLERS_PER_PLY];
        }
    }

    /// Records `mv` as a killer at `ply`.
    ///
    /// Slot 0 holds the most recent killer; on a new distinct killer the old
    /// slot-0 move shifts down to slot 1 (a tiny 2-entry LRU). Recording the
    /// move already in slot 0 is a no-op.
    pub fn record(&mut self, ply: usize, mv: Move) {
        if ply >= self.slots.len() {
            return;
        }
        let s = &mut self.slots[ply];
        if s[0] == Some(mv) {
            return;
        }
        s[1] = s[0];
        s[0] = Some(mv);
    }

    /// Returns the killer moves recorded at `ply` (most recent first).
    /// Out-of-range plies yield no killers.
    #[must_use]
    pub fn killers(&self, ply: usize) -> [Option<Move>; KILLERS_PER_PLY] {
        if ply >= self.slots.len() {
            return [None; KILLERS_PER_PLY];
        }
        self.slots[ply]
    }

    /// Whether `mv` is a killer at `ply`.
    #[must_use]
    pub fn is_killer(&self, ply: usize, mv: Move) -> bool {
        if ply >= self.slots.len() {
            return false;
        }
        let s = &self.slots[ply];
        s[0] == Some(mv) || s[1] == Some(mv)
    }
}

impl Default for KillerTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use othello_core::Coord;

    fn mv(r: u8, c: u8) -> Move {
        Move::Place(Coord::new(r, c))
    }

    #[test]
    fn records_and_reports_killer() {
        let mut k = KillerTable::new();
        k.record(3, mv(2, 3));
        assert!(k.is_killer(3, mv(2, 3)));
        assert!(!k.is_killer(3, mv(4, 5)));
        assert!(!k.is_killer(2, mv(2, 3)));
    }

    #[test]
    fn two_slots_lru_shift() {
        let mut k = KillerTable::new();
        k.record(0, mv(0, 0));
        k.record(0, mv(1, 1));
        assert_eq!(k.killers(0), [Some(mv(1, 1)), Some(mv(0, 0))]);
        // A third distinct killer pushes (0,0) out.
        k.record(0, mv(2, 2));
        assert_eq!(k.killers(0), [Some(mv(2, 2)), Some(mv(1, 1))]);
        assert!(!k.is_killer(0, mv(0, 0)));
    }

    #[test]
    fn duplicate_record_is_noop() {
        let mut k = KillerTable::new();
        k.record(1, mv(3, 3));
        k.record(1, mv(3, 3));
        assert_eq!(k.killers(1), [Some(mv(3, 3)), None]);
    }

    #[test]
    fn out_of_range_ply_is_safe() {
        let mut k = KillerTable::new();
        k.record(usize::MAX, mv(0, 0)); // must not panic
        assert!(!k.is_killer(usize::MAX, mv(0, 0)));
        assert_eq!(k.killers(usize::MAX), [None, None]);
    }
}
