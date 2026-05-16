//! 13-stage (disc-count) evaluation dispatch — design doc §4.4 B2.
//!
//! ```text
//! stage(p) = clamp( floor( (discs(p) - 13) / 4 ), 0, 12 ),
//! discs(p) = 64 - empties
//! ```
//!
//! Stage boundaries are disc counts `{13–16, 17–20, …, 57–60, 61–64}`; any
//! disc count `< 13` clamps to stage 0 (design doc B2, Buro 1997b §5 /
//! Buro 2002). This is the canonical replication choice (Buro's 13 stages);
//! Edax 4.6 itself is stage-less (ply-indexed, 61 tables) but the design doc
//! mandates Buro's staging here.

use othello_core::GameState;

/// Number of evaluation stages (design doc B2). Mirrors
/// [`logistello_core::consts::NUM_STAGES`].
pub const N_STAGES: usize = 13;

/// Evaluation stage `0..N_STAGES` for a position with `discs` stones on the
/// board (`discs = 64 - empties`), per design doc §4.4 B2.
///
/// `stage = clamp(floor((discs - 13) / 4), 0, 12)`. Floor division is over
/// the signed value, so `discs < 13` yields a negative quotient that clamps
/// to `0` (e.g. `discs = 12 -> floor(-1/4) = -1 -> 0`).
#[must_use]
pub fn stage(discs: u32) -> usize {
    let q = (discs as i32 - 13).div_euclid(4);
    q.clamp(0, (N_STAGES - 1) as i32) as usize
}

/// Evaluation stage for a [`GameState`] (`discs = 64 - empties`).
#[must_use]
pub fn stage_for_state(state: &GameState) -> usize {
    let discs = 64 - state.board.empty_count();
    stage(discs)
}

/// Back-compat shim for the Phase-1 placeholder name. Prefer [`stage`].
#[must_use]
pub fn stage_for_disc_count(disc_count: u32) -> usize {
    stage(disc_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_stages_is_13() {
        assert_eq!(N_STAGES, 13);
        assert_eq!(N_STAGES, logistello_core::consts::NUM_STAGES);
    }

    /// Exact B2 boundary table: every quad boundary plus the clamp ends.
    #[test]
    fn b2_exact_boundaries() {
        // (discs, expected stage)
        let cases: &[(u32, usize)] = &[
            (0, 0),
            (4, 0),  // start position
            (12, 0), // < 13 -> clamp to 0
            (13, 0),
            (16, 0),
            (17, 1),
            (20, 1),
            (21, 2),
            (24, 2),
            (25, 3),
            (28, 3),
            (29, 4),
            (32, 4),
            (33, 5),
            (36, 5),
            (37, 6),
            (40, 6),
            (41, 7),
            (44, 7),
            (45, 8),
            (48, 8),
            (49, 9),
            (52, 9),
            (53, 10),
            (56, 10),
            (57, 11),
            (60, 11),
            (61, 12),
            (64, 12),
        ];
        for &(d, s) in cases {
            assert_eq!(stage(d), s, "discs={d} expected stage {s}");
        }
    }

    #[test]
    fn monotonic_nondecreasing() {
        let mut prev = 0;
        for d in 0..=64 {
            let s = stage(d);
            assert!(s >= prev, "stage decreased at discs={d}");
            assert!(s < N_STAGES);
            prev = s;
        }
        assert_eq!(stage(64), 12);
    }

    #[test]
    fn start_position_is_stage_0() {
        let s = GameState::standard_8x8();
        assert_eq!(stage_for_state(&s), 0);
    }
}
