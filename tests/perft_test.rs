//! Workspace integration test: known Othello perft values.
//!
//! These are the standard published perft numbers for Othello from the
//! initial position (one placement = one ply; forced passes are followed
//! transparently). They double as a correctness check on the reused
//! `othello-core` move generator via `logistello_core::perft`.

use logistello_core::perft::perft_standard;

#[test]
fn known_othello_perft_values() {
    assert_eq!(perft_standard(1), 4, "perft(1)");
    assert_eq!(perft_standard(2), 12, "perft(2)");
    assert_eq!(perft_standard(3), 56, "perft(3)");
    assert_eq!(perft_standard(4), 244, "perft(4)");
    assert_eq!(perft_standard(5), 1396, "perft(5)");
    assert_eq!(perft_standard(6), 8200, "perft(6)");
    assert_eq!(perft_standard(7), 55092, "perft(7)");
    assert_eq!(perft_standard(8), 390216, "perft(8)");
}
