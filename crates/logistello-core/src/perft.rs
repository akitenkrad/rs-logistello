//! `perft` — search-tree size verification.
//!
//! `perft(state, depth)` counts the number of distinct legal move sequences
//! (leaf nodes) of exactly `depth` plies reachable from `state`, using the
//! reused `othello-core` move generator and rules.
//!
//! Othello perft convention used here (matches the standard published
//! sequence 4, 12, 56, 244, 1396, 8200, 55092, 390216 from the initial
//! position): a *placement* consumes one ply; a *forced pass* does not
//! consume a ply and is not counted as a branch — search transparently
//! continues for the opponent. A terminal position before `depth` reaches 0
//! contributes a single leaf (the line simply ends early).

use othello_core::{GameState, Move};

/// Counts leaf nodes at `depth` plies from `state`.
///
/// - `depth == 0` counts the current node as one leaf.
/// - A forced pass is followed without decrementing `depth`.
/// - A terminal position counts as one leaf regardless of remaining depth.
#[must_use]
pub fn perft(state: &GameState, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    if state.is_terminal() {
        return 1;
    }

    let moves = state.legal_moves();

    // Forced pass: no placement available but the game is not terminal.
    // Follow the pass transparently without consuming a ply.
    if moves.is_empty() || (moves.len() == 1 && moves[0] == Move::Pass) {
        let mut next = state.clone();
        // `apply_move(Move::Pass)` is legal exactly when there is no placement.
        if next.apply_move(Move::Pass).is_ok() {
            return perft(&next, depth);
        }
        return 1;
    }

    let mut nodes = 0u64;
    for mv in moves {
        if mv == Move::Pass {
            continue;
        }
        let mut next = state.clone();
        if next.apply_move(mv).is_ok() {
            nodes += perft(&next, depth - 1);
        }
    }
    nodes
}

/// Convenience wrapper: perft from the standard 8x8 starting position.
#[must_use]
pub fn perft_standard(depth: u32) -> u64 {
    perft(&GameState::standard_8x8(), depth)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perft_known_values() {
        assert_eq!(perft_standard(1), 4);
        assert_eq!(perft_standard(2), 12);
        assert_eq!(perft_standard(3), 56);
        assert_eq!(perft_standard(4), 244);
        assert_eq!(perft_standard(5), 1396);
        assert_eq!(perft_standard(6), 8200);
    }
}
