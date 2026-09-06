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

/// The same count as [`perft`], calling `on_subtree` once for every node of
/// the depth-`split` frontier before that node's subtree is counted.
///
/// A `perft` is one recursive call that says nothing until it returns, and at
/// depth 12 that is over a minute. Splitting the *top* `split` plies gives a
/// countable unit — one frontier subtree — without touching the count: the
/// enumeration above the frontier is `perft`'s own, move for move, and below
/// it the work is [`perft`] itself. The number of frontier nodes is
/// `perft(state, split)`, which is why a caller can open a **bounded** stage
/// and know it closes at exactly 100%.
///
/// `split` is expected to be `<= depth` (a caller clamps it); a larger one
/// simply makes every real leaf a frontier node.
pub fn perft_observed(
    state: &GameState,
    depth: u32,
    split: u32,
    on_subtree: &mut impl FnMut(),
) -> u64 {
    // The frontier: this node is one unit of work, and the rest of it is a
    // plain `perft`.
    if split == 0 {
        on_subtree();
        return perft(state, depth);
    }
    // A line that ends above the frontier is still one frontier leaf, exactly
    // as `perft(state, split)` counts it.
    if depth == 0 || state.is_terminal() {
        on_subtree();
        return 1;
    }

    let moves = state.legal_moves();

    // Forced pass: followed transparently, consuming neither a ply nor a
    // level of the split — the same rule the count itself obeys.
    if moves.is_empty() || (moves.len() == 1 && moves[0] == Move::Pass) {
        let mut next = state.clone();
        if next.apply_move(Move::Pass).is_ok() {
            return perft_observed(&next, depth, split, on_subtree);
        }
        on_subtree();
        return 1;
    }

    let mut nodes = 0u64;
    for mv in moves {
        if mv == Move::Pass {
            continue;
        }
        let mut next = state.clone();
        if next.apply_move(mv).is_ok() {
            nodes += perft_observed(&next, depth - 1, split - 1, on_subtree);
        }
    }
    nodes
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

    #[test]
    fn observed_counts_the_same_and_ticks_the_frontier() {
        // The split changes nothing about the answer, and the number of
        // frontier subtrees is `perft(split)` exactly — so a stage opened
        // with that total closes at 100% and never past it.
        for depth in 1..=7u32 {
            for split in 0..=depth {
                let mut ticks = 0u64;
                let n =
                    perft_observed(&GameState::standard_8x8(), depth, split, &mut || ticks += 1);
                assert_eq!(n, perft_standard(depth), "depth {depth} split {split}");
                assert_eq!(
                    ticks,
                    perft_standard(split),
                    "frontier size at depth {depth} split {split}"
                );
            }
        }
    }
}
