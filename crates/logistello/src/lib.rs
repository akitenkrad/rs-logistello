//! # logistello
//!
//! Umbrella library for the Logistello reproduction. It re-exports the
//! Logistello-specific crates so downstream code can depend on a single
//! `logistello` crate (and write `use logistello::...`) instead of pulling
//! in the individual `logistello-*` crates one by one.
//!
//! - [`core`]   — Zobrist hashing, perft, board constants (`logistello-core`).
//! - [`eval`]   — Edax-faithful pattern evaluation, 13 stages, GLEM
//!   (`logistello-eval`).
//! - [`search`] — NegaScout/PVS, transposition table, ProbCut,
//!   Multi-ProbCut, exact endgame (`logistello-search`).
//! - [`book`]   — self-play opening-book learning (`logistello-book`).
//!
//! The unified command-line interface lives in the separate
//! `logistello-cli` binary crate (binary name: `logistello`); a binary
//! crate is intentionally not re-exported by this library facade.
//!
//! The bitboard, move generation, WTHOR I/O, and Edax integration are
//! provided by the reusable [`rs-othello-sim`](https://github.com/akitenkrad/rs-othello-sim)
//! library, on which the re-exported crates depend.

#![forbid(unsafe_code)]

pub use logistello_book as book;
pub use logistello_core as core;
pub use logistello_eval as eval;
pub use logistello_search as search;

#[cfg(test)]
mod tests {
    //! Compile-time + value checks that the facade re-exports resolve to the
    //! sub-crates. The `pub use` lines above are themselves the proof that
    //! every sub-crate links; these assertions additionally pin a known
    //! value from `core` and `eval` so a moved public path is caught here.

    #[test]
    fn facade_reexports_resolve() {
        // logistello-core: standard Othello perft(1) == 4.
        assert_eq!(crate::core::perft::perft_standard(1), 4);
        // logistello-eval: design-doc B2 — 13 discs => stage 0.
        assert_eq!(crate::eval::stage::stage(13), 0);
        // logistello-search / logistello-book: referencing the re-exported
        // crate roots is sufficient; resolution here proves the facade
        // exposes them.
        #[allow(unused_imports)]
        use crate::{book as _book_root, search as _search_root};
    }
}
