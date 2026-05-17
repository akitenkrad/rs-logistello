//! # logistello-book
//!
//! Opening-book learning and lookup for the Logistello reproduction
//! (Buro 1999, "Toward Opening Book Learning"; design doc §4.3.7).
//!
//! - [`learner`]     — self-play + Negamax back-propagation of book values
//!   ([`learn_book`], [`negamax_backpropagate`], [`apply_drawishness`],
//!   [`OpeningBook`], [`BookEntry`], [`BookConfig`], `OPB1` I/O).
//! - [`drawishness`] — the drawishness ("引き分けやすさ") `trap` term and the
//!   blended preference [`score`](drawishness::score) (Buro 1999).

pub mod drawishness;
pub mod learner;

pub use drawishness::{SCORE_BOUND, clamp_lambda, score, trap};
pub use learner::{
    BOOK_MAGIC, BOOK_VERSION, BookConfig, BookEntry, DEFAULT_BOOK_ENDGAME_EMPTIES,
    DEFAULT_MAX_BOOK_PLIES, EXPLORE_MARGIN, EXPLORE_PLIES, OpeningBook, apply_drawishness,
    learn_book, negamax_backpropagate,
};
