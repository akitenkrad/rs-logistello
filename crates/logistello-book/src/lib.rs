//! # logistello-book
//!
//! Opening-book learning and lookup for the Logistello reproduction
//! (Buro 1999, "Toward Opening Book Learning").
//!
//! - [`learner`]     — self-play + Negamax back-propagation of book values.
//! - [`drawishness`] — drawishness ("引き分けやすさ") scoring of book lines.

pub mod drawishness;
pub mod learner;
