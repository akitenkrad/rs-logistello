//! # logistello-search
//!
//! Alpha-beta search engine for the Logistello reproduction:
//! NegaScout/PVS with a Zobrist transposition table, killer-move ordering,
//! iterative deepening, and selective forward pruning via single ProbCut and
//! Multi-ProbCut.
//!
//! - [`alphabeta`]      — NegaScout / PVS core.
//! - [`tt`]             — Zobrist-keyed transposition table.
//! - [`killer`]         — killer-move heuristic.
//! - [`iterative`]      — iterative deepening driver.
//! - [`probcut`]        — single ProbCut (Buro 1995 ICCA).
//! - [`multi_probcut`]  — Multi-ProbCut cascade (Buro 1997).

pub mod alphabeta;
pub mod iterative;
pub mod killer;
pub mod multi_probcut;
pub mod probcut;
pub mod tt;

pub use alphabeta::{INF, SearchConfig, SearchContext, negascout, terminal_score};
pub use iterative::{SearchResult, reference_negamax, search, search_with};
pub use killer::KillerTable;
pub use tt::{Bound, Entry, TranspositionTable};
