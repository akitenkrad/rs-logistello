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
//! - [`endgame`]        — exact endgame solver (design doc §4.5 B8).
//! - [`engine`]         — three-phase decision driver + engine player.
//! - [`probcut`]        — single ProbCut (Buro 1995 ICCA).
//! - [`multi_probcut`]  — Multi-ProbCut cascade (Buro 1997).

pub mod alphabeta;
pub mod endgame;
pub mod engine;
pub mod iterative;
pub mod killer;
pub mod multi_probcut;
pub mod probcut;
pub mod tt;

pub use alphabeta::{INF, SearchConfig, SearchContext, negascout, terminal_score};
pub use endgame::{best_endgame_move, solve_exact, solve_wld};
pub use engine::{
    DEFAULT_ENDGAME_EMPTIES, DEFAULT_MAX_DEPTH, Dispatch, EngineConfig, EngineEval,
    LogistelloPlayer, decide_move, decide_move_with_dispatch,
};
pub use iterative::{SearchResult, reference_negamax, search, search_with};
pub use killer::KillerTable;
pub use multi_probcut::{
    DiscPhase, MPC_CASCADE, MpcStageParams, MultiProbCutConfig, stages_for_height,
};
pub use probcut::{PROBCUT_PHASE_SPLIT_DISCS, ProbCutConfig, ProbCutParams, probcut_bounds};
pub use tt::{Bound, Entry, TranspositionTable};
