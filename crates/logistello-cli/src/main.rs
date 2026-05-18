//! # logistello — unified CLI for the Logistello reproduction.
//!
//! Only `perft` is wired end-to-end so far; every other subcommand is a
//! Phase-tagged placeholder that prints a "not yet implemented" message and
//! returns success.

use logistello_cli::edax::{EdaxConfig, EdaxGtpSession, GtpColor, resolve_edax_path};
use logistello_cli::elo::{LevelResult, Outcome};
use logistello_cli::eval_corr::{self, CorrSample};
use logistello_cli::extract;
use logistello_cli::glem_extract;
use logistello_cli::match_replay::{self, ReplayEngine};
use logistello_cli::probcut_fit;
use logistello_cli::results;
use logistello_cli::sweep;
use logistello_cli::wthor_murakami::{self, fmt_algebraic as fmt_alg, parse_algebraic};

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};

use logistello_book::{BookConfig, OpeningBook, learn_book};
use logistello_core::Zobrist;
use logistello_core::perft::perft_standard;
use logistello_eval::{
    BaseFeatureSpec, DiscDiffEval, EvalWeights, GlemEval, GlemModel, LeafEvaluator,
};
use logistello_search::alphabeta::{INF, SearchConfig, SearchContext, negascout};
use logistello_search::killer::KillerTable;
use logistello_search::tt::TranspositionTable;
use logistello_search::{EngineConfig, LogistelloPlayer, MultiProbCutConfig, ProbCutConfig};
use othello_core::{Color, GameState, Move};
use othello_engine::{EngineConfig as GameEngineConfig, GameEngine};
use othello_player::{GreedyPlayer, Player, RandomPlayer};

#[derive(Parser)]
#[command(
    name = "logistello",
    version,
    about = "Logistello (Buro 1994-1999) reproduction CLI",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Play a single game between two players to terminal (Phase 3).
    ///
    /// Each side is one of `engine` (Logistello: basic eval + iterative
    /// deepening + exact endgame), `random`, or `greedy`. Deterministic for
    /// a given seed (the random opponent is seeded; the engine itself is
    /// deterministic).
    Play {
        /// Black player: `engine`, `random`, or `greedy`.
        #[arg(long, default_value = "engine")]
        black: String,
        /// White player: `engine`, `random`, or `greedy`.
        #[arg(long, default_value = "random")]
        white: String,
        /// Selective-midgame iterative-deepening max depth (plies).
        #[arg(long, default_value_t = 8)]
        depth: u32,
        /// Switch to exact endgame play once empties <= this (design doc
        /// §4.5 B8; default 20, Edax-parity 10).
        #[arg(long, default_value_t = 20)]
        endgame_empties: u32,
        /// Seed for the random opponent (engine play is deterministic).
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Optional `LGW1` learned-weight file. When given, the `engine`
        /// player uses the Phase 4 `PatternEval` loaded from it instead of
        /// the Phase 3 `BasicEval` (design doc §4.3.3 / §4.4 B1-B4).
        #[arg(long)]
        eval_weights: Option<PathBuf>,
        /// Optional `GLM1` GLEM model file (Phase 7, design doc §4.3.6).
        /// When given, the `engine` player uses `GlemEval` (auto-generated
        /// conjunction features) as its midgame leaf evaluator instead of
        /// `PatternEval` / `BasicEval`. Mutually exclusive with
        /// `--eval-weights` (PatternEval vs GlemEval are alternative leaf
        /// evaluators); composes with `--probcut` / `--mpc`.
        #[arg(long, conflicts_with = "eval_weights")]
        glem_model: Option<PathBuf>,
        /// Enable single ProbCut for the `engine` player (design doc
        /// §4.3.4 / §4.5 B7). Off by default. Use `--no-probcut` to force
        /// it off explicitly.
        #[arg(long, overrides_with = "no_probcut", default_value_t = false)]
        probcut: bool,
        /// Explicitly disable ProbCut (the default; provided so it can
        /// override an earlier `--probcut`).
        #[arg(long)]
        no_probcut: bool,
        /// ProbCut per-σ confidence `T` (design doc §4.5 B7: 1.5).
        #[arg(long, default_value_t = 1.5)]
        probcut_t: f64,
        /// ProbCut params JSON (a `probcut-fit` output). Required for a
        /// meaningful `--probcut` run (the built-in default has no fitted
        /// coefficients, so ProbCut would just fall through).
        #[arg(long)]
        probcut_params: Option<PathBuf>,
        /// Enable Multi-ProbCut for the `engine` player (design doc §4.3.5 /
        /// §4.5 B7). Off by default. When on it supersedes single ProbCut at
        /// the cascade heights (`3..=13`). Composes with `--eval-weights`
        /// (production path = PatternEval + MPC + exact endgame).
        #[arg(long, default_value_t = false)]
        mpc: bool,
        /// Multi-ProbCut params JSON (a `probcut-fit --mpc-cascade` output).
        /// Required for a meaningful `--mpc` run.
        #[arg(long)]
        mpc_params: Option<PathBuf>,
        /// Optional learned opening book (`OPB1`, a `learn-book` output;
        /// Phase 8, design doc §4.3.7). When given, the `engine` player
        /// consults the book first: if the current position is booked (and
        /// the booked move is legal) it plays the booked move instead of
        /// searching; once play leaves the booked opening it falls back to
        /// the normal search (composes with every other flag).
        #[arg(long)]
        book: Option<PathBuf>,
    },
    /// Benchmark the Phase 2 search from the standard opening using the
    /// trivial disc-difference evaluator. `--probcut` / `--mpc` run the same
    /// nominal depth with single ProbCut / Multi-ProbCut enabled so the
    /// speedup vs the exact search is observable (`probcut_speedup`, design
    /// doc §4.3.8 / §5).
    BenchSearch {
        /// Maximum iterative-deepening depth (plies).
        #[arg(long, default_value_t = 8)]
        depth: u32,
        /// Enable single ProbCut at the same nominal depth (design doc
        /// §4.3.4 / §4.5 B7). Off by default.
        #[arg(long, overrides_with = "no_probcut", default_value_t = false)]
        probcut: bool,
        /// Explicitly disable ProbCut (the default).
        #[arg(long)]
        no_probcut: bool,
        /// ProbCut per-σ confidence `T` (design doc §4.5 B7: 1.5).
        #[arg(long, default_value_t = 1.5)]
        probcut_t: f64,
        /// ProbCut params JSON (a `probcut-fit` output).
        #[arg(long)]
        probcut_params: Option<PathBuf>,
        /// Enable Multi-ProbCut at the same nominal depth (design doc
        /// §4.3.5 / §4.5 B7). Off by default.
        #[arg(long, default_value_t = false)]
        mpc: bool,
        /// Multi-ProbCut params JSON (a `probcut-fit --mpc-cascade` output).
        #[arg(long)]
        mpc_params: Option<PathBuf>,
        /// Run a full-vs-single-ProbCut-vs-MPC comparison at `--depth` from
        /// the same position and report node/time and speedup factors (the
        /// §5 comparisons; design doc §4.3.8 / §5). Uses `--probcut-params`
        /// for the single-ProbCut run and `--mpc-params` for the MPC run
        /// (each optional; a run is skipped with a note if its file is
        /// absent).
        #[arg(long, default_value_t = false)]
        speedup: bool,
        /// Plies of seeded random play before benchmarking (a midgame
        /// position exercises h=8 ProbCut / cascade nodes; 0 = standard
        /// opening).
        #[arg(long, default_value_t = 0)]
        from_plies: usize,
        /// Seed for the `--from-plies` random walk.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Optional `GLM1` GLEM model (Phase 7, design doc §4.3.6). When
        /// given, the benchmark uses `GlemEval` as the leaf evaluator
        /// instead of the trivial disc-difference one (composes with
        /// `--probcut` / `--mpc`).
        #[arg(long)]
        glem_model: Option<PathBuf>,
    },
    /// Run self-play games for data generation (Phase 4).
    Selfplay,
    /// Extract training positions for evaluation-weight learning
    /// (Phase 4b; design doc §4.4 B3, §4.5 B5). Emits the `PEX1` columnar
    /// binary the Python `train-eval` tool consumes (see EXTRACT_FORMAT.md).
    Extract {
        /// Corpus: `selfplay` (RandomPlayer self-play, seeded, no external
        /// data) or `wthor` (real `.wtb` expert games).
        #[arg(long, default_value = "selfplay")]
        source: String,
        /// Directory containing `.wtb` files (required for `--source wthor`).
        #[arg(long)]
        wthor_dir: Option<PathBuf>,
        /// Number of self-play games (`--source selfplay`) / max WTHOR games.
        #[arg(long, default_value_t = 2000)]
        games: usize,
        /// Self-play RNG seed (deterministic).
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Skip positions with more than this many empty squares (drop the
        /// very opening). Omit to keep every non-terminal position.
        #[arg(long)]
        max_empties_skip: Option<u32>,
        /// Output `PEX1` file.
        #[arg(long)]
        output: PathBuf,
    },
    /// Extract GLEM base-literal training positions (Phase 7; design doc
    /// §4.3.6). Rust owns the base-literal extraction (mirrors Phase-4b
    /// `extract`'s "Rust owns canonicalisation"); emits the `GLX1` columnar
    /// binary the Python `train-glem` tool consumes (see GLEM_FORMAT.md).
    GlemExtract {
        /// Comma list of base-feature families
        /// (`cell64`,`mobility`,`corner`); order = the spec id recorded in
        /// `GLX1` / `GLM1`.
        #[arg(long, default_value = "cell64,mobility,corner")]
        base_features: String,
        /// Corpus: `selfplay` (RandomPlayer self-play, seeded, no external
        /// data) or `wthor` (real `.wtb` expert games).
        #[arg(long, default_value = "selfplay")]
        source: String,
        /// Directory containing `.wtb` files (required for `--source wthor`).
        #[arg(long)]
        wthor_dir: Option<PathBuf>,
        /// Number of self-play games (`--source selfplay`) / max WTHOR games.
        #[arg(long, default_value_t = 2000)]
        games: usize,
        /// Self-play RNG seed (deterministic).
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Skip positions with more than this many empty squares (drop the
        /// very opening). Omit to keep every non-terminal position.
        #[arg(long)]
        max_empties_skip: Option<u32>,
        /// Output `GLX1` file.
        #[arg(long)]
        output: PathBuf,
    },
    /// Fit ProbCut / Multi-ProbCut `(a, b, σ)` coefficients (Phase 5/6;
    /// design doc §4.3.4 / §4.3.5 / §4.5 B7). For every sampled position it
    /// measures the true `v_x = NegaScout(x)` for each needed depth under
    /// the production TT discipline (ProbCut/MPC OFF), stratifies by disc
    /// phase (`< 36` vs `≥ 36`), and fits OLS `v_h = a·v_d + b` per group.
    ///
    /// Without `--mpc-cascade`: a single-pair `(d, h)` fit → a
    /// `ProbCutConfig` JSON (Phase 5, unchanged). With `--mpc-cascade`: an
    /// independent OLS per `(disc-phase, h, d)` cascade cell → a
    /// `MultiProbCutConfig` JSON (Phase 6).
    ProbcutFit {
        /// Shallow:deep depth pair `d:h` (design doc §4.5 B7 single
        /// ProbCut = `4:8`). Used when `--mpc-cascade` is absent.
        #[arg(long, default_value = "4:8")]
        single_pair: String,
        /// Multi-ProbCut cascade spec `h:d1[:d2],...` (design doc §4.5 B7,
        /// canonical = `3:1,4:2,5:1,6:2,7:3,8:4,9:3:5,10:4:6,11:3:5,12:4,13:5`).
        /// When given, a `(disc-phase, h, d)` cascade fit is run instead of
        /// the single-pair fit and a `MultiProbCutConfig` JSON is written.
        #[arg(long)]
        mpc_cascade: Option<String>,
        /// Corpus: `selfplay` (seeded `RandomPlayer` self-play, no external
        /// data) or `wthor` (real `.wtb` expert games).
        #[arg(long, default_value = "selfplay")]
        source: String,
        /// Directory with `.wtb` files (required for `--source wthor`).
        #[arg(long)]
        wthor_dir: Option<PathBuf>,
        /// Number of sample positions to fit on.
        #[arg(long, default_value_t = 5000)]
        samples: usize,
        /// Self-play RNG seed (deterministic).
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Optional `LGW1` learned-weight file: fit with `PatternEval`
        /// instead of `BasicEval` (design doc §4.5 B7 targets R² > 0.96
        /// with the pattern evaluator).
        #[arg(long)]
        eval_weights: Option<PathBuf>,
        /// Per-σ confidence `T` written into the (single) `ProbCutConfig`
        /// (design doc §4.5 B7: 1.5). Ignored for `--mpc-cascade` (the MPC
        /// config carries the canonical 2-phase production thresholds
        /// 1.0 / 1.4).
        #[arg(long, default_value_t = 1.5)]
        probcut_t: f64,
        /// Output JSON path (`ProbCutConfig`- or, with `--mpc-cascade`,
        /// `MultiProbCutConfig`-compatible).
        #[arg(long)]
        output: PathBuf,
    },
    /// Extract the Murakami-1997 gold games from a raw WThor DB (Phase 9b;
    /// design doc §4.5 B5). Parses every `.wtb` in `--wthor-dir`, resolves
    /// player ids via the sibling `.jou`, keeps games where one player is
    /// "Logistello" and the other "Murakami" (case-insensitive), and writes
    /// the tiny committed JSON gold set. Reports the count + names found
    /// honestly (expects 6 for 1997).
    MurakamiExtract {
        /// Directory containing the FFO `.wtb` + `WTHOR.JOU` (fetch via
        /// `scripts/fetch_wthor.sh`; gitignored `data/wthor/`).
        #[arg(long, default_value = "data/wthor")]
        wthor_dir: PathBuf,
        /// Output JSON (the committed gold set, e.g.
        /// `tests/data/murakami_1997.json`).
        #[arg(long, default_value = "tests/data/murakami_1997.json")]
        output: PathBuf,
    },
    /// Replay a recorded match (Murakami 1997) and report our engine's
    /// move-match rate (Phase 9b; design doc §4.3.8
    /// `move_match_rate_murakami`, §5 ≥80% on main positions).
    MatchReplay {
        /// The committed gold-set JSON (`murakami-extract` output).
        #[arg(long, default_value = "tests/data/murakami_1997.json")]
        games: PathBuf,
        /// `LGW1` learned weights for the `PatternEval` (the production
        /// midgame evaluator). Omit to use the Phase-3 `BasicEval`.
        #[arg(long)]
        eval_weights: Option<PathBuf>,
        /// Optional Phase-8 opening book (`OPB1`).
        #[arg(long)]
        book: Option<PathBuf>,
        /// Selective-midgame iterative-deepening depth (plies).
        #[arg(long, default_value_t = 8)]
        depth: u32,
        /// Exact-endgame switch threshold (empties; design doc §4.5 B8).
        #[arg(long, default_value_t = 20)]
        endgame_empties: u32,
        /// Optional Multi-ProbCut params JSON (a `probcut-fit` output).
        #[arg(long)]
        mpc_params: Option<PathBuf>,
        /// Inclusive "main position" ply window low bound (design doc §5).
        #[arg(long, default_value_t = match_replay::MAIN_PLY_LO)]
        main_lo: u32,
        /// Inclusive "main position" ply window high bound.
        #[arg(long, default_value_t = match_replay::MAIN_PLY_HI)]
        main_hi: u32,
        /// Output CSV (also copied into `results/<ts>/`).
        #[arg(long, default_value = "results/murakami_replay.csv")]
        output: PathBuf,
    },
    /// Estimate ELO versus Edax at various levels via the direct-GTP
    /// full-game driver (Phase 9b; design doc §4.3.8 `elo_vs_edax_level_N`).
    /// Bounded by default (levels 1,3 × 2 games); the full sweep is a
    /// documented README command.
    EloVsEdax {
        /// Path to the Edax binary (default: gitignored `.edax/edax`).
        #[arg(long, default_value = EdaxConfig::DEFAULT_EDAX_PATH)]
        edax_path: PathBuf,
        /// Comma list of Edax `-level N` strengths (default `1,3`).
        #[arg(long, default_value = "1,3")]
        edax_levels: String,
        /// Games per level (colour-balanced; rounded up to even). Default 2.
        #[arg(long, default_value_t = 2)]
        num_games_per_level: u32,
        /// `LGW1` weights for our engine's `PatternEval` (else `BasicEval`).
        #[arg(long)]
        eval_weights: Option<PathBuf>,
        /// Our engine's selective-midgame depth (kept small for the demo).
        #[arg(long, default_value_t = 6)]
        depth: u32,
        /// Our engine's exact-endgame switch threshold (empties).
        #[arg(long, default_value_t = 16)]
        endgame_empties: u32,
        /// Determinism seed (recorded; our engine + book-off Edax are
        /// already deterministic, so runs reproduce regardless).
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Output CSV (also copied into `results/<ts>/`).
        #[arg(long, default_value = "results/elo_vs_edax.csv")]
        output: PathBuf,
    },
    /// Pearson correlation of our `PatternEval` value vs Edax's evaluation
    /// on a bounded position set (Phase 9b; design doc §4.3.8
    /// `eval_correlation_edax`, Edax as ground truth).
    EvalCorrelationEdax {
        /// Path to the Edax binary (default: gitignored `.edax/edax`).
        #[arg(long, default_value = EdaxConfig::DEFAULT_EDAX_PATH)]
        edax_path: PathBuf,
        /// Edax fixed strength used for its `genmove`-based evaluation.
        #[arg(long, default_value_t = 6)]
        edax_level: u32,
        /// `LGW1` weights for our `PatternEval` (else `BasicEval`).
        #[arg(long)]
        eval_weights: Option<PathBuf>,
        /// Number of positions to sample (bounded; default 24).
        #[arg(long, default_value_t = 24)]
        positions: usize,
        /// Seed for the deterministic position walk.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Output CSV (also copied into `results/<ts>/`).
        #[arg(long, default_value = "results/eval_correlation_edax.csv")]
        output: PathBuf,
    },
    /// Learn the opening book via self-play + Negamax back-propagation +
    /// drawishness (Phase 8; design doc §4.3.7 / Buro 1999). Writes the
    /// explicit little-endian `OPB1` book (see `BOOK_FORMAT.md`).
    /// Deterministic for a fixed `--seed`.
    LearnBook {
        /// Number of self-play games (design doc §5.1: 10000).
        #[arg(long, default_value_t = 1000)]
        num_games: u32,
        /// Per-move self-play search depth (design doc §5.1: 24; §6
        /// `--book-depth-values {12,18,24,30}`).
        #[arg(long, default_value_t = 24)]
        depth: u32,
        /// Drawishness blend `λ ∈ [0,0.5]` (design doc §5.1: 0.3; §6 range
        /// `0.0..0.5`). Clamped into range.
        #[arg(long, default_value_t = 0.3)]
        drawishness: f64,
        /// RNG seed for the (deterministic) exploration policy.
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Cap on the parent ply still booked (opening-only book bound).
        #[arg(long, default_value_t = 20)]
        max_book_plies: u32,
        /// Empties floor below which positions are not booked (keeps the
        /// book clear of the Phase-3 exact endgame).
        #[arg(long, default_value_t = 30)]
        book_endgame_empties: u32,
        /// Optional `LGW1` learned-weight file: self-play with the Phase 4
        /// `PatternEval` instead of the Phase 3 `BasicEval`.
        #[arg(long)]
        eval_weights: Option<PathBuf>,
        /// Output `OPB1` book path.
        #[arg(long)]
        output: PathBuf,
    },
    /// Run a §6 sensitivity-analysis parameter sweep (Phase 10; design
    /// doc §6 / §5.1 / §4.2). Sweep **one** §6 parameter at a time (the §6
    /// table is one row per parameter); each condition is run for `--runs`
    /// independent seeded trials. Writes `results/<ts>/sweep_config.json`
    /// (the resolved spec) + `metrics.csv` (one row per trial) and refreshes
    /// `results/latest`. With no parameter flag it lists the sweepable
    /// parameters and exits non-zero.
    ///
    /// Metric per parameter (see `crate::sweep`): ProbCut-T / depth-pair /
    /// multi-stages / max-depth / TT-size / endgame-empties → search node
    /// count (+ value/time/nps/dispatch); eval-stages → held-out eval
    /// absolute error; GLEM order/support → generated feature count;
    /// book-depth → learned-book position count; drawishness → self-play
    /// score.
    Sweep {
        /// ProbCut confidence `T` grid min (design doc §6: 1.0..2.5).
        #[arg(long)]
        probcut_t_min: Option<f64>,
        /// ProbCut `T` grid max.
        #[arg(long)]
        probcut_t_max: Option<f64>,
        /// ProbCut `T` grid step (design doc §6: 0.25).
        #[arg(long)]
        probcut_t_step: Option<f64>,
        /// ProbCut depth-pair candidates `d:h,...` (design doc §6:
        /// `1:5,3:7,5:9,3:9,5:11`).
        #[arg(long)]
        probcut_depth_pairs_values: Option<String>,
        /// Multi-ProbCut cascade-stage candidates (design doc §6: `1..5`).
        #[arg(long)]
        multi_stages_values: Option<String>,
        /// Evaluation-stage-count candidates (design doc §6:
        /// `1,5,10,13,20,30`).
        #[arg(long)]
        eval_stages_values: Option<String>,
        /// GLEM max-conjunction-order candidates (design doc §6: `1..4`).
        #[arg(long)]
        glem_max_order_values: Option<String>,
        /// GLEM support-threshold grid min as a `log10` exponent (design
        /// doc §6: `1e-4..1e-2` ⇒ pass `-4`).
        #[arg(long)]
        glem_support_min: Option<f64>,
        /// GLEM support-threshold grid max as a `log10` exponent (`1e-2` ⇒
        /// pass `-2`).
        #[arg(long)]
        glem_support_max: Option<f64>,
        /// GLEM support-threshold `log10` step (design doc §6: 0.25).
        #[arg(long)]
        glem_support_step: Option<f64>,
        /// Iterative-deepening max-depth candidates (design doc §6:
        /// `6,8,10,12`).
        #[arg(long)]
        max_depth_values: Option<String>,
        /// Transposition-table size candidates as `log2` exponents (design
        /// doc §6: `2^20..2^26` ⇒ `20,22,24,26`).
        #[arg(long)]
        tt_size_values: Option<String>,
        /// Opening-book self-play depth candidates (design doc §6:
        /// `12,18,24,30`).
        #[arg(long)]
        book_depth_values: Option<String>,
        /// Exact-endgame switch (empties) candidates (design doc §6 / B8:
        /// `8,10,16,20,24`; default 20).
        #[arg(long)]
        endgame_empties_values: Option<String>,
        /// Drawishness-weight grid min (design doc §6: 0.0..0.5).
        #[arg(long)]
        drawishness_min: Option<f64>,
        /// Drawishness-weight grid max.
        #[arg(long)]
        drawishness_max: Option<f64>,
        /// Drawishness-weight grid step (design doc §6: 0.1).
        #[arg(long)]
        drawishness_step: Option<f64>,
        /// Independent seeded trials per condition (design doc §6: 30;
        /// Edax-based conditions ≥50). Keep small for the demo.
        #[arg(long, default_value_t = 30)]
        runs: u32,
        /// Base RNG seed; the per-trial seed is derived explicitly as
        /// `seed + condition_index*runs + trial_index` (no wall-clock, so
        /// `metrics.csv` is byte-identical across runs).
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Verify search-tree size against known Othello perft values (Phase 1).
    Perft {
        /// Depth (plies) to enumerate from the standard starting position.
        #[arg(long, default_value_t = 6)]
        depth: u32,
    },
}

/// Formats a move as algebraic notation (`f5`) or `pass`.
fn fmt_move(m: Move) -> String {
    match m {
        Move::Pass => "pass".to_string(),
        Move::Place(c) => {
            let file = (b'a' + c.col) as char;
            let rank = c.row + 1;
            format!("{file}{rank}")
        }
    }
}

/// A `LogistelloPlayer` with an optional Phase-8 opening book consulted
/// first (design doc §4.3.7). If the current position is booked and the
/// booked move is currently legal, the booked move is played without
/// searching; once play leaves the booked opening, the wrapped engine
/// searches normally. The book is opening-only (bounded by plies/empties at
/// learn time, re-checked by `OpeningBook::probe`), so it can never override
/// the Phase-3 exact endgame.
struct BookedEngine {
    inner: LogistelloPlayer,
    book: Option<OpeningBook>,
}

impl Player for BookedEngine {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn color(&self) -> Color {
        self.inner.color()
    }
    fn select_move(&mut self, state: &GameState) -> Result<Move, othello_player::PlayerError> {
        if let Some(b) = &self.book
            && let Some(m) = b.probe(state)
        {
            return Ok(m);
        }
        self.inner.select_move(state)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// A `play`-CLI player: one concrete variant per `--black` / `--white`
/// kind. `GameEngine::run` is generic over `Sized` players, so we dispatch
/// through this enum instead of a `Box<dyn Player>` (trait objects are
/// unsized and `&mut dyn Player` does not itself implement `Player`).
enum CliPlayer {
    // Every variant is boxed so the enum stays pointer-sized regardless of
    // the (large) `LogistelloPlayer`, which owns a transposition table
    // (clippy::large_enum_variant).
    Engine(Box<BookedEngine>),
    Random(Box<RandomPlayer>),
    Greedy(Box<GreedyPlayer>),
}

impl Player for CliPlayer {
    fn name(&self) -> &str {
        match self {
            CliPlayer::Engine(p) => p.name(),
            CliPlayer::Random(p) => p.name(),
            CliPlayer::Greedy(p) => p.name(),
        }
    }

    fn color(&self) -> Color {
        match self {
            CliPlayer::Engine(p) => p.color(),
            CliPlayer::Random(p) => p.color(),
            CliPlayer::Greedy(p) => p.color(),
        }
    }

    fn select_move(&mut self, state: &GameState) -> Result<Move, othello_player::PlayerError> {
        match self {
            CliPlayer::Engine(p) => p.select_move(state),
            CliPlayer::Random(p) => p.select_move(state),
            CliPlayer::Greedy(p) => p.select_move(state),
        }
    }

    fn reset(&mut self) {
        match self {
            CliPlayer::Engine(p) => p.reset(),
            CliPlayer::Random(p) => p.reset(),
            CliPlayer::Greedy(p) => p.reset(),
        }
    }
}

/// Which midgame leaf evaluator the `engine` player should use. `Basic` is
/// the Phase 3 default; `Pattern` is the Phase 4 `PatternEval`
/// (`--eval-weights`); `Glem` is the Phase 7 `GlemEval` (`--glem-model`).
/// `Pattern` and `Glem` are mutually exclusive (auto vs manual features).
#[derive(Clone)]
enum LeafChoice<'a> {
    Basic,
    Pattern(&'a logistello_eval::PatternEval),
    Glem(&'a GlemEval),
}

/// Builds a [`CliPlayer`] of the requested kind for `color`.
///
/// `engine` -> [`LogistelloPlayer`] with `cfg` (using `PatternEval` /
/// `GlemEval` per `leaf`, else `BasicEval`); `random` -> seeded
/// [`RandomPlayer`]; `greedy` -> [`GreedyPlayer`].
fn make_player(
    kind: &str,
    color: Color,
    cfg: &EngineConfig,
    seed: u64,
    leaf: &LeafChoice<'_>,
    book: Option<&OpeningBook>,
) -> Result<CliPlayer> {
    match kind {
        "engine" => {
            let p = match leaf {
                LeafChoice::Pattern(pe) => {
                    LogistelloPlayer::with_pattern(color, cfg.clone(), (*pe).clone())
                }
                LeafChoice::Glem(ge) => {
                    LogistelloPlayer::with_glem(color, cfg.clone(), (*ge).clone())
                }
                LeafChoice::Basic => LogistelloPlayer::new(color, cfg.clone()),
            };
            Ok(CliPlayer::Engine(Box::new(BookedEngine {
                inner: p,
                book: book.cloned(),
            })))
        }
        "random" => Ok(CliPlayer::Random(Box::new(RandomPlayer::with_seed(
            color, seed,
        )))),
        "greedy" => Ok(CliPlayer::Greedy(Box::new(GreedyPlayer::with_color(color)))),
        other => bail!("unknown player kind '{other}' (want engine|random|greedy)"),
    }
}

/// Builds the [`ProbCutConfig`] for a CLI run.
///
/// `--no-probcut` / the default keeps ProbCut OFF (Phase 2/3 behaviour
/// unchanged). `--probcut` turns it on; if `--probcut-params FILE` is given
/// the fitted coefficients are loaded from it (and `T` overridden by
/// `--probcut-t`), otherwise the built-in canonical-but-unfitted config is
/// used (which simply falls through — a warning is printed).
fn build_probcut(
    enable: bool,
    no_probcut: bool,
    t: f64,
    params: Option<&PathBuf>,
) -> Result<ProbCutConfig> {
    if no_probcut || !enable {
        return Ok(ProbCutConfig::default()); // OFF
    }
    let mut cfg = match params {
        Some(p) => ProbCutConfig::load_json(p)
            .map_err(|e| anyhow::anyhow!("load probcut params {}: {e}", p.display()))?,
        None => {
            eprintln!(
                "warning: --probcut without --probcut-params: no fitted \
                 coefficients, ProbCut will fall through to the exact search"
            );
            ProbCutConfig::default()
        }
    };
    cfg.enabled = true;
    cfg.t = t;
    Ok(cfg)
}

/// Builds the [`MultiProbCutConfig`] for a CLI run (Phase 6).
///
/// The default keeps MPC OFF (Phase 2/3/5 behaviour unchanged). `--mpc`
/// turns it on; if `--mpc-params FILE` is given the fitted cascade
/// coefficients are loaded from it, otherwise the built-in unfitted config
/// is used (which simply falls through — a warning is printed). The loaded
/// config keeps its canonical 2-phase production thresholds (1.0 / 1.4).
fn build_multi_probcut(enable: bool, params: Option<&PathBuf>) -> Result<MultiProbCutConfig> {
    if !enable {
        return Ok(MultiProbCutConfig::default()); // OFF
    }
    let mut cfg = match params {
        Some(p) => MultiProbCutConfig::load_json(p)
            .map_err(|e| anyhow::anyhow!("load mpc params {}: {e}", p.display()))?,
        None => {
            eprintln!(
                "warning: --mpc without --mpc-params: no fitted cascade \
                 coefficients, Multi-ProbCut will fall through to the exact \
                 search"
            );
            MultiProbCutConfig::default()
        }
    };
    cfg.enabled = true;
    Ok(cfg)
}

/// Parses a `d:h` depth pair string.
fn parse_pair(s: &str) -> Result<(u32, u32)> {
    let mut it = s.split(':');
    let d = it
        .next()
        .and_then(|x| x.trim().parse().ok())
        .ok_or_else(|| anyhow::anyhow!("bad --single-pair '{s}' (want d:h)"))?;
    let h = it
        .next()
        .and_then(|x| x.trim().parse().ok())
        .ok_or_else(|| anyhow::anyhow!("bad --single-pair '{s}' (want d:h)"))?;
    if it.next().is_some() {
        bail!("--single-pair '{s}' has too many ':' parts (want d:h)");
    }
    if d >= h {
        bail!("--single-pair d:h needs d < h (got {d}:{h})");
    }
    Ok((d, h))
}

/// Random-walk `plies` placement moves from the standard start (seeded;
/// forced passes followed transparently per §4.5 B6).
fn walk_position(plies: usize, seed: u64) -> GameState {
    use rand::SeedableRng;
    use rand::seq::SliceRandom;
    use rand_chacha::ChaCha20Rng;
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut s = GameState::standard_8x8();
    let mut made = 0;
    while made < plies {
        if s.is_terminal() {
            break;
        }
        let mv = s.legal_moves();
        if mv.is_empty() {
            s.apply_move(Move::Pass).expect("pass legal when stuck");
            continue;
        }
        let m = *mv.choose(&mut rng).expect("non-empty");
        s.apply_move(m).expect("legal move applies");
        made += 1;
    }
    s
}

/// One fixed-depth NegaScout from `root` with the given evaluator and
/// ProbCut + MPC config, returning `(value, nodes, elapsed_secs)`.
fn bench_one_with<E: LeafEvaluator>(
    root: &GameState,
    depth: u32,
    eval: &E,
    pc: ProbCutConfig,
    mpc: MultiProbCutConfig,
) -> (i32, u64, f64) {
    let mut tt = TranspositionTable::new();
    let mut killers = KillerTable::new();
    let z = Zobrist::new();
    let cfg = SearchConfig {
        probcut: pc,
        multi_probcut: mpc,
        ..SearchConfig::default()
    };
    let mut ctx = SearchContext {
        evaluator: eval,
        tt: &mut tt,
        killers: &mut killers,
        zobrist: &z,
        config: cfg,
        nodes: 0,
    };
    let start = Instant::now();
    let v = negascout(&mut ctx, root, -INF, INF, depth, 0);
    (v, ctx.nodes, start.elapsed().as_secs_f64())
}

// ============================ Phase 9b helpers ============================

/// Builds the Phase-9b engine config (selective-midgame depth + exact
/// endgame threshold + optional Multi-ProbCut), shared by `match-replay`
/// and `elo-vs-edax`.
fn phase9_engine_config(depth: u32, endgame_empties: u32, mpc: MultiProbCutConfig) -> EngineConfig {
    EngineConfig {
        max_depth: depth,
        endgame_empties,
        multi_probcut: mpc,
        ..EngineConfig::default()
    }
}

/// Optional learned `PatternEval` from an `LGW1` file.
fn load_pattern(p: Option<&PathBuf>) -> Result<Option<logistello_eval::PatternEval>> {
    match p {
        Some(path) => {
            let w = EvalWeights::load(path)
                .map_err(|e| anyhow::anyhow!("load {}: {e}", path.display()))?;
            Ok(Some(logistello_eval::PatternEval::new(w)))
        }
        None => Ok(None),
    }
}

/// A `LogistelloPlayer` (Pattern or Basic leaf) + optional Phase-8 book,
/// adapted to [`ReplayEngine`]: `pick` consults the book then the search;
/// `score` returns the search value (side-to-move POV, disc scale) — the
/// same exact Phase 3/4 path the rest of the CLI uses.
struct EngineReplay {
    inner: LogistelloPlayer,
    book: Option<OpeningBook>,
}

impl EngineReplay {
    fn build(
        color: Color,
        cfg: &EngineConfig,
        pattern: Option<&logistello_eval::PatternEval>,
        book: Option<&OpeningBook>,
    ) -> Self {
        let inner = match pattern {
            Some(pe) => LogistelloPlayer::with_pattern(color, cfg.clone(), pe.clone()),
            None => LogistelloPlayer::new(color, cfg.clone()),
        };
        Self {
            inner,
            book: book.cloned(),
        }
    }
}

impl ReplayEngine for EngineReplay {
    fn pick(&mut self, state: &GameState) -> Result<Move> {
        if let Some(b) = &self.book
            && let Some(m) = b.probe(state)
        {
            return Ok(m);
        }
        Ok(self.inner.decide(state).1.best_move)
    }
    fn score(&mut self, state: &GameState) -> i32 {
        self.inner.decide(state).1.value
    }
    fn reset(&mut self) {
        use othello_player::Player;
        self.inner.reset();
    }
}

/// GTP colour for an `othello_core::Color`.
fn gtp_color(c: Color) -> GtpColor {
    match c {
        Color::Black => GtpColor::Black,
        Color::White => GtpColor::White,
    }
}

/// Plays ONE full game: `our_color` is our engine; the other side is the
/// already-started `EdaxGtpSession` (book-off, deterministic). Returns the
/// final `(black_discs, white_discs)`. Drives standard GTP directly — Edax
/// commits its own `genmove`; we only `play` *our* moves to Edax (never
/// re-`play` Edax's move → no `? wrong color`).
fn play_one_edax_game(
    sess: &mut EdaxGtpSession,
    our: &mut EngineReplay,
    our_color: Color,
) -> Result<(u32, u32)> {
    use othello_player::Player;
    sess.new_game()
        .map_err(|e| anyhow::anyhow!("edax new_game: {e}"))?;
    our.inner.reset();
    let mut state = GameState::standard_8x8();
    let mut guard = 0;
    while !state.is_terminal() {
        guard += 1;
        if guard > 200 {
            bail!("game did not terminate in 200 plies (protocol desync?)");
        }
        let stm = state.side_to_move;
        if state.must_pass() {
            // The side to move has no legal move. We mirror a pass on our
            // board. Edax tracks this itself: if it is Edax's turn it will
            // return "pass" from genmove; if it is ours we tell Edax we
            // passed. Either way our board applies Pass.
            if stm == our_color {
                sess.play(gtp_color(stm), "pass")
                    .map_err(|e| anyhow::anyhow!("edax play pass: {e}"))?;
            } else {
                let mv = sess
                    .genmove(gtp_color(stm))
                    .map_err(|e| anyhow::anyhow!("edax genmove: {e}"))?;
                if mv != "pass" {
                    bail!("Edax did not pass when it had no move (got {mv:?})");
                }
            }
            state
                .apply_move(Move::Pass)
                .map_err(|e| anyhow::anyhow!("apply forced pass: {e}"))?;
            continue;
        }
        if stm == our_color {
            let m = our.pick(&state)?;
            let alg = fmt_alg(m);
            sess.play(gtp_color(stm), &alg)
                .map_err(|e| anyhow::anyhow!("edax play {alg}: {e}"))?;
            state
                .apply_move(m)
                .map_err(|e| anyhow::anyhow!("apply our move {alg}: {e}"))?;
        } else {
            let mv = sess
                .genmove(gtp_color(stm))
                .map_err(|e| anyhow::anyhow!("edax genmove: {e}"))?;
            if mv == "resign" {
                // Edax resigned: score the rest as a loss for Edax by
                // counting current discs (Edax forfeits).
                break;
            }
            let parsed =
                parse_algebraic(&mv).map_err(|e| anyhow::anyhow!("parse edax move {mv:?}: {e}"))?;
            if !state.legal_moves().contains(&parsed) {
                bail!("Edax returned ILLEGAL move {mv:?} (protocol desync)");
            }
            state
                .apply_move(parsed)
                .map_err(|e| anyhow::anyhow!("apply edax move {mv:?}: {e}"))?;
        }
    }
    Ok((
        state.board.count(Color::Black),
        state.board.count(Color::White),
    ))
}

/// Edax plays *both* sides to terminal from the position reached by
/// replaying `line` from the standard start (book-off, fixed level),
/// returning final `(black_discs, white_discs)`. Used by
/// `eval-correlation-edax` as Edax's strong value estimate of that position
/// (its resulting disc outcome under self-play) — the ground-truth signal
/// we correlate our static leaf eval against. GTP has no `setboard`, so the
/// position is established by replaying `line` move-by-move into Edax first.
fn edax_selfplay_from_line(sess: &mut EdaxGtpSession, line: &[Move]) -> Result<(u32, u32)> {
    sess.new_game()
        .map_err(|e| anyhow::anyhow!("edax new_game: {e}"))?;
    let mut state = GameState::standard_8x8();
    for m in line {
        let stm = state.side_to_move;
        let alg = fmt_alg(*m);
        sess.play(gtp_color(stm), &alg)
            .map_err(|e| anyhow::anyhow!("edax replay {alg}: {e}"))?;
        state
            .apply_move(*m)
            .map_err(|e| anyhow::anyhow!("replay {alg}: {e}"))?;
    }
    let mut guard = 0;
    while !state.is_terminal() {
        guard += 1;
        if guard > 200 {
            bail!("edax self-play did not terminate");
        }
        let stm = state.side_to_move;
        if state.must_pass() {
            let mv = sess
                .genmove(gtp_color(stm))
                .map_err(|e| anyhow::anyhow!("edax genmove: {e}"))?;
            if mv != "pass" {
                bail!("Edax did not pass when forced (got {mv:?})");
            }
            state.apply_move(Move::Pass).ok();
            continue;
        }
        let mv = sess
            .genmove(gtp_color(stm))
            .map_err(|e| anyhow::anyhow!("edax genmove: {e}"))?;
        if mv == "resign" {
            break;
        }
        let parsed = parse_algebraic(&mv).map_err(|e| anyhow::anyhow!("parse {mv:?}: {e}"))?;
        if !state.legal_moves().contains(&parsed) {
            bail!("Edax ILLEGAL move {mv:?} in self-play");
        }
        state
            .apply_move(parsed)
            .map_err(|e| anyhow::anyhow!("apply {mv:?}: {e}"))?;
    }
    Ok((
        state.board.count(Color::Black),
        state.board.count(Color::White),
    ))
}

/// Deterministic move line of `walk_position(plies, seed)` (same RNG/body),
/// returned so Edax can be replayed to that exact position.
fn walk_line(plies: usize, seed: u64) -> (GameState, Vec<Move>) {
    use rand::SeedableRng;
    use rand::seq::SliceRandom;
    use rand_chacha::ChaCha20Rng;
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut s = GameState::standard_8x8();
    let mut line = Vec::new();
    let mut made = 0;
    while made < plies {
        if s.is_terminal() {
            break;
        }
        let mv = s.legal_moves();
        if mv.is_empty() {
            s.apply_move(Move::Pass).expect("pass legal when stuck");
            line.push(Move::Pass);
            continue;
        }
        let m = *mv.choose(&mut rng).expect("non-empty");
        s.apply_move(m).expect("legal move applies");
        line.push(m);
        made += 1;
    }
    (s, line)
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Play {
            black,
            white,
            depth,
            endgame_empties,
            seed,
            eval_weights,
            glem_model,
            probcut,
            no_probcut,
            probcut_t,
            probcut_params,
            mpc,
            mpc_params,
            book,
        } => {
            let pc = build_probcut(probcut, no_probcut, probcut_t, probcut_params.as_ref())?;
            let mpc_cfg = build_multi_probcut(mpc, mpc_params.as_ref())?;
            let cfg = EngineConfig {
                max_depth: depth,
                endgame_empties,
                probcut: pc,
                multi_probcut: mpc_cfg.clone(),
                ..EngineConfig::default()
            };
            let learned = match &eval_weights {
                Some(path) => {
                    let w = EvalWeights::load(path)
                        .map_err(|e| anyhow::anyhow!("load {}: {e}", path.display()))?;
                    Some(logistello_eval::PatternEval::new(w))
                }
                None => None,
            };
            let glem = match &glem_model {
                Some(path) => {
                    let m = GlemModel::load(path)
                        .map_err(|e| anyhow::anyhow!("load {}: {e}", path.display()))?;
                    Some(GlemEval::new(m))
                }
                None => None,
            };
            let leaf = match (learned.as_ref(), glem.as_ref()) {
                (_, Some(g)) => LeafChoice::Glem(g),
                (Some(p), None) => LeafChoice::Pattern(p),
                (None, None) => LeafChoice::Basic,
            };
            let book_loaded = match &book {
                Some(path) => Some(
                    OpeningBook::load(path)
                        .map_err(|e| anyhow::anyhow!("load book {}: {e}", path.display()))?,
                ),
                None => None,
            };
            let mut black_player = make_player(
                &black,
                Color::Black,
                &cfg,
                seed,
                &leaf,
                book_loaded.as_ref(),
            )?;
            let mut white_player = make_player(
                &white,
                Color::White,
                &cfg,
                seed,
                &leaf,
                book_loaded.as_ref(),
            )?;

            let mut engine = GameEngine::new(GameEngineConfig::standard())?;
            let result = engine
                .run(&mut black_player, &mut white_player)
                .map_err(|e| anyhow::anyhow!("game engine error: {e}"))?;

            let moves: Vec<String> = engine
                .history()
                .moves()
                .iter()
                .map(|m| fmt_move(*m))
                .collect();

            let winner = match result.winner {
                Some(Color::Black) => "black",
                Some(Color::White) => "white",
                None => "draw",
            };
            let eval_kind = match (&glem_model, &eval_weights) {
                (Some(g), _) => format!("glem({})", g.display()),
                (None, Some(p)) => format!("pattern({})", p.display()),
                (None, None) => "basic".to_string(),
            };
            let pc_kind = if pc.enabled {
                format!("on(T={},d={},h={})", pc.t, pc.d, pc.h)
            } else {
                "off".to_string()
            };
            let mpc_kind = if mpc_cfg.enabled {
                format!(
                    "on(T={}/{},cells={})",
                    mpc_cfg.t_lt36,
                    mpc_cfg.t_ge36,
                    mpc_cfg.cell_count()
                )
            } else {
                "off".to_string()
            };
            let book_kind = match (&book, &book_loaded) {
                (Some(p), Some(b)) => format!("on({}, {} pos)", p.display(), b.len()),
                _ => "off".to_string(),
            };
            println!(
                "play black={black} white={white} depth={depth} \
                 endgame_empties={endgame_empties} seed={seed} eval={eval_kind} \
                 probcut={pc_kind} mpc={mpc_kind} book={book_kind}"
            );
            println!("moves: {}", moves.join(" "));
            println!(
                "final score: black={} white={} winner={winner}",
                result.black, result.white
            );

            // §4.2 single-run output contract: a timestamped run dir with
            // `config.json` (the resolved single condition) + `metrics.csv`
            // (one summary row) so `show-experiment-settings` / `visualize`
            // can consume a `run`. Best-effort (a results-dir failure must
            // not fail the game itself).
            if let Ok(run) = results::new_run_dir(std::path::Path::new("results")) {
                let cfg = serde_json::json!({
                    "command": "play",
                    "black": black,
                    "white": white,
                    "depth": depth,
                    "endgame_empties": endgame_empties,
                    "seed": seed,
                    "eval": eval_kind,
                    "probcut": pc_kind,
                    "mpc": mpc_kind,
                    "book": book_kind,
                });
                let _ = std::fs::write(
                    run.join("config.json"),
                    serde_json::to_string_pretty(&cfg).unwrap_or_default(),
                );
                let plies = moves.len();
                let metrics = format!(
                    "black_score,white_score,winner,plies,seed\n{},{},{winner},{plies},{seed}\n",
                    result.black, result.white
                );
                let _ = std::fs::write(run.join("metrics.csv"), metrics);
                println!("results: {}", run.display());
            }
        }
        Command::BenchSearch {
            depth,
            probcut,
            no_probcut,
            probcut_t,
            probcut_params,
            mpc,
            mpc_params,
            speedup,
            from_plies,
            seed,
            glem_model,
        } => {
            let root = walk_position(from_plies, seed);
            let nps = |nodes: u64, secs: f64| -> u64 {
                if secs > 0.0 {
                    (nodes as f64 / secs) as u64
                } else {
                    0
                }
            };
            // GLEM leaf evaluator selectable; default = trivial disc-diff.
            let glem_eval = match &glem_model {
                Some(path) => {
                    let m = GlemModel::load(path)
                        .map_err(|e| anyhow::anyhow!("load {}: {e}", path.display()))?;
                    Some(GlemEval::new(m))
                }
                None => None,
            };
            // Single closure so every call site (full / single / mpc) uses
            // the same chosen evaluator without duplicating the match.
            let bench_one = |root: &GameState,
                             depth: u32,
                             pc: ProbCutConfig,
                             mc: MultiProbCutConfig|
             -> (i32, u64, f64) {
                match &glem_eval {
                    Some(g) => bench_one_with(root, depth, g, pc, mc),
                    None => bench_one_with(root, depth, &DiscDiffEval, pc, mc),
                }
            };

            if speedup {
                // The §5 comparison (design doc §4.3.8 / §5): same nominal
                // depth and root, full search vs single ProbCut vs
                // Multi-ProbCut. Single / MPC runs use their respective
                // params files (each skipped with a note if absent).
                let (v_off, n_off, t_off) = bench_one(
                    &root,
                    depth,
                    ProbCutConfig::default(),
                    MultiProbCutConfig::default(),
                );
                println!("bench-search depth={depth} from_plies={from_plies} seed={seed}");
                println!(
                    "  full  : value={v_off} nodes={n_off} time={t_off:.3}s nps={}",
                    nps(n_off, t_off)
                );

                // Single ProbCut.
                let (n_single, single_done) = if probcut_params.is_some() {
                    let pc = build_probcut(true, false, probcut_t, probcut_params.as_ref())?;
                    let (v, n, t) = bench_one(&root, depth, pc, MultiProbCutConfig::default());
                    let r = n as f64 / n_off.max(1) as f64;
                    println!(
                        "  single: value={v} nodes={n} time={t:.3}s nps={} \
                         node_ratio(vs full)={r:.4} reduction={:.1}% \
                         time_speedup(full/single)={:.2}x value_delta={}",
                        nps(n, t),
                        (1.0 - r) * 100.0,
                        if t > 0.0 { t_off / t } else { 0.0 },
                        v - v_off
                    );
                    (n, true)
                } else {
                    println!("  single: skipped (no --probcut-params)");
                    (0, false)
                };

                // Multi-ProbCut.
                if mpc_params.is_some() {
                    let mc = build_multi_probcut(true, mpc_params.as_ref())?;
                    let (v, n, t) = bench_one(&root, depth, ProbCutConfig::default(), mc);
                    let r_full = n as f64 / n_off.max(1) as f64;
                    println!(
                        "  mpc   : value={v} nodes={n} time={t:.3}s nps={} \
                         node_ratio(vs full)={r_full:.4} reduction={:.1}% \
                         time_speedup(full/mpc)={:.2}x value_delta={}",
                        nps(n, t),
                        (1.0 - r_full) * 100.0,
                        if t > 0.0 { t_off / t } else { 0.0 },
                        v - v_off
                    );
                    if single_done && n_single > 0 {
                        let factor = n_single as f64 / n.max(1) as f64;
                        println!(
                            "  speedup_ordering: nodes(full={n_off}) > \
                             nodes(single={n_single}) ? {}  ;  MPC vs single \
                             node_factor={factor:.3}x (§5 target ≈1.5×, ≥1.3×)",
                            n_off > n_single
                        );
                    }
                } else {
                    println!("  mpc   : skipped (no --mpc-params)");
                }
            } else {
                let pc = build_probcut(probcut, no_probcut, probcut_t, probcut_params.as_ref())?;
                let mc = build_multi_probcut(mpc, mpc_params.as_ref())?;
                let (value, nodes, secs) = bench_one(&root, depth, pc, mc.clone());
                let pc_kind = if pc.enabled { "on" } else { "off" };
                let mpc_kind = if mc.enabled { "on" } else { "off" };
                let eval_kind = match &glem_model {
                    Some(p) => format!("glem({})", p.display()),
                    None => "discdiff".to_string(),
                };
                println!(
                    "bench-search depth={depth} from_plies={from_plies} \
                     eval={eval_kind} probcut={pc_kind} mpc={mpc_kind} \
                     value={value} nodes={nodes} time={secs:.3}s nps={}",
                    nps(nodes, secs)
                );
            }
        }
        Command::Selfplay => {
            println!("selfplay: not yet implemented (Phase 4)");
        }
        Command::Extract {
            source,
            wthor_dir,
            games,
            seed,
            max_empties_skip,
            output,
        } => {
            let records = match source.as_str() {
                "selfplay" => extract::extract_selfplay(games, seed, max_empties_skip)?,
                "wthor" => {
                    let dir = wthor_dir.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("--wthor-dir is required for --source wthor")
                    })?;
                    extract::extract_wthor(dir, Some(games), max_empties_skip)?
                }
                other => bail!("unknown --source '{other}' (want selfplay|wthor)"),
            };
            extract::write_pex1(&output, &records)?;
            println!(
                "extract source={source} games<={games} \
                 records={} output={}",
                records.len(),
                output.display()
            );
        }
        Command::GlemExtract {
            base_features,
            source,
            wthor_dir,
            games,
            seed,
            max_empties_skip,
            output,
        } => {
            let spec =
                BaseFeatureSpec::parse(&base_features).map_err(|e| anyhow::anyhow!("{e}"))?;
            let records = match source.as_str() {
                "selfplay" => glem_extract::extract_selfplay(&spec, games, seed, max_empties_skip)?,
                "wthor" => {
                    let dir = wthor_dir.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("--wthor-dir is required for --source wthor")
                    })?;
                    glem_extract::extract_wthor(&spec, dir, Some(games), max_empties_skip)?
                }
                other => bail!("unknown --source '{other}' (want selfplay|wthor)"),
            };
            glem_extract::write_glx1(&output, &spec, &records)?;
            println!(
                "glem-extract base-features={} source={source} \
                 games<={games} n_literals={} records={} output={}",
                spec.to_spec_string(),
                spec.n_literals(),
                records.len(),
                output.display()
            );
        }
        Command::ProbcutFit {
            single_pair,
            mpc_cascade,
            source,
            wthor_dir,
            samples,
            seed,
            eval_weights,
            probcut_t,
            output,
        } => {
            let src = match source.as_str() {
                "selfplay" => probcut_fit::Source::Selfplay {
                    games: samples, // one position per game minimum; capped by samples
                    seed,
                },
                "wthor" => {
                    let dir = wthor_dir.ok_or_else(|| {
                        anyhow::anyhow!("--wthor-dir is required for --source wthor")
                    })?;
                    probcut_fit::Source::Wthor {
                        dir,
                        max_games: samples,
                    }
                }
                other => bail!("unknown --source '{other}' (want selfplay|wthor)"),
            };
            let eval_kind = match &eval_weights {
                Some(p) => format!("pattern({})", p.display()),
                None => "basic".to_string(),
            };

            match &mpc_cascade {
                // ---- Phase 6: Multi-ProbCut cascade fit ------------------
                Some(spec) => {
                    let cascade = probcut_fit::parse_mpc_cascade(spec)?;
                    let res = probcut_fit::run_mpc_fit(
                        &cascade,
                        &src,
                        samples,
                        eval_weights.as_deref(),
                        &output,
                    )?;
                    println!(
                        "probcut-fit mpc-cascade={spec} source={source} \
                         samples<={samples} seed={seed} eval={eval_kind} \
                         T(prod)=1.0/1.4 cells={} output={}",
                        res.cells.len(),
                        output.display()
                    );
                    println!("  phase    h   d        a          b      sigma       R2      n");
                    for c in &res.cells {
                        let ph = match c.phase {
                            logistello_search::DiscPhase::Lt36 => "<36 ",
                            logistello_search::DiscPhase::Ge36 => ">=36",
                        };
                        println!(
                            "  {ph}  {:>3} {:>3}  {:>9.5} {:>9.5} {:>9.5} {:>8.5} {:>6}",
                            c.h, c.d, c.fit.a, c.fit.b, c.fit.sigma, c.fit.r2, c.fit.n
                        );
                    }
                }
                // ---- Phase 5: single-pair fit (unchanged) ----------------
                None => {
                    let (d, h) = parse_pair(&single_pair)?;
                    let res = probcut_fit::run_fit(
                        d,
                        h,
                        &src,
                        samples,
                        eval_weights.as_deref(),
                        probcut_t,
                        &output,
                    )?;
                    println!(
                        "probcut-fit single-pair={d}:{h} source={source} \
                         samples<={samples} seed={seed} eval={eval_kind} \
                         T={probcut_t} output={}",
                        output.display()
                    );
                    for (name, ph) in [("phase<36", res.lt36), ("phase>=36", res.ge36)] {
                        println!(
                            "  {name}: a={:.6} b={:.6} sigma={:.6} R2={:.6} n={}",
                            ph.a, ph.b, ph.sigma, ph.r2, ph.n
                        );
                    }
                }
            }
        }
        Command::MurakamiExtract { wthor_dir, output } => {
            let set = wthor_murakami::extract_murakami(&wthor_dir).map_err(|e| {
                anyhow::anyhow!(
                    "murakami-extract from {}: {e} (did you run \
                     scripts/fetch_wthor.sh?)",
                    wthor_dir.display()
                )
            })?;
            if let Some(p) = output.parent() {
                std::fs::create_dir_all(p).ok();
            }
            std::fs::write(&output, wthor_murakami::to_json(&set)?)?;
            println!(
                "murakami-extract wthor-dir={} year={} games_found={} \
                 output={}",
                wthor_dir.display(),
                set.year,
                set.games.len(),
                output.display()
            );
            for g in &set.games {
                let (logi, opp) = if g.logistello_is_black {
                    (&g.black_name, &g.white_name)
                } else {
                    (&g.white_name, &g.black_name)
                };
                println!(
                    "  game#{:<5} B={:?} W={:?} logistello={} ({}) \
                     result(black_discs)={} theoretical={} plies={}",
                    g.wtb_index,
                    g.black_name,
                    g.white_name,
                    logi,
                    if g.logistello_is_black {
                        "black"
                    } else {
                        "white"
                    },
                    g.result_black_discs,
                    g.result_theoretical_black_discs,
                    g.moves.len()
                );
                let _ = opp;
            }
            if set.games.len() == 6 {
                println!(
                    "OK: exactly 6 Logistello-vs-Murakami games (the 1997 \
                     gold set; design doc §4.5 B5 / §5)."
                );
            } else {
                println!(
                    "NOTE: expected 6 (1997 Murakami match) but found {} — \
                     reported honestly (no fabrication).",
                    set.games.len()
                );
            }
        }
        Command::MatchReplay {
            games,
            eval_weights,
            book,
            depth,
            endgame_empties,
            mpc_params,
            main_lo,
            main_hi,
            output,
        } => {
            let set = wthor_murakami::load_set(&games)
                .map_err(|e| anyhow::anyhow!("load {}: {e}", games.display()))?;
            let mpc = build_multi_probcut(mpc_params.is_some(), mpc_params.as_ref())?;
            let cfg = phase9_engine_config(depth, endgame_empties, mpc);
            let pattern = load_pattern(eval_weights.as_ref())?;
            let book_loaded = match &book {
                Some(p) => Some(
                    OpeningBook::load(p)
                        .map_err(|e| anyhow::anyhow!("load book {}: {e}", p.display()))?,
                ),
                None => None,
            };
            // One engine per colour (each owns its TT); replay_game resets.
            let mut all = Vec::new();
            for g in &set.games {
                let color = if g.logistello_is_black {
                    Color::Black
                } else {
                    Color::White
                };
                let mut eng =
                    EngineReplay::build(color, &cfg, pattern.as_ref(), book_loaded.as_ref());
                all.extend(match_replay::replay_game(g, &mut eng, main_lo, main_hi)?);
            }
            let summary = match_replay::summarize(&all);
            if let Some(p) = output.parent() {
                std::fs::create_dir_all(p).ok();
            }
            match_replay::write_csv(&output, &all)?;
            // Mirror into results/<ts>/.
            if let Ok(run) = results::new_run_dir(std::path::Path::new("results")) {
                let _ = match_replay::write_csv(&run.join("murakami_replay.csv"), &all);
            }
            let eval_kind = eval_weights.as_ref().map_or_else(
                || "basic".to_string(),
                |p| format!("pattern({})", p.display()),
            );
            println!(
                "match-replay games={} engine={} depth={depth} \
                 endgame_empties={endgame_empties} main_window=[{main_lo},{main_hi}] \
                 output={}",
                set.games.len(),
                eval_kind,
                output.display()
            );
            println!(
                "  scored_decisions={} matched={} overall_match_rate={:.4}",
                summary.total,
                summary.matched,
                summary.overall_rate()
            );
            println!(
                "  main_decisions={} main_matched={} main_match_rate={:.4} \
                 (§5 target ≥0.80 with the production-scale Logistello-2 \
                 weights; medium-trained weights are weaker — see README)",
                summary.main_total,
                summary.main_matched,
                summary.main_rate()
            );
        }
        Command::EloVsEdax {
            edax_path,
            edax_levels,
            num_games_per_level,
            eval_weights,
            depth,
            endgame_empties,
            seed,
            output,
        } => {
            let path = resolve_edax_path(Some(&edax_path));
            let levels: Vec<u32> = edax_levels
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if levels.is_empty() {
                bail!("--edax-levels parsed to nothing (want e.g. 1,3)");
            }
            // Colour-balanced: round up to an even number of games.
            let n = num_games_per_level.max(1);
            let n = if n % 2 == 0 { n } else { n + 1 };
            let probe = EdaxConfig::new(&path, *levels.first().unwrap());
            if !probe.is_available() {
                println!(
                    "elo-vs-edax SKIPPED: no Edax binary at {} (gitignored \
                     install absent — run scripts/setup_edax.sh). Harness + \
                     unit tests still cover the logic; nothing fabricated.",
                    path.display()
                );
                return Ok(());
            }
            let pattern = load_pattern(eval_weights.as_ref())?;
            let cfg = phase9_engine_config(depth, endgame_empties, MultiProbCutConfig::default());
            let mut rows: Vec<LevelResult> = Vec::new();
            let t0 = Instant::now();
            for &lvl in &levels {
                let ecfg =
                    EdaxConfig::new(&path, lvl).with_timeout(std::time::Duration::from_secs(120));
                let mut sess = EdaxGtpSession::start(&ecfg)
                    .map_err(|e| anyhow::anyhow!("start Edax level {lvl}: {e}"))?;
                let mut lr = LevelResult::new(lvl);
                for game_idx in 0..n {
                    // Alternate: even = our engine Black, odd = our White.
                    let our_color = if game_idx % 2 == 0 {
                        Color::Black
                    } else {
                        Color::White
                    };
                    let mut eng = EngineReplay::build(our_color, &cfg, pattern.as_ref(), None);
                    let (bd, wd) = play_one_edax_game(&mut sess, &mut eng, our_color)?;
                    let (our_d, edax_d) = if our_color == Color::Black {
                        (bd, wd)
                    } else {
                        (wd, bd)
                    };
                    lr.record(Outcome::from_discs(our_d, edax_d));
                }
                println!(
                    "  level {lvl}: games={} W-D-L={}-{}-{} score_rate={:.4} \
                     win_rate={:.4} elo_delta={:+.1}",
                    lr.games,
                    lr.wins,
                    lr.draws,
                    lr.losses,
                    lr.score_rate(),
                    lr.win_rate(),
                    lr.elo_delta()
                );
                rows.push(lr);
            }
            let wall = t0.elapsed().as_secs_f64();
            if let Some(p) = output.parent() {
                std::fs::create_dir_all(p).ok();
            }
            let mut csv =
                String::from("level,games,wins,draws,losses,score_rate,win_rate,elo_delta\n");
            for r in &rows {
                csv.push_str(&format!(
                    "{},{},{},{},{},{:.6},{:.6},{:.3}\n",
                    r.level,
                    r.games,
                    r.wins,
                    r.draws,
                    r.losses,
                    r.score_rate(),
                    r.win_rate(),
                    r.elo_delta()
                ));
            }
            std::fs::write(&output, &csv)?;
            if let Ok(run) = results::new_run_dir(std::path::Path::new("results")) {
                let _ = std::fs::write(run.join("elo_vs_edax.csv"), &csv);
            }
            let eval_kind = eval_weights.as_ref().map_or_else(
                || "basic".to_string(),
                |p| format!("pattern({})", p.display()),
            );
            println!(
                "elo-vs-edax edax={} levels={:?} games_per_level={n} \
                 engine={} depth={depth} seed={seed} wall={wall:.1}s output={} \
                 (BOUNDED demo — Edax level→absolute strength is qualitative; \
                 full sweep is a documented README command)",
                path.display(),
                levels,
                eval_kind,
                output.display()
            );
        }
        Command::EvalCorrelationEdax {
            edax_path,
            edax_level,
            eval_weights,
            positions,
            seed,
            output,
        } => {
            let path = resolve_edax_path(Some(&edax_path));
            let probe = EdaxConfig::new(&path, edax_level);
            if !probe.is_available() {
                println!(
                    "eval-correlation-edax SKIPPED: no Edax binary at {} \
                     (gitignored install absent — run scripts/setup_edax.sh). \
                     Pearson logic is unit-tested; nothing fabricated.",
                    path.display()
                );
                return Ok(());
            }
            let pattern = load_pattern(eval_weights.as_ref())?;
            // Sample distinct midgame positions via a deterministic walk;
            // for each, our PatternEval/BasicEval leaf value vs Edax's
            // ground-truth value = the disc-diff Edax reaches by playing
            // *both* sides to terminal from that position at the fixed
            // level (a strong estimate; GTP exposes no static-eval query,
            // so the solved self-play outcome is the principled signal),
            // side-to-move POV to match our leaf convention.
            let ecfg =
                EdaxConfig::new(&path, edax_level).with_timeout(std::time::Duration::from_secs(60));
            let mut samples: Vec<CorrSample> = Vec::new();
            let leaf_score = |s: &GameState| -> i32 {
                use logistello_eval::LeafEvaluator;
                match &pattern {
                    Some(pe) => pe.eval(s),
                    None => logistello_eval::BasicEval::default().eval(s),
                }
            };
            let mut sess =
                EdaxGtpSession::start(&ecfg).map_err(|e| anyhow::anyhow!("start Edax: {e}"))?;
            let mut idx = 0usize;
            for k in 0..positions {
                let plies = 8 + (k % 24);
                let (st, line) = walk_line(plies, seed.wrapping_add(k as u64));
                if st.is_terminal() || st.must_pass() {
                    continue;
                }
                let Ok((bd, wd)) = edax_selfplay_from_line(&mut sess, &line) else {
                    continue;
                };
                let edax_diff = match st.side_to_move {
                    Color::Black => bd as i32 - wd as i32,
                    Color::White => wd as i32 - bd as i32,
                };
                samples.push(CorrSample {
                    idx,
                    our_score: leaf_score(&st),
                    edax_score: edax_diff,
                });
                idx += 1;
            }
            let xs: Vec<f64> = samples.iter().map(|s| s.our_score as f64).collect();
            let ys: Vec<f64> = samples.iter().map(|s| s.edax_score as f64).collect();
            let r = eval_corr::pearson(&xs, &ys);
            if let Some(p) = output.parent() {
                std::fs::create_dir_all(p).ok();
            }
            eval_corr::write_csv(&output, &samples, r)?;
            if let Ok(run) = results::new_run_dir(std::path::Path::new("results")) {
                let _ = eval_corr::write_csv(&run.join("eval_correlation_edax.csv"), &samples, r);
            }
            let eval_kind = eval_weights.as_ref().map_or_else(
                || "basic".to_string(),
                |p| format!("pattern({})", p.display()),
            );
            let edax_disp = path.display();
            let out_disp = output.display();
            let r_disp = r.map_or_else(|| "NA".to_string(), |v| format!("{v:.4}"));
            let n = samples.len();
            println!(
                "eval-correlation-edax edax={edax_disp} level={edax_level} \
                 engine={eval_kind} n={n} pearson_r={r_disp} output={out_disp} \
                 (Edax as ground truth; bounded position count)"
            );
        }
        Command::LearnBook {
            num_games,
            depth,
            drawishness,
            seed,
            max_book_plies,
            book_endgame_empties,
            eval_weights,
            output,
        } => {
            let cfg = BookConfig {
                num_games,
                depth_limit: depth,
                drawishness,
                seed,
                max_book_plies,
                endgame_empties: book_endgame_empties,
            };
            let book = match &eval_weights {
                Some(path) => {
                    let w = EvalWeights::load(path)
                        .map_err(|e| anyhow::anyhow!("load {}: {e}", path.display()))?;
                    learn_book(&logistello_eval::PatternEval::new(w), &cfg)
                }
                None => learn_book(&logistello_eval::BasicEval::default(), &cfg),
            };
            book.save(&output)
                .map_err(|e| anyhow::anyhow!("save {}: {e}", output.display()))?;

            let eval_kind = match &eval_weights {
                Some(p) => format!("pattern({})", p.display()),
                None => "basic".to_string(),
            };
            let start = GameState::standard_8x8();
            let root_best = book
                .get(&start)
                .map(|e| fmt_move(e.best_move))
                .unwrap_or_else(|| "<none>".to_string());
            println!(
                "learn-book num_games={num_games} depth={depth} \
                 drawishness={drawishness} seed={seed} \
                 max_book_plies={max_book_plies} \
                 book_endgame_empties={book_endgame_empties} eval={eval_kind} \
                 output={}",
                output.display()
            );
            println!(
                "book size={} distinct_positions={} root_best_move={root_best}",
                book.to_bytes().len(),
                book.len()
            );
        }
        Command::Sweep {
            probcut_t_min,
            probcut_t_max,
            probcut_t_step,
            probcut_depth_pairs_values,
            multi_stages_values,
            eval_stages_values,
            glem_max_order_values,
            glem_support_min,
            glem_support_max,
            glem_support_step,
            max_depth_values,
            tt_size_values,
            book_depth_values,
            endgame_empties_values,
            drawishness_min,
            drawishness_max,
            drawishness_step,
            runs,
            seed,
        } => {
            // A `min[/max/step]` triple ⇒ `(min, max||min, step||1.0)`; a
            // bare `--x-min` (no max/step) is a degenerate single-point grid
            // (the expander returns just `[min]`).
            let triple = |min: Option<f64>,
                          max: Option<f64>,
                          step: Option<f64>|
             -> Option<(f64, f64, f64)> {
                min.map(|lo| (lo, max.unwrap_or(lo), step.unwrap_or(1.0)))
            };
            let spec = sweep::SweepSpec {
                probcut_t: triple(probcut_t_min, probcut_t_max, probcut_t_step),
                probcut_depth_pairs_values,
                multi_stages_values,
                eval_stages_values,
                glem_max_order_values,
                glem_support: triple(glem_support_min, glem_support_max, glem_support_step),
                max_depth_values,
                tt_size_values,
                book_depth_values,
                endgame_empties_values,
                drawishness: triple(drawishness_min, drawishness_max, drawishness_step),
                runs,
                seed,
            };
            let resolved = match spec.resolve() {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("sweep: {e}");
                    std::process::exit(2);
                }
            };
            println!(
                "sweep param={} metric={} conditions={} runs={runs} \
                 trials={} seed={seed}",
                resolved.axis.col(),
                resolved.axis.metric(),
                resolved.points.len(),
                resolved.points.len() * runs as usize,
            );
            let (run_dir, rows) = sweep::run_sweep(&resolved, std::path::Path::new("results"))?;
            println!(
                "wrote {} ({} rows) + {}",
                run_dir.join("metrics.csv").display(),
                rows.len(),
                run_dir.join("sweep_config.json").display(),
            );
            // A short per-condition mean of the headline metric so the run
            // is self-describing without the Python viz.
            use std::collections::BTreeMap;
            let mut by_val: BTreeMap<String, (f64, u64)> = BTreeMap::new();
            for r in &rows {
                let m = match resolved.axis.metric() {
                    "nodes" => r.nodes as f64,
                    "eval_abs_err" => r.eval_abs_err,
                    "n_features" => r.n_features as f64,
                    "book_positions" => r.book_positions as f64,
                    "selfplay_score" => r.selfplay_score as f64,
                    _ => 0.0,
                };
                let e = by_val.entry(r.value.clone()).or_insert((0.0, 0));
                e.0 += m;
                e.1 += 1;
            }
            for (v, (sum, n)) in by_val {
                println!(
                    "  {}={v:<10} mean_{}={:.4} (n={n})",
                    resolved.axis.col(),
                    resolved.axis.metric(),
                    sum / n.max(1) as f64
                );
            }
        }
        Command::Perft { depth } => {
            let nodes = perft_standard(depth);
            println!("perft(depth={depth}) = {nodes}");
        }
    }

    Ok(())
}
