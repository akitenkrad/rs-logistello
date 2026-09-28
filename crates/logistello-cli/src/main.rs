//! # logistello — unified CLI for the Logistello reproduction.
//!
//! Subcommands that **measure** something (`play`, `bench-search`,
//! `match-replay`, `elo-vs-edax`, `eval-correlation-edax`, `sweep`) record
//! themselves into a runvault run directory under `--results-root`; the
//! directory's name and identity are runvault's, not ours. Subcommands that
//! only *produce data or an artefact* (`extract`, `glem-extract`,
//! `probcut-fit`, `murakami-extract`, `learn-book`, `perft`) write their
//! `--output` file and record nothing: what they make is an input to a later
//! run, and it is that run that carries it (as a `Dataset` with the file's
//! content hash).
//!
//! ## Progress
//!
//! Every subcommand here can be given flags that put it well past a minute —
//! `play --depth 14` is over six minutes for one game, `bench-search
//! --depth 19` is 3m24s for one search — so each one reports what it is doing
//! at the granularity of its own work (`runvault::progress`). Lines go to
//! standard error; standard output is untouched. A subcommand that opens a
//! run reports through `Run::stage`, so its lines are mirrored into the run
//! and stop when the manifest is sealed; one that produces only an artefact,
//! and one whose work happens before `Run::start`, uses
//! `Progress::to_stderr()` — there is no run directory to mirror into.

use logistello_cli::edax::{EdaxConfig, EdaxGtpSession, GtpColor, resolve_edax_path};
use logistello_cli::elo::{LevelResult, Outcome};
use logistello_cli::eval_corr::{self, CorrSample};
use logistello_cli::extract;
use logistello_cli::glem_extract;
use logistello_cli::match_replay::{self, ReplayEngine};
use logistello_cli::probcut_fit;
use logistello_cli::record;
use logistello_cli::sweep;
use logistello_cli::wthor_murakami::{self, fmt_algebraic as fmt_alg, parse_algebraic};

use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use runvault::{Progress, Run, Stage};
use serde_json::json;

use logistello_book::{BookConfig, OpeningBook, learn_book_observed};
use logistello_core::Zobrist;
use logistello_core::perft::{perft_observed, perft_standard};
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
    /// Development run: write it under results/_scratch/ so it is never synced to the vault.
    #[arg(long, global = true)]
    scratch: bool,
    /// Root the runvault run directories are created under. Every
    /// measuring subcommand writes into `<root>/logistello/<run-slug>/`;
    /// runvault owns the naming.
    #[arg(long, global = true, default_value = "results")]
    results_root: PathBuf,
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
        /// Optional extra copy of the per-decision table as a CSV, written
        /// outside the run. The run itself records every decision as an
        /// `observation` event, so this is a convenience, not the record.
        #[arg(long)]
        output: Option<PathBuf>,
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
        /// Optional extra copy of the per-level table as a CSV, written
        /// outside the run. The run itself records one event per level.
        #[arg(long)]
        output: Option<PathBuf>,
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
        /// Optional extra copy of the per-position table as a CSV, written
        /// outside the run. The run itself records one event per sampled
        /// position.
        #[arg(long)]
        output: Option<PathBuf>,
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
    /// doc §6 / §5.1). Sweep **one** §6 parameter at a time (the §6 table is
    /// one row per parameter); each condition is run for `--runs` independent
    /// seeded trials. Recorded as a runvault sweep parent (the resolved grid)
    /// plus one child run per condition (its trials as events). With no
    /// parameter flag it lists the sweepable parameters and exits non-zero.
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

/// The stage both players of one `play` game tick.
///
/// `GameEngine::run` takes the two players for the length of the game, so
/// neither of them can borrow the caller's `Stage`; they share one and `main`
/// takes it back to close it. `Player` is `Send` (the engine's batch runner is
/// parallel), which is why this is an `Arc<Mutex<..>>` and not the
/// `Rc<RefCell<..>>` the same shape uses where no such bound exists.
type MoveObserver = Arc<Mutex<Option<Stage>>>;

/// A [`CliPlayer`] that ticks a shared stage once per move it decides.
///
/// One game is the unit `play` reports at its end, and one game is minutes:
/// the *move* is the unit inside it, and a move is exactly one call here.
/// Forced passes are substituted by the engine without asking a player and so
/// are not counted — the count is of decisions, which is where the time is.
struct ObservedPlayer {
    inner: CliPlayer,
    moves: MoveObserver,
}

impl Player for ObservedPlayer {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn color(&self) -> Color {
        self.inner.color()
    }

    fn select_move(&mut self, state: &GameState) -> Result<Move, othello_player::PlayerError> {
        let m = self.inner.select_move(state)?;
        if let Ok(mut guard) = self.moves.lock()
            && let Some(stage) = guard.as_mut()
        {
            stage.tick();
        }
        Ok(m)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// How many leaf evaluations one tick of a search stage stands for.
///
/// A `Stage::tick` reads the clock, and a leaf evaluation is a few dozen
/// nanoseconds, so ticking every leaf would tax the very benchmark
/// `bench-search` exists to report. A block of a million lands several times a
/// second at the rates measured here, which is far finer than the thirty
/// seconds a stage may otherwise stay silent for.
const LEAVES_PER_TICK: u64 = 1 << 20;

/// A leaf evaluator that says how far a single search has got.
///
/// One `negascout` at `--depth 19` is 3m24s inside a single call, and the only
/// unit below it is the node. There is no denominator for those — the tree is
/// what the search is measuring — and splitting the root by hand would change
/// the node count and the value the subcommand prints, which is the number
/// being reported. So the stage is **unbounded** and counts leaf evaluations
/// in blocks of [`LEAVES_PER_TICK`]: not a share of the work, but a rate, and
/// a rate is what tells a reader that the search is running rather than
/// wedged.
///
/// The wrapper returns `inner`'s value unchanged, so the evaluator is still
/// the pure function of the position the search's minimax invariants need.
struct ObservedEval<'a, E: LeafEvaluator> {
    inner: &'a E,
    seen: Cell<u64>,
    stage: RefCell<Stage>,
}

impl<'a, E: LeafEvaluator> ObservedEval<'a, E> {
    fn new(inner: &'a E, stage: Stage) -> Self {
        Self {
            inner,
            seen: Cell::new(0),
            stage: RefCell::new(stage),
        }
    }

    /// The stage back, for the caller to close where the search ends.
    fn into_stage(self) -> Stage {
        self.stage.into_inner()
    }
}

impl<E: LeafEvaluator> LeafEvaluator for ObservedEval<'_, E> {
    fn eval(&self, state: &GameState) -> i32 {
        let seen = self.seen.get() + 1;
        self.seen.set(seen);
        if seen.is_multiple_of(LEAVES_PER_TICK) {
            self.stage.borrow_mut().tick();
        }
        self.inner.eval(state)
    }
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
    on_ply: &mut impl FnMut(),
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
            on_ply();
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
            on_ply();
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
            on_ply();
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
    let Cli {
        command,
        results_root,
        scratch,
    } = Cli::parse();

    match command {
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
            // The game is played before there is a run to mirror into (the
            // run's parameters and metrics are the game's result), so the
            // lines go to standard error only. The stage is unbounded: a
            // game's length is not known until it is over — at most sixty
            // placements, but how many of them are decisions rather than
            // forced passes is what the game decides.
            let moves_seen: MoveObserver = Arc::new(Mutex::new(Some(
                Progress::to_stderr().unbounded_stage("moves"),
            )));
            let mut black_player = ObservedPlayer {
                inner: make_player(
                    &black,
                    Color::Black,
                    &cfg,
                    seed,
                    &leaf,
                    book_loaded.as_ref(),
                )?,
                moves: Arc::clone(&moves_seen),
            };
            let mut white_player = ObservedPlayer {
                inner: make_player(
                    &white,
                    Color::White,
                    &cfg,
                    seed,
                    &leaf,
                    book_loaded.as_ref(),
                )?,
                moves: Arc::clone(&moves_seen),
            };

            let mut engine = GameEngine::new(GameEngineConfig::standard())?;
            let result = engine
                .run(&mut black_player, &mut white_player)
                .map_err(|e| anyhow::anyhow!("game engine error: {e}"))?;
            if let Some(stage) = moves_seen.lock().ok().and_then(|mut g| g.take()) {
                stage.close();
            }

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

            // The condition is what the flags resolved to; the *contents*
            // of the weight / model / book / MPC files decide the result
            // just as much, and a path does not identify them — they go in
            // as datasets, hashed.
            let parameters = json!({
                "black": black,
                "white": white,
                "depth": depth,
                "endgame_empties": endgame_empties,
                "seed": seed,
                "eval": match (&glem_model, &eval_weights) {
                    (Some(_), _) => "glem",
                    (None, Some(_)) => "pattern",
                    (None, None) => "basic",
                },
                "probcut": { "enabled": pc.enabled, "t": pc.t, "d": pc.d, "h": pc.h },
                "mpc": {
                    "enabled": mpc_cfg.enabled,
                    "t_lt36": mpc_cfg.t_lt36,
                    "t_ge36": mpc_cfg.t_ge36,
                    "cells": mpc_cfg.cell_count(),
                },
                "book": book_loaded.is_some(),
            });
            let mut data = Vec::new();
            if let Some(path) = &eval_weights {
                data.push(record::learned_artifact("eval-weights", path)?);
            }
            if let Some(path) = &glem_model {
                data.push(record::learned_artifact("glem-model", path)?);
            }
            if let Some(path) = &probcut_params {
                data.push(record::learned_artifact("probcut-params", path)?);
            }
            if let Some(path) = &mpc_params {
                data.push(record::learned_artifact("mpc-params", path)?);
            }
            if let Some(path) = &book {
                data.push(record::learned_artifact("opening-book", path)?);
            }

            let mut run = Run::start(
                record::options(
                    "play",
                    record::DOMAIN_SIMULATION,
                    &results_root,
                    record::replication_play(),
                    scratch,
                )
                .parameters(&parameters)
                .context("runvault: parameters の組み立てに失敗")?
                .seed_pointers(["/seed"])
                .master_seed(seed)
                .data(data),
            )
            .context("runvault: run の開始に失敗")?;

            // `plies` counts the recorded moves (passes included), which is
            // what the old metrics.csv column held. The terminal event's `t`
            // counts discs placed — that is the number the 60-square budget
            // bounds.
            let placements = moves.iter().filter(|m| *m != "pass").count() as u64;
            run.log_metrics(
                "run",
                &[
                    ("black_score", f64::from(result.black)),
                    ("white_score", f64::from(result.white)),
                    ("plies", moves.len() as f64),
                ],
            )
            .context("run スコープの指標の記録に失敗")?;
            record::log_game(&mut run, "game", placements, winner, &moves.join(" "))?;
            let dir = run.finish().context("runvault: run の終了に失敗")?;
            println!("results: {}", dir.display());
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
            // The measurement is decided by the flags plus the *contents*
            // of the fitted-coefficient files, so those go in hashed.
            let parameters = json!({
                "depth": depth,
                "from_plies": from_plies,
                "seed": seed,
                "eval": if glem_model.is_some() { "glem" } else { "discdiff" },
                "mode": if speedup { "speedup" } else { "single" },
                "probcut": {
                    "enabled": speedup && probcut_params.is_some()
                        || !speedup && probcut && !no_probcut,
                    "t": probcut_t,
                },
                "mpc": { "enabled": if speedup { mpc_params.is_some() } else { mpc } },
            });
            let mut data = Vec::new();
            if let Some(path) = &glem_model {
                data.push(record::learned_artifact("glem-model", path)?);
            }
            if let Some(path) = &probcut_params {
                data.push(record::learned_artifact("probcut-params", path)?);
            }
            if let Some(path) = &mpc_params {
                data.push(record::learned_artifact("mpc-params", path)?);
            }
            let mut run = Run::start(
                record::options(
                    "bench-search",
                    record::DOMAIN_SIMULATION,
                    &results_root,
                    record::replication_bench_search(),
                    scratch,
                )
                .parameters(&parameters)
                .context("runvault: parameters の組み立てに失敗")?
                .seed_pointers(["/seed"])
                .master_seed(seed)
                .data(data),
            )
            .context("runvault: run の開始に失敗")?;

            // Single closure so every call site (full / single / mpc) uses
            // the same chosen evaluator without duplicating the match.
            //
            // Each call takes its **own** stage rather than sharing one over
            // the whole subcommand: the three conditions of `--speedup` are
            // the same search with pruning off, single ProbCut and
            // Multi-ProbCut, which is a difference of orders of magnitude by
            // construction — that is what the subcommand measures — so they
            // are split and named after the condition instead of being
            // weighted against a cost model nobody has before the run. What a
            // stage counts is in `ObservedEval`.
            let bench_one = |root: &GameState,
                             depth: u32,
                             pc: ProbCutConfig,
                             mc: MultiProbCutConfig,
                             stage: Stage|
             -> (i32, u64, f64) {
                match &glem_eval {
                    Some(g) => {
                        let observed = ObservedEval::new(g, stage);
                        let out = bench_one_with(root, depth, &observed, pc, mc);
                        observed.into_stage().close();
                        out
                    }
                    None => {
                        let disc_diff = DiscDiffEval;
                        let observed = ObservedEval::new(&disc_diff, stage);
                        let out = bench_one_with(root, depth, &observed, pc, mc);
                        observed.into_stage().close();
                        out
                    }
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
                    run.unbounded_stage("full"),
                );
                record::log_bench_condition(&mut run, "full", n_off, v_off)?;
                println!("bench-search depth={depth} from_plies={from_plies} seed={seed}");
                println!(
                    "  full  : value={v_off} nodes={n_off} time={t_off:.3}s nps={}",
                    nps(n_off, t_off)
                );

                // Single ProbCut.
                let (n_single, single_done) = if probcut_params.is_some() {
                    let pc = build_probcut(true, false, probcut_t, probcut_params.as_ref())?;
                    let (v, n, t) = bench_one(
                        &root,
                        depth,
                        pc,
                        MultiProbCutConfig::default(),
                        run.unbounded_stage("single"),
                    );
                    record::log_bench_condition(&mut run, "single", n, v)?;
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
                    let (v, n, t) = bench_one(
                        &root,
                        depth,
                        ProbCutConfig::default(),
                        mc,
                        run.unbounded_stage("mpc"),
                    );
                    record::log_bench_condition(&mut run, "mpc", n, v)?;
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
                let (value, nodes, secs) =
                    bench_one(&root, depth, pc, mc.clone(), run.unbounded_stage("search"));
                let pc_kind = if pc.enabled { "on" } else { "off" };
                let mpc_kind = if mc.enabled { "on" } else { "off" };
                let eval_kind = match &glem_model {
                    Some(p) => format!("glem({})", p.display()),
                    None => "discdiff".to_string(),
                };
                // One measurement, so it is the run's own number. Time and
                // NPS are not recorded: `status.json`'s `duration_sec` is
                // the record of time, and a node count is a different
                // quantity that cannot be derived from it.
                run.log_metrics(
                    "run",
                    &[("nodes", nodes as f64), ("search_value", f64::from(value))],
                )
                .context("run スコープの指標の記録に失敗")?;
                println!(
                    "bench-search depth={depth} from_plies={from_plies} \
                     eval={eval_kind} probcut={pc_kind} mpc={mpc_kind} \
                     value={value} nodes={nodes} time={secs:.3}s nps={}",
                    nps(nodes, secs)
                );
            }
            let dir = run.finish().context("runvault: run の終了に失敗")?;
            println!("results: {}", dir.display());
        }
        Command::Selfplay => {
            // No stage: there is no work to report on yet.
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
            // No run of its own (the `.pex1` is an input to a later one), so
            // the lines go to standard error only. The shape of the stage
            // comes from the source, because the two sources bound the loop
            // differently: `--source selfplay` plays exactly `--games` games,
            // while `--source wthor` also stops when the `.wtb` files run out
            // — which is what the documented `--games 1000000` does — so
            // `--games` there is a ceiling and not a total.
            let progress = Progress::to_stderr();
            let records = match source.as_str() {
                "selfplay" => {
                    let mut stage = progress.stage("games", games);
                    let r =
                        extract::extract_selfplay_observed(games, seed, max_empties_skip, |_| {
                            stage.tick()
                        })?;
                    stage.close();
                    r
                }
                "wthor" => {
                    let dir = wthor_dir.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("--wthor-dir is required for --source wthor")
                    })?;
                    let mut stage = progress.unbounded_stage("games");
                    let r = extract::extract_wthor_observed(
                        dir,
                        Some(games),
                        max_empties_skip,
                        |_| stage.tick(),
                    )?;
                    stage.close();
                    r
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
            // Same two shapes, and for the same reason, as `extract`.
            let progress = Progress::to_stderr();
            let records = match source.as_str() {
                "selfplay" => {
                    let mut stage = progress.stage("games", games);
                    let r = glem_extract::extract_selfplay_observed(
                        &spec,
                        games,
                        seed,
                        max_empties_skip,
                        |_| stage.tick(),
                    )?;
                    stage.close();
                    r
                }
                "wthor" => {
                    let dir = wthor_dir.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("--wthor-dir is required for --source wthor")
                    })?;
                    let mut stage = progress.unbounded_stage("games");
                    let r = glem_extract::extract_wthor_observed(
                        &spec,
                        dir,
                        Some(games),
                        max_empties_skip,
                        |_| stage.tick(),
                    )?;
                    stage.close();
                    r
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

            // Two phases, counted separately, because the corpus is
            // collected by replaying games (0.1s for a thousand of them) and
            // the fit searches every collected position twice at the pair's
            // depths — orders of magnitude apart, and a single count would
            // extrapolate the first phase's rate onto the second. The corpus
            // stage is unbounded: both sources stop as soon as `--samples`
            // positions are in hand, so `--samples` games is a ceiling. The
            // fit stage is bounded by the corpus that came out, which is why
            // it is opened at the first position and not before.
            let progress = Progress::to_stderr();
            let mut corpus = progress.unbounded_stage("corpus");
            let mut fit: Option<Stage> = None;
            let mut on_position = |_i: usize, total: usize| {
                fit.get_or_insert_with(|| progress.stage("fit", total))
                    .tick();
            };

            match &mpc_cascade {
                // ---- Phase 6: Multi-ProbCut cascade fit ------------------
                Some(spec) => {
                    let cascade = probcut_fit::parse_mpc_cascade(spec)?;
                    let res = probcut_fit::run_mpc_fit_observed(
                        &cascade,
                        &src,
                        samples,
                        eval_weights.as_deref(),
                        &output,
                        |_| corpus.tick(),
                        &mut on_position,
                    )?;
                    corpus.close();
                    if let Some(stage) = fit.take() {
                        stage.close();
                    }
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
                    let res = probcut_fit::run_fit_observed(
                        d,
                        h,
                        &src,
                        samples,
                        eval_weights.as_deref(),
                        probcut_t,
                        &output,
                        |_| corpus.tick(),
                        &mut on_position,
                    )?;
                    corpus.close();
                    if let Some(stage) = fit.take() {
                        stage.close();
                    }
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
            // No stage: this reads one WThor year and keeps the six
            // Logistello-vs-Murakami games out of it (0.1s measured, and the
            // whole published corpus would be seconds). There is no flag that
            // scales it — the six games of the 1997 match are the fixture.
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
            let mpc_enabled = mpc.enabled;
            let cfg = phase9_engine_config(depth, endgame_empties, mpc);
            let pattern = load_pattern(eval_weights.as_ref())?;
            let book_loaded = match &book {
                Some(p) => Some(
                    OpeningBook::load(p)
                        .map_err(|e| anyhow::anyhow!("load book {}: {e}", p.display()))?,
                ),
                None => None,
            };
            // The replay happens before there is a run to mirror into, so
            // the lines go to standard error only. The unit is one scored
            // Logistello decision — `pick` and `score`, two full searches —
            // and not the game: the gold set is six games and the whole of it
            // is 6m05s at `--depth 8` (measured), so a per-game count would
            // move five times in six minutes. The number of decisions is not
            // known first, because a game stops at its first unparsable or
            // illegal recorded move, so the stage is unbounded.
            let mut decisions = Progress::to_stderr().unbounded_stage("decisions");

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
                all.extend(match_replay::replay_game_observed(
                    g,
                    &mut eng,
                    main_lo,
                    main_hi,
                    |_| decisions.tick(),
                )?);
            }
            decisions.close();
            let summary = match_replay::summarize(&all);

            // No RNG is drawn anywhere here: the recorded games are replayed
            // move by move and the engine is deterministic. Hence `analysis`
            // — `simulation` would demand a master seed nothing consumes.
            let parameters = json!({
                "depth": depth,
                "endgame_empties": endgame_empties,
                "main_lo": main_lo,
                "main_hi": main_hi,
                "eval": if eval_weights.is_some() { "pattern" } else { "basic" },
                "book": book_loaded.is_some(),
                "mpc": { "enabled": mpc_enabled },
            });
            let mut data = vec![record::murakami_dataset(&games, set.games.len())?];
            if let Some(path) = &eval_weights {
                data.push(record::learned_artifact("eval-weights", path)?);
            }
            if let Some(path) = &book {
                data.push(record::learned_artifact("opening-book", path)?);
            }
            if let Some(path) = &mpc_params {
                data.push(record::learned_artifact("mpc-params", path)?);
            }
            let mut run = Run::start(
                record::options(
                    "match-replay",
                    record::DOMAIN_ANALYSIS,
                    &results_root,
                    record::replication_match_replay(),
                    scratch,
                )
                .parameters(&parameters)
                .context("runvault: parameters の組み立てに失敗")?
                .data(data),
            )
            .context("runvault: run の開始に失敗")?;
            record::log_decisions(&mut run, &all)?;
            run.log_metrics(
                "run",
                &[
                    ("n_units", set.games.len() as f64),
                    ("scored_decisions", f64::from(summary.total)),
                    ("matched", f64::from(summary.matched)),
                    ("overall_match_rate", summary.overall_rate()),
                    ("main_decisions", f64::from(summary.main_total)),
                    ("main_matched", f64::from(summary.main_matched)),
                    ("main_match_rate", summary.main_rate()),
                ],
            )
            .context("run スコープの指標の記録に失敗")?;
            let dir = run.finish().context("runvault: run の終了に失敗")?;

            // The run holds every decision; `--output` is an extra copy for
            // whoever wants the flat table, and it is written outside the run
            // so `manifest.csv` stays the one `finish()` sealed.
            if let Some(path) = &output {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent).ok();
                }
                match_replay::write_csv(path, &all)?;
            }
            let eval_kind = eval_weights.as_ref().map_or_else(
                || "basic".to_string(),
                |p| format!("pattern({})", p.display()),
            );
            println!(
                "match-replay games={} engine={} depth={depth} \
                 endgame_empties={endgame_empties} main_window=[{main_lo},{main_hi}] \
                 results={}",
                set.games.len(),
                eval_kind,
                dir.display()
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
            // The games run before there is a run to mirror into, so the
            // lines go to standard error only. One stage per Edax level, and
            // never one over all of them: an Edax level is its search depth,
            // and one game measures 1.9s at level 5 / `--depth 8` against
            // 44s at level 15 and 117s at level 20 / `--depth 12` — sixty
            // times — so a single count would read the cheap level's rate
            // onto the expensive one, which is the estimate this API exists
            // not to make. Inside a level the unit is the **ply** and not the
            // game, because at the documented sweep (levels 5,10,15,20 ×
            // `--num-games-per-level 30`, `--depth 12`) a game is two minutes
            // and a per-game count would sit still that long. How many plies
            // a game takes is not known before it is played, so the stage is
            // unbounded.
            let progress = Progress::to_stderr();
            for &lvl in &levels {
                let ecfg =
                    EdaxConfig::new(&path, lvl).with_timeout(std::time::Duration::from_secs(120));
                let mut sess = EdaxGtpSession::start(&ecfg)
                    .map_err(|e| anyhow::anyhow!("start Edax level {lvl}: {e}"))?;
                let mut plies = progress.unbounded_stage(&format!("level {lvl}"));
                let mut lr = LevelResult::new(lvl);
                for game_idx in 0..n {
                    // Alternate: even = our engine Black, odd = our White.
                    let our_color = if game_idx % 2 == 0 {
                        Color::Black
                    } else {
                        Color::White
                    };
                    let mut eng = EngineReplay::build(our_color, &cfg, pattern.as_ref(), None);
                    let (bd, wd) =
                        play_one_edax_game(&mut sess, &mut eng, our_color, &mut || plies.tick())?;
                    let (our_d, edax_d) = if our_color == Color::Black {
                        (bd, wd)
                    } else {
                        (wd, bd)
                    };
                    lr.record(Outcome::from_discs(our_d, edax_d));
                }
                plies.close();
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

            // `--seed` is accepted and recorded but decides nothing here: our
            // engine is deterministic and Edax runs single-threaded with its
            // book off. It is kept out of `config_hash` so two runs of the
            // same condition stay one condition, and the run is `analysis`
            // rather than `simulation` for the same reason — no master seed.
            let parameters = json!({
                "edax_levels": levels,
                "num_games_per_level": n,
                "depth": depth,
                "endgame_empties": endgame_empties,
                "eval": if eval_weights.is_some() { "pattern" } else { "basic" },
                "seed": seed,
            });
            let mut data = record::edax_datasets(&path, &probe.eval_file)?;
            if let Some(p) = &eval_weights {
                data.push(record::learned_artifact("eval-weights", p)?);
            }
            let mut run = Run::start(
                record::options(
                    "elo-vs-edax",
                    record::DOMAIN_ANALYSIS,
                    &results_root,
                    record::replication_elo_vs_edax(),
                    scratch,
                )
                .parameters(&parameters)
                .context("runvault: parameters の組み立てに失敗")?
                .hash_exclude(["/seed"])
                .data(data),
            )
            .context("runvault: run の開始に失敗")?;
            record::log_edax_levels(&mut run, &rows)?;
            run.log_metrics("run", &[("n_units", rows.len() as f64)])
                .context("run スコープの指標の記録に失敗")?;
            let dir = run.finish().context("runvault: run の終了に失敗")?;

            if let Some(out) = &output {
                if let Some(parent) = out.parent() {
                    std::fs::create_dir_all(parent).ok();
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
                std::fs::write(out, &csv)?;
            }
            let eval_kind = eval_weights.as_ref().map_or_else(
                || "basic".to_string(),
                |p| format!("pattern({})", p.display()),
            );
            println!(
                "elo-vs-edax edax={} levels={:?} games_per_level={n} \
                 engine={} depth={depth} seed={seed} wall={wall:.1}s results={} \
                 (BOUNDED demo — Edax level→absolute strength is qualitative; \
                 full sweep is a documented README command)",
                path.display(),
                levels,
                eval_kind,
                dir.display()
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
            // The sampling loop runs before there is a run to mirror into,
            // so the lines go to standard error only. `--positions` is a real
            // total — the loop is `0..positions` and nothing stops it early —
            // so the stage is bounded and its estimate is worth reading. Each
            // position is one Edax self-play to terminal at `--edax-level`,
            // which is where the time is and what `--edax-level` moves:
            // 0.08s per position at level 5 and 1.7s at level 15 (measured),
            // so a minute is 36 positions at level 15. The two skip paths are
            // positions tried, so they tick as well — through the labelled
            // block, which is what keeps a `continue` from bypassing the
            // count — and the stage still closes at exactly 100%.
            let mut sampled = Progress::to_stderr().stage("positions", positions);
            for k in 0..positions {
                'position: {
                    let plies = 8 + (k % 24);
                    let (st, line) = walk_line(plies, seed.wrapping_add(k as u64));
                    if st.is_terminal() || st.must_pass() {
                        break 'position;
                    }
                    let Ok((bd, wd)) = edax_selfplay_from_line(&mut sess, &line) else {
                        break 'position;
                    };
                    let edax_diff = match st.side_to_move {
                        Color::Black => bd as i32 - wd as i32,
                        Color::White => wd as i32 - bd as i32,
                    };
                    // The sample's index is how many were kept before it —
                    // the skipped positions above never had one.
                    let idx = samples.len();
                    samples.push(CorrSample {
                        idx,
                        our_score: leaf_score(&st),
                        edax_score: edax_diff,
                    });
                }
                sampled.tick();
            }
            sampled.close();
            let xs: Vec<f64> = samples.iter().map(|s| s.our_score as f64).collect();
            let ys: Vec<f64> = samples.iter().map(|s| s.edax_score as f64).collect();
            let r = eval_corr::pearson(&xs, &ys);

            // The sampled positions come from seeded ChaCha20 walks, so the
            // seed decides which positions are measured: `simulation`.
            let parameters = json!({
                "edax_level": edax_level,
                "positions": positions,
                "seed": seed,
                "eval": if eval_weights.is_some() { "pattern" } else { "basic" },
            });
            let mut data = record::edax_datasets(&path, &probe.eval_file)?;
            if let Some(p) = &eval_weights {
                data.push(record::learned_artifact("eval-weights", p)?);
            }
            let mut run = Run::start(
                record::options(
                    "eval-correlation-edax",
                    record::DOMAIN_SIMULATION,
                    &results_root,
                    record::replication_eval_correlation(),
                    scratch,
                )
                .parameters(&parameters)
                .context("runvault: parameters の組み立てに失敗")?
                .seed_pointers(["/seed"])
                .master_seed(seed)
                .data(data),
            )
            .context("runvault: run の開始に失敗")?;
            record::log_eval_samples(&mut run, &samples)?;
            let mut values = vec![("n_units", samples.len() as f64)];
            // Pearson's r is undefined for n < 2 or a flat series. An
            // undefined value is not zero, so the row is simply not written.
            if let Some(r) = r {
                values.push(("pearson_r", r));
            }
            run.log_metrics("run", &values)
                .context("run スコープの指標の記録に失敗")?;
            let dir = run.finish().context("runvault: run の終了に失敗")?;

            if let Some(out) = &output {
                if let Some(parent) = out.parent() {
                    std::fs::create_dir_all(parent).ok();
                }
                eval_corr::write_csv(out, &samples, r)?;
            }
            let eval_kind = eval_weights.as_ref().map_or_else(
                || "basic".to_string(),
                |p| format!("pattern({})", p.display()),
            );
            let edax_disp = path.display();
            let dir_disp = dir.display();
            let r_disp = r.map_or_else(|| "NA".to_string(), |v| format!("{v:.4}"));
            let n = samples.len();
            println!(
                "eval-correlation-edax edax={edax_disp} level={edax_level} \
                 engine={eval_kind} n={n} pearson_r={r_disp} results={dir_disp} \
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
            // `learn-book` records nothing of its own — the book is an input
            // to a later run — so the lines go to standard error only.
            //
            // Which unit is ticked comes from `--depth`, the way
            // `hegselmann2005` took its denominator from the mean operator. A
            // self-play game is 0.05s at `--depth 6` and 5.3s at `--depth 10`,
            // where `0..num_games` is a real total with no early exit and the
            // share of `--num-games` done is the thing worth knowing; but the
            // same game is over four minutes at `--depth 14` (measured), and a
            // count that moves once every few minutes is the silence this
            // reporting exists to end. Past `GAME_TICK_MAX_DEPTH` the unit
            // drops to the move — one search — and the stage becomes
            // unbounded, because how many moves a game takes is not known
            // before it is played.
            const GAME_TICK_MAX_DEPTH: u32 = 12;
            let progress = Progress::to_stderr();
            let mut by_game =
                (depth <= GAME_TICK_MAX_DEPTH).then(|| progress.stage("games", num_games as usize));
            let mut by_move =
                (depth > GAME_TICK_MAX_DEPTH).then(|| progress.unbounded_stage("moves"));
            let mut on_game = |_| {
                if let Some(stage) = by_game.as_mut() {
                    stage.tick();
                }
            };
            let mut on_move = || {
                if let Some(stage) = by_move.as_mut() {
                    stage.tick();
                }
            };
            let book = match &eval_weights {
                Some(path) => {
                    let w = EvalWeights::load(path)
                        .map_err(|e| anyhow::anyhow!("load {}: {e}", path.display()))?;
                    learn_book_observed(
                        &logistello_eval::PatternEval::new(w),
                        &cfg,
                        &mut on_game,
                        &mut on_move,
                    )
                }
                None => learn_book_observed(
                    &logistello_eval::BasicEval::default(),
                    &cfg,
                    &mut on_game,
                    &mut on_move,
                ),
            };
            if let Some(stage) = by_game.take() {
                stage.close();
            }
            if let Some(stage) = by_move.take() {
                stage.close();
            }
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
            let (parent_dir, rows) = sweep::run_sweep(&resolved, &results_root, scratch)?;
            println!(
                "sweep parent: {} ({} conditions, {} trials)",
                parent_dir.display(),
                resolved.points.len(),
                rows.len(),
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
            // One recursive call that says nothing until it returns: depth 11
            // is 9.3s, depth 12 is over a minute and each further ply is
            // roughly eight times the last. Splitting the top `SPLIT` plies
            // gives a countable unit — one frontier subtree — and leaves the
            // count untouched, and the number of those subtrees is
            // `perft(SPLIT)`, which is 244 from the standard opening and
            // costs nothing to work out first. So the stage is bounded and
            // closes at exactly 100%.
            const SPLIT: u32 = 4;
            let split = depth.min(SPLIT);
            let mut subtrees =
                Progress::to_stderr().stage("subtrees", perft_standard(split) as usize);
            let nodes = perft_observed(&GameState::standard_8x8(), depth, split, &mut || {
                subtrees.tick();
            });
            subtrees.close();
            println!("perft(depth={depth}) = {nodes}");
        }
    }

    Ok(())
}
