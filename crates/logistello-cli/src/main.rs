//! # logistello — unified CLI for the Logistello reproduction.
//!
//! Only `perft` is wired end-to-end so far; every other subcommand is a
//! Phase-tagged placeholder that prints a "not yet implemented" message and
//! returns success.

use logistello_cli::extract;
use logistello_cli::probcut_fit;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};

use logistello_core::Zobrist;
use logistello_core::perft::perft_standard;
use logistello_eval::{DiscDiffEval, EvalWeights};
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
    /// Replay a recorded match (e.g. Murakami 1997) (Phase 9).
    MatchReplay,
    /// Estimate ELO versus Edax at various levels (Phase 9).
    EloVsEdax,
    /// Learn the opening book via self-play (Phase 8).
    LearnBook,
    /// Run a parameter sweep (Phase 10).
    Sweep,
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

/// A `play`-CLI player: one concrete variant per `--black` / `--white`
/// kind. `GameEngine::run` is generic over `Sized` players, so we dispatch
/// through this enum instead of a `Box<dyn Player>` (trait objects are
/// unsized and `&mut dyn Player` does not itself implement `Player`).
enum CliPlayer {
    // Every variant is boxed so the enum stays pointer-sized regardless of
    // the (large) `LogistelloPlayer`, which owns a transposition table
    // (clippy::large_enum_variant).
    Engine(Box<LogistelloPlayer>),
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

/// Builds a [`CliPlayer`] of the requested kind for `color`.
///
/// `engine` -> [`LogistelloPlayer`] with `cfg` (using the learned
/// `PatternEval` when `learned` is supplied, else `BasicEval`);
/// `random` -> seeded [`RandomPlayer`]; `greedy` -> [`GreedyPlayer`].
fn make_player(
    kind: &str,
    color: Color,
    cfg: &EngineConfig,
    seed: u64,
    learned: Option<&logistello_eval::PatternEval>,
) -> Result<CliPlayer> {
    match kind {
        "engine" => {
            let p = match learned {
                Some(pe) => LogistelloPlayer::with_pattern(color, cfg.clone(), pe.clone()),
                None => LogistelloPlayer::new(color, cfg.clone()),
            };
            Ok(CliPlayer::Engine(Box::new(p)))
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

/// One fixed-depth NegaScout from `root` with the given ProbCut + MPC
/// config, returning `(value, nodes, elapsed_secs)`.
fn bench_one(
    root: &GameState,
    depth: u32,
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
        evaluator: &DiscDiffEval,
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
            probcut,
            no_probcut,
            probcut_t,
            probcut_params,
            mpc,
            mpc_params,
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
            let mut black_player = make_player(&black, Color::Black, &cfg, seed, learned.as_ref())?;
            let mut white_player = make_player(&white, Color::White, &cfg, seed, learned.as_ref())?;

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
            let eval_kind = match &eval_weights {
                Some(p) => format!("pattern({})", p.display()),
                None => "basic".to_string(),
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
            println!(
                "play black={black} white={white} depth={depth} \
                 endgame_empties={endgame_empties} seed={seed} eval={eval_kind} \
                 probcut={pc_kind} mpc={mpc_kind}"
            );
            println!("moves: {}", moves.join(" "));
            println!(
                "final score: black={} white={} winner={winner}",
                result.black, result.white
            );
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
        } => {
            let root = walk_position(from_plies, seed);
            let nps = |nodes: u64, secs: f64| -> u64 {
                if secs > 0.0 {
                    (nodes as f64 / secs) as u64
                } else {
                    0
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
                println!(
                    "bench-search depth={depth} from_plies={from_plies} \
                     probcut={pc_kind} mpc={mpc_kind} value={value} \
                     nodes={nodes} time={secs:.3}s nps={}",
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
        Command::MatchReplay => {
            println!("match-replay: not yet implemented (Phase 9)");
        }
        Command::EloVsEdax => {
            println!("elo-vs-edax: not yet implemented (Phase 9)");
        }
        Command::LearnBook => {
            println!("learn-book: not yet implemented (Phase 8)");
        }
        Command::Sweep => {
            println!("sweep: not yet implemented (Phase 10)");
        }
        Command::Perft { depth } => {
            let nodes = perft_standard(depth);
            println!("perft(depth={depth}) = {nodes}");
        }
    }

    Ok(())
}
