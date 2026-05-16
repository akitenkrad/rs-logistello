//! # logistello — unified CLI for the Logistello reproduction.
//!
//! Only `perft` is wired end-to-end so far; every other subcommand is a
//! Phase-tagged placeholder that prints a "not yet implemented" message and
//! returns success.

use std::time::Instant;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};

use logistello_core::perft::perft_standard;
use logistello_eval::DiscDiffEval;
use logistello_search::iterative::search;
use logistello_search::{EngineConfig, LogistelloPlayer};
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
    },
    /// Benchmark the Phase 2 search from the standard opening using the
    /// trivial disc-difference evaluator (one working end-to-end path).
    BenchSearch {
        /// Maximum iterative-deepening depth (plies).
        #[arg(long, default_value_t = 8)]
        depth: u32,
    },
    /// Run self-play games for data generation (Phase 4).
    Selfplay,
    /// Fit ProbCut (a, b, sigma) parameters (Phase 5).
    ProbcutFit,
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
/// `engine` -> [`LogistelloPlayer`] with `cfg`; `random` -> seeded
/// [`RandomPlayer`]; `greedy` -> [`GreedyPlayer`].
fn make_player(kind: &str, color: Color, cfg: EngineConfig, seed: u64) -> Result<CliPlayer> {
    match kind {
        "engine" => Ok(CliPlayer::Engine(Box::new(LogistelloPlayer::new(
            color, cfg,
        )))),
        "random" => Ok(CliPlayer::Random(Box::new(RandomPlayer::with_seed(
            color, seed,
        )))),
        "greedy" => Ok(CliPlayer::Greedy(Box::new(GreedyPlayer::with_color(color)))),
        other => bail!("unknown player kind '{other}' (want engine|random|greedy)"),
    }
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
        } => {
            let cfg = EngineConfig {
                max_depth: depth,
                endgame_empties,
                ..EngineConfig::default()
            };
            let mut black_player = make_player(&black, Color::Black, cfg, seed)?;
            let mut white_player = make_player(&white, Color::White, cfg, seed)?;

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
            println!(
                "play black={black} white={white} depth={depth} \
                 endgame_empties={endgame_empties} seed={seed}"
            );
            println!("moves: {}", moves.join(" "));
            println!(
                "final score: black={} white={} winner={winner}",
                result.black, result.white
            );
        }
        Command::BenchSearch { depth } => {
            let root = GameState::standard_8x8();
            let start = Instant::now();
            let result = search(&root, depth, &DiscDiffEval);
            let elapsed = start.elapsed();
            let best = match result.best_move {
                Move::Pass => "pass".to_string(),
                Move::Place(c) => {
                    // Algebraic: column letter (a-h) + row digit (1-8).
                    let file = (b'a' + c.col) as char;
                    let rank = c.row + 1;
                    format!("{file}{rank}")
                }
            };
            let nps = if elapsed.as_secs_f64() > 0.0 {
                (result.nodes as f64 / elapsed.as_secs_f64()) as u64
            } else {
                0
            };
            println!(
                "bench-search depth={} best={} value={} nodes={} time={:.3}s nps={}",
                result.depth,
                best,
                result.value,
                result.nodes,
                elapsed.as_secs_f64(),
                nps
            );
        }
        Command::Selfplay => {
            println!("selfplay: not yet implemented (Phase 4)");
        }
        Command::ProbcutFit => {
            println!("probcut-fit: not yet implemented (Phase 5)");
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
