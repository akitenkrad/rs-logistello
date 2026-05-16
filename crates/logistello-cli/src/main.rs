//! # logistello — unified CLI for the Logistello reproduction.
//!
//! Only `perft` is wired end-to-end so far; every other subcommand is a
//! Phase-tagged placeholder that prints a "not yet implemented" message and
//! returns success.

use std::time::Instant;

use anyhow::Result;
use clap::{Parser, Subcommand};

use logistello_core::perft::perft_standard;
use logistello_eval::DiscDiffEval;
use logistello_search::iterative::search;
use othello_core::{GameState, Move};

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
    /// Play a single game against the engine (Phase 2).
    Play,
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

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Play => {
            println!("play: not yet implemented (Phase 2)");
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
