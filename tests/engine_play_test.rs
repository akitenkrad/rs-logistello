//! Workspace integration test for the Phase 3 engine player driven by the
//! reused `othello-engine` `GameEngine` (design doc §4.3.3 / §4.5 B8).
//!
//! - `LogistelloPlayer` vs `RandomPlayer` plays a complete legal game to a
//!   valid `GameResult`.
//! - Deterministic: the same seed + same engine config reproduces the exact
//!   same game record across two runs.
//! - Engine vs engine also terminates (both deterministic).

use logistello_search::{EngineConfig, LogistelloPlayer};
use othello_core::Color;
use othello_engine::{EngineConfig as GameEngineConfig, GameEngine};
use othello_player::RandomPlayer;

/// Plays one game (Logistello black vs Random white) and returns the move
/// list (algebraic) plus the final stone counts.
fn play_engine_vs_random(seed: u64, depth: u32, endgame_empties: u32) -> (Vec<String>, u32, u32) {
    let mut engine = GameEngine::new(GameEngineConfig::standard()).expect("engine builds");
    let cfg = EngineConfig {
        max_depth: depth,
        endgame_empties,
        ..EngineConfig::default()
    };
    let mut black = LogistelloPlayer::new(Color::Black, cfg);
    let mut white = RandomPlayer::with_seed(Color::White, seed);
    let result = engine.run(&mut black, &mut white).expect("game runs");
    assert!(engine.state().is_terminal(), "game must reach a terminal");
    assert!(result.black + result.white <= 64);

    let moves: Vec<String> = engine
        .history()
        .moves()
        .iter()
        .map(|m| match m {
            othello_core::Move::Pass => "pass".to_string(),
            othello_core::Move::Place(c) => {
                let file = (b'a' + c.col) as char;
                let rank = c.row + 1;
                format!("{file}{rank}")
            }
        })
        .collect();
    (moves, result.black, result.white)
}

#[test]
fn logistello_vs_random_completes_a_legal_game() {
    let (moves, black, white) = play_engine_vs_random(42, 4, 8);
    assert!(moves.len() > 10, "a full game has many moves");
    assert!(black + white >= 4, "game produced stones");
}

#[test]
fn logistello_vs_random_is_deterministic() {
    let a = play_engine_vs_random(42, 4, 8);
    let b = play_engine_vs_random(42, 4, 8);
    assert_eq!(a, b, "same seed + config must reproduce the exact game");
}

#[test]
fn engine_vs_engine_terminates() {
    let mut engine = GameEngine::new(GameEngineConfig::standard()).expect("engine builds");
    let cfg = EngineConfig {
        max_depth: 3,
        endgame_empties: 8,
        ..EngineConfig::default()
    };
    let mut black = LogistelloPlayer::new(Color::Black, cfg);
    let cfg2 = EngineConfig {
        max_depth: 3,
        endgame_empties: 8,
        ..EngineConfig::default()
    };
    let mut white = LogistelloPlayer::new(Color::White, cfg2);
    let result = engine.run(&mut black, &mut white).expect("game runs");
    assert!(engine.state().is_terminal());
    assert!(result.black + result.white <= 64);
}
