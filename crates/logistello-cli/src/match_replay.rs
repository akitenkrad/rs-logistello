//! `match-replay`: replay a recorded match and measure our engine's
//! move-match rate against it (design doc `Logistello.md` §4.3.8
//! `move_match_rate_murakami`, §5 "≥ 80% on main positions"; Phase 9b).
//!
//! For each game we replay the *recorded* moves on a fresh `GameState`.
//! Whenever **Logistello** (the colour Logistello actually played in that
//! game, per the gold JSON) is to move, we ask our engine for its move at
//! that exact position and record:
//!
//! - `game` (wtb index), `ply` (0-based count of *placement* plies played
//!   so far in the game), the recorded `logistello_move`, `our_move`,
//!   `match` (bool), our leaf score (side-to-move POV), and `is_main`
//!   (whether the ply is inside the "main" window).
//!
//! Forced B6 passes (made explicit in the gold JSON) are applied verbatim
//! and never scored (a pass is not a choice). Terminal / pass-heavy lines
//! never panic — replay simply stops at the recorded end.
//!
//! "Main positions" (design doc §5: the engine's strength is judged on the
//! consequential middle of the game, not booked openings nor the trivially
//! exact endgame): plies in `[MAIN_PLY_LO, MAIN_PLY_HI]` (default
//! `10..=50`, documented & overridable). The aggregate is reported both
//! overall and main-only.
//!
//! The engine is abstracted behind [`ReplayEngine`] so the scoring logic is
//! unit-tested with a deterministic stub (no search, no weights), and the
//! production path wraps the real [`LogistelloPlayer`] (+ optional Phase-8
//! book) — exactly the Phase-3/4 search the rest of the CLI uses.

use anyhow::{Context, Result};
use othello_core::{Color, GameState, Move};

use crate::wthor_murakami::{MurakamiGame, MurakamiSet, parse_algebraic};

/// Default inclusive "main position" ply window (design doc §5). Plies
/// before `LO` are opening/book territory; after `HI` the exact endgame
/// dominates — neither stresses the learned midgame evaluator.
pub const MAIN_PLY_LO: u32 = 10;
/// Upper bound of the default main-position window (inclusive).
pub const MAIN_PLY_HI: u32 = 50;

/// Anything that can pick a move and score a position from Logistello's
/// point of view. Implemented by the real engine wrapper and by the test
/// stub so the replay/scoring logic is verified without any search.
pub trait ReplayEngine {
    /// Logistello's move at `state` (must be one of `state.legal_moves()`).
    fn pick(&mut self, state: &GameState) -> Result<Move>;
    /// Leaf score at `state` from the side-to-move POV (disc scale).
    fn score(&mut self, state: &GameState) -> i32;
    /// Called once at the start of each game (clear TT/killers).
    fn reset(&mut self) {}
}

/// One scored decision row of the replay (one CSV record).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayRow {
    /// Source game's `.wtb` index.
    pub game: usize,
    /// 0-based placement-ply index within the game.
    pub ply: u32,
    /// The move Logistello actually played (algebraic).
    pub logistello_move: String,
    /// The move our engine chose at the same position (algebraic).
    pub our_move: String,
    /// Whether the two moves are identical.
    pub matched: bool,
    /// Our leaf score at the position (side-to-move POV, disc scale).
    pub our_score: i32,
    /// Whether `ply ∈ [lo, hi]` (the "main" window).
    pub is_main: bool,
}

/// Aggregate match-rate summary for a whole replay.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReplaySummary {
    /// Total scored Logistello decisions (passes excluded).
    pub total: u32,
    /// Matches over all scored decisions.
    pub matched: u32,
    /// Scored decisions inside the main window.
    pub main_total: u32,
    /// Matches inside the main window.
    pub main_matched: u32,
}

impl ReplaySummary {
    /// Overall match rate in `[0, 1]` (0 when no decisions were scored).
    #[must_use]
    pub fn overall_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            f64::from(self.matched) / f64::from(self.total)
        }
    }

    /// Main-window match rate in `[0, 1]` (0 when none were scored).
    #[must_use]
    pub fn main_rate(&self) -> f64 {
        if self.main_total == 0 {
            0.0
        } else {
            f64::from(self.main_matched) / f64::from(self.main_total)
        }
    }
}

/// Replays one game and produces its scored rows.
///
/// `lo..=hi` is the inclusive main-position ply window. Logistello's colour
/// is taken from the gold record (`logistello_is_black`). Recorded passes
/// are applied but never scored. If a recorded move does not parse or is
/// illegal at the reached position the game's replay stops there (the
/// remaining moves are skipped) — pass-heavy / truncated lines never panic.
///
/// # Errors
///
/// Only propagates a hard engine failure ([`ReplayEngine::pick`] erroring);
/// recorded-move issues are handled gracefully (replay just stops).
pub fn replay_game<E: ReplayEngine>(
    game: &MurakamiGame,
    engine: &mut E,
    lo: u32,
    hi: u32,
) -> Result<Vec<ReplayRow>> {
    replay_game_observed(game, engine, lo, hi, |_| {})
}

/// The same rows, calling `on_decision` once for every scored Logistello
/// decision — the point at which the engine has searched the position.
///
/// A decision is two full searches (`pick` then `score`), which at the
/// documented `--depth 12` is where the time is; the *game* is far too coarse
/// a unit for a six-game set. The number of decisions is not known before the
/// replay, because a game stops at the first unparsable or illegal recorded
/// move, so a caller counts them in an *unbounded* stage.
///
/// # Errors
///
/// Only propagates a hard engine failure ([`ReplayEngine::pick`] erroring).
pub fn replay_game_observed<E: ReplayEngine>(
    game: &MurakamiGame,
    engine: &mut E,
    lo: u32,
    hi: u32,
    mut on_decision: impl FnMut(u32),
) -> Result<Vec<ReplayRow>> {
    engine.reset();
    let logi_color = if game.logistello_is_black {
        Color::Black
    } else {
        Color::White
    };
    let mut state = GameState::standard_8x8();
    let mut rows = Vec::new();
    let mut ply: u32 = 0;
    for tok in &game.moves {
        if state.is_terminal() {
            break;
        }
        let Ok(rec_move) = parse_algebraic(tok) else {
            break;
        };
        match rec_move {
            Move::Pass => {
                // B6 pass: apply verbatim, never scored.
                if state.apply_move(Move::Pass).is_err() {
                    break;
                }
                continue;
            }
            Move::Place(_) => {
                // Only score if it is Logistello's turn AND the recorded
                // move is legal here (else the line desynced — stop).
                if !state.legal_moves().contains(&rec_move) {
                    break;
                }
                if state.side_to_move == logi_color {
                    let our = engine.pick(&state)?;
                    let sc = engine.score(&state);
                    on_decision(ply);
                    let is_main = ply >= lo && ply <= hi;
                    rows.push(ReplayRow {
                        game: game.wtb_index,
                        ply,
                        logistello_move: tok.clone(),
                        our_move: crate::wthor_murakami::fmt_algebraic(our),
                        matched: our == rec_move,
                        our_score: sc,
                        is_main,
                    });
                }
                if state.apply_move(rec_move).is_err() {
                    break;
                }
                ply += 1;
            }
        }
    }
    Ok(rows)
}

/// Replays a whole [`MurakamiSet`], returning every scored row.
///
/// # Errors
///
/// Propagates a hard engine failure from [`replay_game`].
pub fn replay_set<E: ReplayEngine>(
    set: &MurakamiSet,
    engine: &mut E,
    lo: u32,
    hi: u32,
) -> Result<Vec<ReplayRow>> {
    let mut all = Vec::new();
    for g in &set.games {
        all.extend(replay_game(g, engine, lo, hi)?);
    }
    Ok(all)
}

/// Aggregates rows into a [`ReplaySummary`].
#[must_use]
pub fn summarize(rows: &[ReplayRow]) -> ReplaySummary {
    let mut s = ReplaySummary {
        total: 0,
        matched: 0,
        main_total: 0,
        main_matched: 0,
    };
    for r in rows {
        s.total += 1;
        if r.matched {
            s.matched += 1;
        }
        if r.is_main {
            s.main_total += 1;
            if r.matched {
                s.main_matched += 1;
            }
        }
    }
    s
}

/// Writes the rows as a deterministic CSV (stable column order, no
/// timestamps) so two identical replays yield byte-identical files.
///
/// # Errors
///
/// Propagates I/O errors.
pub fn write_csv(path: &std::path::Path, rows: &[ReplayRow]) -> Result<()> {
    use std::io::Write;
    let mut buf = String::from("game,ply,logistello_move,our_move,match,our_score,is_main\n");
    for r in rows {
        buf.push_str(&format!(
            "{},{},{},{},{},{},{}\n",
            r.game, r.ply, r.logistello_move, r.our_move, r.matched, r.our_score, r.is_main
        ));
    }
    let mut f =
        std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    f.write_all(buf.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wthor_murakami::fmt_algebraic;

    /// A stub engine: always plays the first legal move, scores = 0. Pure &
    /// deterministic — exercises the scoring/windowing logic with no search.
    struct FirstLegal;
    impl ReplayEngine for FirstLegal {
        fn pick(&mut self, state: &GameState) -> Result<Move> {
            Ok(state.legal_moves()[0])
        }
        fn score(&mut self, _s: &GameState) -> i32 {
            0
        }
    }

    /// A stub that replays a fixed script of *expected* moves (so every
    /// scored decision matches) — used to verify the match counter & window.
    struct Scripted {
        moves: Vec<Move>,
        i: usize,
    }
    impl ReplayEngine for Scripted {
        fn pick(&mut self, _s: &GameState) -> Result<Move> {
            let m = self.moves[self.i];
            self.i += 1;
            Ok(m)
        }
        fn score(&mut self, _s: &GameState) -> i32 {
            7
        }
        fn reset(&mut self) {
            self.i = 0;
        }
    }

    fn game_logistello_black(moves: &[&str]) -> MurakamiGame {
        MurakamiGame {
            wtb_index: 0,
            black_name: "Logistello (buro)".to_string(),
            white_name: "Murakami Takeshi".to_string(),
            logistello_is_black: true,
            result_black_discs: 32,
            result_theoretical_black_discs: 32,
            moves: moves.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn scores_only_logistello_plies_and_detects_color() {
        // Standard diagonal opening: f5 d6 c3 d3 ...
        // Black plays plies 0,2,4...; Logistello = Black here.
        let g = game_logistello_black(&["f5", "d6", "c3", "d3"]);
        let mut e = FirstLegal;
        let rows = replay_game(&g, &mut e, 0, 100).unwrap();
        // f5 (ply0,Black=Logi) and c3 (ply2,Black=Logi) are scored;
        // d6 (ply1,White=Murakami) and d3 (ply3,White) are NOT.
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].ply, 0);
        assert_eq!(rows[0].logistello_move, "f5");
        assert_eq!(rows[1].ply, 2);
        assert_eq!(rows[1].logistello_move, "c3");
        // FirstLegal won't always match the recorded move; just assert the
        // bool is consistent with the strings.
        for r in &rows {
            assert_eq!(r.matched, r.our_move == r.logistello_move);
        }
    }

    #[test]
    fn logistello_white_is_scored_on_white_plies() {
        let mut g = game_logistello_black(&["f5", "d6", "c3", "d3"]);
        g.logistello_is_black = false;
        g.black_name = "Murakami Takeshi".to_string();
        g.white_name = "Logistello (buro)".to_string();
        let mut e = FirstLegal;
        let rows = replay_game(&g, &mut e, 0, 100).unwrap();
        // Now White plies (1: d6, 3: d3) are scored.
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].ply, 1);
        assert_eq!(rows[0].logistello_move, "d6");
        assert_eq!(rows[1].ply, 3);
        assert_eq!(rows[1].logistello_move, "d3");
    }

    #[test]
    fn perfect_engine_yields_full_match_rate_and_score_recorded() {
        let g = game_logistello_black(&["f5", "d6", "c3", "d3", "c4"]);
        // Scripted to play exactly Logistello's Black moves: f5, c3, c4.
        let scripted = Scripted {
            moves: vec![
                parse_algebraic("f5").unwrap(),
                parse_algebraic("c3").unwrap(),
                parse_algebraic("c4").unwrap(),
            ],
            i: 0,
        };
        let mut e = scripted;
        let rows = replay_game(&g, &mut e, 0, 100).unwrap();
        assert_eq!(rows.len(), 3);
        assert!(rows.iter().all(|r| r.matched), "scripted = perfect match");
        assert!(rows.iter().all(|r| r.our_score == 7));
        let s = summarize(&rows);
        assert_eq!(s.total, 3);
        assert_eq!(s.matched, 3);
        assert!((s.overall_rate() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn main_window_is_inclusive_and_partitions_correctly() {
        // Build a longer legal line by deterministically replaying.
        let mut s = GameState::standard_8x8();
        let mut toks = Vec::new();
        let mut rng = 1u64;
        while !s.is_terminal() && toks.len() < 30 {
            let lm = s.legal_moves();
            if lm.is_empty() {
                s.apply_move(Move::Pass).unwrap();
                toks.push("pass".to_string());
                continue;
            }
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let m = lm[(rng >> 33) as usize % lm.len()];
            toks.push(fmt_algebraic(m));
            s.apply_move(m).unwrap();
        }
        let tk: Vec<&str> = toks.iter().map(String::as_str).collect();
        let g = game_logistello_black(&tk);
        let mut e = FirstLegal;
        // window [4, 8] inclusive.
        let rows = replay_game(&g, &mut e, 4, 8).unwrap();
        for r in &rows {
            assert_eq!(r.is_main, r.ply >= 4 && r.ply <= 8, "ply {}", r.ply);
        }
        let sm = summarize(&rows);
        assert!(sm.main_total <= sm.total);
        let want = rows.iter().filter(|r| r.ply >= 4 && r.ply <= 8).count() as u32;
        assert_eq!(sm.main_total, want);
    }

    #[test]
    fn pass_heavy_and_truncated_lines_never_panic() {
        // A bogus / desyncing line: "zz" fails to parse -> replay stops.
        let g = game_logistello_black(&["f5", "zz", "c3"]);
        let mut e = FirstLegal;
        let rows = replay_game(&g, &mut e, 0, 100).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].logistello_move, "f5");

        // Empty game -> zero rows, no panic.
        let g2 = game_logistello_black(&[]);
        assert!(
            replay_game(&g2, &mut FirstLegal, 0, 100)
                .unwrap()
                .is_empty()
        );

        // Explicit pass token: applied or stops, never panics.
        let g3 = game_logistello_black(&["f5", "pass", "d6"]);
        let _ = replay_game(&g3, &mut FirstLegal, 0, 100).unwrap();
    }

    #[test]
    fn csv_is_deterministic() {
        let g = game_logistello_black(&["f5", "d6", "c3", "d3"]);
        let r1 = replay_game(&g, &mut FirstLegal, 0, 100).unwrap();
        let r2 = replay_game(&g, &mut FirstLegal, 0, 100).unwrap();
        assert_eq!(r1, r2);
        let dir = std::env::temp_dir();
        let p1 = dir.join(format!("mr_a_{}.csv", std::process::id()));
        let p2 = dir.join(format!("mr_b_{}.csv", std::process::id()));
        write_csv(&p1, &r1).unwrap();
        write_csv(&p2, &r2).unwrap();
        let a = std::fs::read(&p1).unwrap();
        let b = std::fs::read(&p2).unwrap();
        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);
        assert_eq!(a, b, "identical replays -> byte-identical CSV");
        let txt = String::from_utf8(a).unwrap();
        assert!(txt.starts_with("game,ply,logistello_move,our_move,match,our_score,is_main\n"));
    }
}
