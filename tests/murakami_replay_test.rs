//! Workspace integration tests for Phase 9b: Murakami-1997 replay,
//! Edax-Elo, and eval-correlation harness (design doc `Logistello.md`
//! §4.3.8, §4.5 **B5**, §5).
//!
//! All tests here are **fast** (no heavy real runs inside `cargo test`):
//! synthetic fixtures, the real engine only on a tiny capped game, and the
//! real Edax part **skips-and-passes** when the gitignored `.edax/` install
//! is absent (CI / fresh clones stay green).
//!
//! The exhaustive unit coverage of the pure logic (JOU parsing, the B5
//! Murakami filter, the `10·row+col` decode, match-rate / windowing, the
//! `Δ = -400·log10(1/p-1)` Elo formula incl. clamping, Pearson `r`,
//! deterministic CSV, the `results/` layout) lives in the modules'
//! `#[cfg(test)]` blocks and runs as part of `cargo test` too. This file
//! adds the cross-module + real-engine regression guards the spec calls
//! out, in particular the **`? wrong color`** Phase-9a regression guard.

use std::time::Duration;

use logistello_cli::edax::{EdaxConfig, EdaxGtpSession, GtpColor, locate_default_install};
use logistello_cli::elo::{LevelResult, Outcome, elo_delta};
use logistello_cli::eval_corr::pearson;
use logistello_cli::match_replay::{ReplayEngine, replay_game, summarize};
use logistello_cli::wthor_murakami::{
    JOU_RECORD_BYTES, MurakamiGame, WTHOR_HEADER_BYTES, extract_murakami,
    is_logistello_vs_murakami, parse_algebraic,
};

use anyhow::Result;
use othello_core::{Color, GameState, Move};

// --------------------------------------------------------------------------
// 1. WThor Murakami filter on a hand-built synthetic .wtb + .jou fixture.
// --------------------------------------------------------------------------

/// Build a tiny `.wtb` (3 game blocks) + matching `.jou`, run the *real*
/// `WthorReader` + this crate's filter, and assert only the
/// Logistello-vs-Murakami block survives with the `10·row+col` decode and
/// Black-first ordering correct (spec test 1; no network).
#[test]
fn synthetic_wtb_jou_filter_returns_only_gold_game() {
    let dir = std::env::temp_dir().join(format!("murakami_it_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // .jou: id0=Murakami Takeshi, id1=Logistello (buro), id2=Decoy.
    let mut jou = vec![0u8; WTHOR_HEADER_BYTES];
    for nm in [
        &b"Murakami Takeshi"[..],
        &b"Logistello (buro)"[..],
        &b"Decoy Player"[..],
    ] {
        let mut r = nm.to_vec();
        r.resize(JOU_RECORD_BYTES, 0);
        jou.extend_from_slice(&r);
    }
    std::fs::write(dir.join("WTHOR.JOU"), &jou).unwrap();

    // .wtb header: n_games=3 (@4 LE), year=1997 (@10 LE), board=8.
    let mut wtb = vec![0u8; WTHOR_HEADER_BYTES];
    wtb[4..8].copy_from_slice(&3u32.to_le_bytes());
    wtb[10..12].copy_from_slice(&1997u16.to_le_bytes());
    wtb[12] = 8;
    wtb[13] = 0;
    // Legal opening line as 10·row+col bytes: f5 d6 c3 d3 -> 56 64 33 34.
    let line = [56u8, 64, 33, 34];
    let mk = |bid: u16, wid: u16, real: u8, theo: u8| -> [u8; 68] {
        let mut b = [0u8; 68];
        b[0..2].copy_from_slice(&7u16.to_le_bytes());
        b[2..4].copy_from_slice(&bid.to_le_bytes());
        b[4..6].copy_from_slice(&wid.to_le_bytes());
        b[6] = real;
        b[7] = theo;
        b[8..8 + line.len()].copy_from_slice(&line);
        b
    };
    wtb.extend_from_slice(&mk(2, 2, 32, 30)); // decoy v decoy  -> out
    wtb.extend_from_slice(&mk(0, 1, 20, 22)); // Murakami v Logi -> IN
    wtb.extend_from_slice(&mk(2, 0, 40, 40)); // decoy v Murakami-> out
    std::fs::write(dir.join("WTH_1997.wtb"), &wtb).unwrap();

    let set = extract_murakami(&dir).unwrap();
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(set.year, 1997);
    assert_eq!(set.games.len(), 1, "only the gold block survives");
    let g = &set.games[0];
    assert_eq!(g.wtb_index, 1);
    assert_eq!(g.black_name, "Murakami Takeshi");
    assert_eq!(g.white_name, "Logistello (buro)");
    assert!(!g.logistello_is_black);
    assert_eq!(g.result_black_discs, 20);
    assert_eq!(g.result_theoretical_black_discs, 22);
    // 10·row+col decode + pass-insertion correctness (no passes here).
    assert_eq!(g.moves, vec!["f5", "d6", "c3", "d3"]);
    // The filter predicate accepts both colour orders, rejects decoys.
    assert!(is_logistello_vs_murakami("Murakami Takeshi", "Logistello"));
    assert!(!is_logistello_vs_murakami("Decoy", "Murakami"));
}

// --------------------------------------------------------------------------
// 2. match-replay logic with a deterministic stub engine.
// --------------------------------------------------------------------------

struct FirstLegal;
impl ReplayEngine for FirstLegal {
    fn pick(&mut self, s: &GameState) -> Result<Move> {
        Ok(s.legal_moves()[0])
    }
    fn score(&mut self, _s: &GameState) -> i32 {
        0
    }
}

/// A stub that plays exactly Logistello's recorded Black moves (perfect).
struct Perfect {
    moves: Vec<Move>,
    i: usize,
}
impl ReplayEngine for Perfect {
    fn pick(&mut self, _s: &GameState) -> Result<Move> {
        let m = self.moves[self.i];
        self.i += 1;
        Ok(m)
    }
    fn score(&mut self, _s: &GameState) -> i32 {
        3
    }
    fn reset(&mut self) {
        self.i = 0;
    }
}

fn gold_black(moves: &[&str]) -> MurakamiGame {
    MurakamiGame {
        wtb_index: 7,
        black_name: "Logistello (buro)".to_string(),
        white_name: "Murakami Takeshi".to_string(),
        logistello_is_black: true,
        result_black_discs: 33,
        result_theoretical_black_discs: 33,
        moves: moves.iter().map(|s| s.to_string()).collect(),
    }
}

#[test]
fn match_replay_color_detection_window_and_rate() {
    // Logistello = Black; Black plies are 0 (f5) and 2 (c3).
    let g = gold_black(&["f5", "d6", "c3", "d3"]);
    let rows = replay_game(&g, &mut FirstLegal, 0, 100).unwrap();
    assert_eq!(rows.len(), 2, "only Logistello (Black) plies are scored");
    assert_eq!(rows[0].game, 7);
    assert_eq!(rows[0].ply, 0);
    assert_eq!(rows[1].ply, 2);

    // Perfect engine -> 100% match; main window [0,1] selects only ply 0.
    let perfect = Perfect {
        moves: vec![
            parse_algebraic("f5").unwrap(),
            parse_algebraic("c3").unwrap(),
        ],
        i: 0,
    };
    let rows = replay_game(&g, &mut { perfect }, 0, 1).unwrap();
    let s = summarize(&rows);
    assert_eq!(s.total, 2);
    assert_eq!(s.matched, 2);
    assert!((s.overall_rate() - 1.0).abs() < 1e-12);
    assert_eq!(s.main_total, 1, "only ply 0 is in [0,1]");
    assert_eq!(s.main_matched, 1);
    assert!((s.main_rate() - 1.0).abs() < 1e-12);
}

#[test]
fn match_replay_never_panics_on_pass_heavy_or_terminal_lines() {
    // Composition / spec test 6: bogus + empty + explicit-pass lines.
    for moves in [
        vec!["f5", "zz", "c3"],   // unparsable mid-line -> stop
        vec![],                   // empty
        vec!["f5", "pass", "d6"], // explicit pass token
        vec!["pass", "pass"],     // all-pass
    ] {
        let g = gold_black(&moves);
        let _ = replay_game(&g, &mut FirstLegal, 0, 100).unwrap();
    }
}

// --------------------------------------------------------------------------
// 3. ELO formula incl. clamping + score aggregation (both colours).
// --------------------------------------------------------------------------

#[test]
fn elo_formula_and_aggregation() {
    assert!(elo_delta(0.5).abs() < 1e-9);
    // D=400 -> p = 1/(1+10^-1) ≈ 0.9091.
    let p = 1.0 / (1.0 + 10f64.powf(-1.0));
    assert!((elo_delta(p) - 400.0).abs() < 1e-6);
    // Clean sweeps clamp finite (not ±∞), symmetric.
    let hi = elo_delta(1.0);
    let lo = elo_delta(0.0);
    assert!(hi.is_finite() && lo.is_finite());
    assert!((hi + lo).abs() < 1e-6);

    // Colour-balanced aggregation: 2 wins as Black + 1 win + 1 loss as
    // White -> 3-0-1, score 0.75.
    let mut r = LevelResult::new(3);
    r.record(Outcome::from_discs(40, 24)); // Black win
    r.record(Outcome::from_discs(33, 31)); // Black win
    r.record(Outcome::from_discs(50, 14)); // White win
    r.record(Outcome::from_discs(20, 44)); // White loss
    assert_eq!((r.wins, r.draws, r.losses), (3, 0, 1));
    assert!((r.score_rate() - 0.75).abs() < 1e-12);
    assert!(r.elo_delta() > 0.0);
}

// --------------------------------------------------------------------------
// 4. EdaxGtpSession: short bounded real game, skip-if-absent. This is the
//    `? wrong color` Phase-9a regression guard (spec test 4).
// --------------------------------------------------------------------------

/// Tiny engine: always the first legal move (no search; keeps the real
/// game fast). Mirrors the production `EngineReplay::pick` contract.
fn first_legal(state: &GameState) -> Move {
    state.legal_moves()[0]
}

/// Drive a SHORT capped game our-first-legal-vs-Edax over the direct-GTP
/// session: assert every move legal, the game progresses with **no
/// `? wrong color` / protocol error**, and two runs are byte-identical
/// (deterministic). Skips+passes when `.edax/` is absent.
#[test]
fn edax_gtp_session_plays_bounded_game_no_wrong_color() {
    let Some(binary) = locate_default_install() else {
        eprintln!(
            "SKIP: no .edax/edax found (gitignored Edax absent). \
             Run scripts/setup_edax.sh to enable. Passing."
        );
        return;
    };
    let cfg = EdaxConfig::new(&binary, 2).with_timeout(Duration::from_secs(60));
    if !cfg.is_available() || !cfg.eval_file.is_file() {
        eprintln!("SKIP: Edax binary/weights incomplete. Passing.");
        return;
    }

    // Play `CAP` plies (our engine = Black, Edax = White), recording the
    // move sequence; assert legality + no protocol error; repeat → equal.
    const CAP: usize = 14;
    let run = || -> Result<Vec<String>> {
        let mut sess = EdaxGtpSession::start(&cfg).map_err(|e| anyhow::anyhow!("start: {e}"))?;
        sess.new_game()
            .map_err(|e| anyhow::anyhow!("new_game: {e}"))?;
        let mut state = GameState::standard_8x8();
        let mut seq = Vec::new();
        let our = Color::Black;
        let mut plies = 0;
        while !state.is_terminal() && plies < CAP {
            let stm = state.side_to_move;
            if state.must_pass() {
                if stm == our {
                    sess.play(GtpColor::Black, "pass")
                        .map_err(|e| anyhow::anyhow!("play pass: {e}"))?;
                } else {
                    let mv = sess
                        .genmove(GtpColor::White)
                        .map_err(|e| anyhow::anyhow!("genmove: {e}"))?;
                    assert_eq!(mv, "pass", "Edax must pass when forced");
                }
                state.apply_move(Move::Pass).unwrap();
                continue;
            }
            if stm == our {
                let m = first_legal(&state);
                let alg = match m {
                    Move::Place(c) => format!("{}{}", (b'a' + c.col) as char, c.row + 1),
                    Move::Pass => "pass".to_string(),
                };
                // This `play` is OUR move only — never Edax's own move.
                sess.play(GtpColor::Black, &alg)
                    .map_err(|e| anyhow::anyhow!("play {alg}: {e}"))?;
                state.apply_move(m).unwrap();
                seq.push(alg);
            } else {
                // genmove: Edax commits internally; we do NOT play it back.
                let mv = sess
                    .genmove(GtpColor::White)
                    .map_err(|e| anyhow::anyhow!("genmove: {e}"))?;
                let parsed =
                    parse_algebraic(&mv).map_err(|e| anyhow::anyhow!("parse {mv:?}: {e}"))?;
                assert!(
                    state.legal_moves().contains(&parsed),
                    "Edax returned ILLEGAL move {mv:?}"
                );
                state.apply_move(parsed).unwrap();
                seq.push(mv);
            }
            plies += 1;
        }
        Ok(seq)
    };

    let a = run().expect("bounded Edax game must complete with no ? wrong color");
    let b = run().expect("second run must also complete");
    assert!(!a.is_empty(), "the game must have progressed");
    assert_eq!(
        a, b,
        "deterministic: Edax (-level/-book-usage off/-n 1) + first-legal \
         must reproduce the exact move sequence twice"
    );
    eprintln!(
        "Edax bounded game ({} plies) OK, no `? wrong color`: {:?}",
        a.len(),
        a
    );
}

// --------------------------------------------------------------------------
// 5. Pearson sanity (cross-module): perfect (anti)correlation.
// --------------------------------------------------------------------------

#[test]
fn pearson_cross_module_sanity() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let y = [3.0, 5.0, 7.0, 9.0]; // y = 2x + 1
    assert!((pearson(&x, &y).unwrap() - 1.0).abs() < 1e-12);
    let yn = [9.0, 7.0, 5.0, 3.0];
    assert!((pearson(&x, &yn).unwrap() + 1.0).abs() < 1e-12);
}
