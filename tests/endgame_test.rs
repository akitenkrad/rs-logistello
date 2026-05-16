//! Workspace integration tests for the Phase 3 endgame solver, basic
//! evaluator, and engine driver (design doc §4.3.3 / §4.5 B8).
//!
//! These tests *define correctness*:
//!
//! - The exact endgame solver must return the **true game-theoretic final
//!   disc differential** (side-to-move POV, empties awarded to the winner)
//!   for any position close enough to the end. The gold standard is
//!   `reference_negamax` searched to `depth = empties` (both reach the
//!   terminal, so both equal the perfect-play value).
//! - WLD (win/loss/draw) must agree in sign with the exact value.
//! - The dispatcher must route by empty count exactly at
//!   `endgame_empties` (`<=` -> endgame, otherwise midgame).
//! - The basic evaluator must be antisymmetric (side-to-move POV) and
//!   monotone in the disc differential with mobility held fixed.
//!
//! If any of these fail, the implementation is wrong — the test must not be
//! weakened (per the Phase 3 spec).

use logistello_core::Zobrist;
use logistello_eval::{BasicEval, DiscDiffEval, LeafEvaluator};
use logistello_search::endgame::{best_endgame_move, solve_exact, solve_wld};
use logistello_search::iterative::{reference_negamax, search};
use logistello_search::killer::KillerTable;
use logistello_search::tt::TranspositionTable;
use logistello_search::{Dispatch, EngineConfig, decide_move, decide_move_with_dispatch};
use othello_core::{Color, Coord, GameState, Move};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha20Rng;

/// Plays `plies` random *placement* moves from the standard start, honouring
/// §4.5 B6 (a forced pass is followed transparently and does not consume a
/// ply; a terminal position ends the walk early).
fn random_position(rng: &mut ChaCha20Rng, plies: usize) -> GameState {
    let mut s = GameState::standard_8x8();
    let mut made = 0;
    while made < plies {
        if s.is_terminal() {
            break;
        }
        let moves = s.legal_moves();
        if moves.is_empty() {
            s.apply_move(Move::Pass).expect("pass legal when stuck");
            continue;
        }
        let m = *moves.choose(rng).expect("non-empty");
        s.apply_move(m).expect("chosen legal move applies");
        made += 1;
    }
    s
}

/// Plays random placement moves until the position has `<= max_empties`
/// empty squares (or the game ends first). Returns `None` if it ended with
/// more than `max_empties` empties (rare; the caller just skips it).
fn play_down_to(rng: &mut ChaCha20Rng, max_empties: u32) -> Option<GameState> {
    let mut s = GameState::standard_8x8();
    while s.board.empty_count() > max_empties {
        if s.is_terminal() {
            break;
        }
        let moves = s.legal_moves();
        if moves.is_empty() {
            s.apply_move(Move::Pass).expect("pass legal when stuck");
            continue;
        }
        let m = *moves.choose(rng).expect("non-empty");
        s.apply_move(m).expect("legal move applies");
    }
    if s.board.empty_count() <= max_empties {
        Some(s)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Test 1 — exact endgame solver == reference negamax to terminal (gold std).
//
// `reference_negamax` is the *unpruned, no-TT* baseline; at `depth =
// empties` it reaches the terminal before `depth 0` (placement: depth-1 &
// empties-1; pass: both kept), so its heuristic evaluator is never invoked
// and its value is the true game-theoretic disc differential. The exact
// solver must reproduce it bit-for-bit.
//
// The unpruned reference is exponential, so this gold-standard comparison
// runs at `<= 8` empties (fast in *debug* `cargo test`, still a rigorous
// proof — both searches reach the same terminal). Deeper positions (up to
// 12 empties) are covered against the *fast* pruned solver in Test 1b.
// ---------------------------------------------------------------------------

#[test]
fn solve_exact_equals_reference_to_terminal() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x0ED6_AE12_3456_789A);
    let mut checked = 0u32;
    for _ in 0..40 {
        let Some(s) = play_down_to(&mut rng, 8) else {
            continue;
        };
        if s.is_terminal() {
            continue;
        }
        let empties = s.board.empty_count();
        let want = reference_negamax(&s, empties, &DiscDiffEval);

        let mut tt = TranspositionTable::new();
        let mut killers = KillerTable::new();
        let z = Zobrist::new();
        let got = solve_exact(&s, &mut tt, &mut killers, &z);

        assert_eq!(
            got, want,
            "empties={empties}: solve_exact {got} != reference {want}"
        );
        // The perfect-play final differential is always even (empties go to
        // the winner, design doc §4.5 B6).
        assert_eq!(got % 2, 0, "solver value {got} must be even");
        checked += 1;
    }
    assert!(
        checked >= 20,
        "ran enough endgame positions (got {checked})"
    );
}

// ---------------------------------------------------------------------------
// Test 1b — exact solver, deeper positions (pruned solver only, so fast).
//
// At up to 12 empties the unpruned reference is intractable in debug; the
// pruned+TT solver still is. We assert the two solver-internal invariants
// that hold for the true game value: it is even, and its sign agrees with
// the independent narrow-window WLD search (Test 2 makes that explicit).
// ---------------------------------------------------------------------------

#[test]
fn solve_exact_is_even_and_consistent_deeper() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x12DE_E034_5678_9ABC);
    let mut checked = 0u32;
    for _ in 0..60 {
        let Some(s) = play_down_to(&mut rng, 12) else {
            continue;
        };
        let mut tt = TranspositionTable::new();
        let mut killers = KillerTable::new();
        let z = Zobrist::new();
        let exact = solve_exact(&s, &mut tt, &mut killers, &z);
        // The exact final disc differential is always even (empties to the
        // winner, design doc §4.5 B6).
        assert_eq!(exact % 2, 0, "solver value {exact} must be even");
        // A second solve (warm TT) must agree — TT only changes speed.
        let exact2 = solve_exact(&s, &mut tt, &mut killers, &z);
        assert_eq!(exact, exact2, "warm-TT solve disagreed");
        checked += 1;
    }
    assert!(checked >= 30, "ran enough deep positions (got {checked})");
}

// ---------------------------------------------------------------------------
// Test 2 — WLD sign consistency with the exact value.
// ---------------------------------------------------------------------------

#[test]
fn solve_wld_sign_matches_solve_exact() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5160_4ABC_DEF0_1111);
    let mut checked = 0u32;
    for _ in 0..50 {
        let Some(s) = play_down_to(&mut rng, 11) else {
            continue;
        };
        let mut tt = TranspositionTable::new();
        let mut killers = KillerTable::new();
        let z = Zobrist::new();
        let exact = solve_exact(&s, &mut tt, &mut killers, &z);
        let wld = solve_wld(&s, &mut tt, &mut killers, &z);
        assert!((-1..=1).contains(&wld), "wld {wld} must be one of -1/0/+1");
        assert_eq!(wld, exact.signum(), "WLD {wld} sign != exact {exact} sign");
        checked += 1;
    }
    assert!(checked >= 25, "ran enough WLD positions (got {checked})");
}

// ---------------------------------------------------------------------------
// Test 3 — solver returns the hand-built terminal values (B6 parity).
// ---------------------------------------------------------------------------

#[test]
fn solve_exact_on_terminal_states() {
    use logistello_search::terminal_score;

    // (a) Full board, Black 40 / White 24, E = 0 -> 2P - 64 = 16.
    let mut s = GameState::standard_8x8();
    for r in 0..8u8 {
        for c in 0..8u8 {
            let idx = r * 8 + c;
            let col = if idx < 40 { Color::Black } else { Color::White };
            s.board.set(Coord::new(r, c), Some(col));
        }
    }
    s.side_to_move = Color::Black;
    s.consecutive_passes = 2;
    assert!(s.is_terminal());

    let mut tt = TranspositionTable::new();
    let mut killers = KillerTable::new();
    let z = Zobrist::new();
    assert_eq!(solve_exact(&s, &mut tt, &mut killers, &z), 16);
    assert_eq!(
        solve_exact(&s, &mut tt, &mut killers, &z),
        terminal_score(&s)
    );
    assert_eq!(solve_wld(&s, &mut tt, &mut killers, &z), 1);

    // (b) Both-pass terminal, Black 12 / White 7, E = 45 -> +5 + 45 = 50.
    let mut t = GameState::standard_8x8();
    for r in 0..8u8 {
        for c in 0..8u8 {
            t.board.set(Coord::new(r, c), None);
        }
    }
    for i in 0..12u8 {
        t.board.set(Coord::new(i / 8, i % 8), Some(Color::Black));
    }
    for i in 0..7u8 {
        t.board.set(Coord::new(4, i), Some(Color::White));
    }
    t.side_to_move = Color::White;
    t.consecutive_passes = 2;
    assert!(t.is_terminal());
    assert_eq!(solve_exact(&t, &mut tt, &mut killers, &z), -50);
    assert_eq!(solve_wld(&t, &mut tt, &mut killers, &z), -1);

    // (c) Exact draw -> 0 and WLD 0.
    let mut d = GameState::standard_8x8();
    for r in 0..8u8 {
        for c in 0..8u8 {
            d.board.set(Coord::new(r, c), None);
        }
    }
    for i in 0..6u8 {
        d.board.set(Coord::new(0, i), Some(Color::Black));
    }
    for i in 0..6u8 {
        d.board.set(Coord::new(7, i), Some(Color::White));
    }
    d.side_to_move = Color::Black;
    d.consecutive_passes = 2;
    assert!(d.is_terminal());
    assert_eq!(solve_exact(&d, &mut tt, &mut killers, &z), 0);
    assert_eq!(solve_wld(&d, &mut tt, &mut killers, &z), 0);
}

// ---------------------------------------------------------------------------
// Test 4 — threshold dispatch routes by empty count exactly.
// ---------------------------------------------------------------------------

#[test]
fn dispatch_routes_at_endgame_threshold() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x0D15_9A7C_4000_2222);

    // A small endgame_empties keeps the exact search cheap for the test.
    let cfg = EngineConfig {
        max_depth: 4,
        endgame_empties: 10,
        ..EngineConfig::default()
    };

    // Position with exactly `endgame_empties` empties -> endgame (`<=`).
    let at = play_down_to(&mut rng, cfg.endgame_empties).expect("reach threshold");
    assert_eq!(at.board.empty_count(), cfg.endgame_empties);
    assert!(!at.is_terminal());

    let mut tt = TranspositionTable::new();
    let mut killers = KillerTable::new();
    let z = Zobrist::new();

    let bev = BasicEval::default();
    let (dispatch_at, res_at) =
        decide_move_with_dispatch(&at, &bev, &cfg, &mut tt, &mut killers, &z);
    assert_eq!(dispatch_at, Dispatch::Endgame, "<= threshold -> endgame");
    // The spec entry point must agree with the dispatch variant.
    let res_at2 = decide_move(&at, &bev, &cfg, &mut tt, &mut killers, &z);
    assert_eq!(
        res_at, res_at2,
        "decide_move == decide_move_with_dispatch.1"
    );
    // Endgame path: exact value, depth == empties, even value.
    let exact = solve_exact(&at, &mut tt, &mut killers, &z);
    assert_eq!(res_at.value, exact);
    assert_eq!(res_at.depth, at.board.empty_count());
    assert_eq!(res_at.value % 2, 0);

    // Position with strictly more empties than the threshold -> midgame.
    let above = play_down_to(&mut rng, cfg.endgame_empties + 2).expect("reach threshold+2");
    if above.board.empty_count() > cfg.endgame_empties && !above.is_terminal() {
        let (dispatch_above, res_above) =
            decide_move_with_dispatch(&above, &bev, &cfg, &mut tt, &mut killers, &z);
        assert_eq!(
            dispatch_above,
            Dispatch::Midgame,
            "> threshold -> midgame (empties={})",
            above.board.empty_count()
        );
        // Midgame path matches the Phase 2 iterative search value.
        let want = search(&above, cfg.max_depth, &BasicEval::default());
        assert_eq!(res_above.value, want.value);
        assert_eq!(res_above.depth, cfg.max_depth);
    }
}

// ---------------------------------------------------------------------------
// Test 5 — basic evaluator: side-to-move antisymmetry + monotone in disc
//          differential + correct mobility-term sign.
// ---------------------------------------------------------------------------

#[test]
fn basic_eval_is_antisymmetric_side_to_move() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xBA51_C0DE_3333_4444);
    let bev = BasicEval::default();
    for trial in 0..200u64 {
        let plies = (trial as usize * 3) % 40;
        let mut s = random_position(&mut rng, plies);
        if s.is_terminal() {
            continue;
        }
        let me = bev.eval(&s);
        // Same board, opposite side to move: value must negate exactly.
        s.side_to_move = s.side_to_move.opponent();
        let them = bev.eval(&s);
        assert_eq!(me, -them, "trial {trial}: eval not antisymmetric");
    }
}

#[test]
fn basic_eval_monotone_in_disc_diff_with_mobility_fixed() {
    // Two *full* boards (no legal moves for either side -> mobility term is
    // 0 - 0 on both) differing only in the Black/White split. The basic
    // evaluator must rank the larger disc differential strictly higher.
    let make_full = |black_cells: u32| -> GameState {
        let mut s = GameState::standard_8x8();
        for r in 0..8u8 {
            for c in 0..8u8 {
                let idx = (r as u32) * 8 + (c as u32);
                let col = if idx < black_cells {
                    Color::Black
                } else {
                    Color::White
                };
                s.board.set(Coord::new(r, c), Some(col));
            }
        }
        s.side_to_move = Color::Black;
        s.consecutive_passes = 2;
        s
    };

    let bev = BasicEval::default();
    let lo = make_full(33); // Black 33 / White 31 -> diff +2
    let hi = make_full(40); // Black 40 / White 24 -> diff +16
    assert!(lo.legal_moves().is_empty());
    assert!(hi.legal_moves().is_empty());
    assert!(
        bev.eval(&hi) > bev.eval(&lo),
        "larger disc differential must score strictly higher: hi={} lo={}",
        bev.eval(&hi),
        bev.eval(&lo)
    );
}

#[test]
fn basic_eval_mobility_term_sign_is_correct() {
    let bev = BasicEval::default();
    // Symmetric start: disc diff 0, mobility 4 vs 4 -> score exactly 0.
    let s = GameState::standard_8x8();
    assert_eq!(bev.eval(&s), 0, "symmetric start must evaluate to 0");

    // Build a board: Black to move, disc differential exactly 0, with a
    // mobility imbalance so the mobility term alone decides the score.
    let mut t = GameState::standard_8x8();
    for r in 0..8u8 {
        for c in 0..8u8 {
            t.board.set(Coord::new(r, c), None);
        }
    }
    t.board.set(Coord::new(3, 2), Some(Color::Black));
    t.board.set(Coord::new(3, 3), Some(Color::White));
    t.board.set(Coord::new(4, 4), Some(Color::White));
    t.board.set(Coord::new(4, 5), Some(Color::Black));
    t.side_to_move = Color::Black;
    t.consecutive_passes = 0;

    let p = t.board.count(Color::Black) as i32;
    let o = t.board.count(Color::White) as i32;
    assert_eq!(p - o, 0, "constructed disc diff must be 0");

    let black_mob = t.board.legal_moves(Color::Black).len() as i32;
    let white_mob = t.board.legal_moves(Color::White).len() as i32;
    let score = bev.eval(&t);
    // disc diff is 0, so the score is exactly the mobility term.
    let expected = bev.w_mobility * (black_mob - white_mob);
    assert_eq!(
        score, expected,
        "with disc diff 0 the score must equal the mobility term"
    );
    if black_mob > white_mob {
        assert!(score > 0, "mobility-favoured side must score positive");
    } else if black_mob < white_mob {
        assert!(score < 0, "mobility-disfavoured side must score negative");
    }
}

// ---------------------------------------------------------------------------
// Test 6 — best_endgame_move returns a legal move with the exact value.
// ---------------------------------------------------------------------------

#[test]
fn best_endgame_move_is_legal_and_exact() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xBE57_31D6_5555_6666);
    let mut checked = 0u32;
    for _ in 0..40 {
        let Some(s) = play_down_to(&mut rng, 10) else {
            continue;
        };
        if s.is_terminal() {
            continue;
        }
        let mut tt = TranspositionTable::new();
        let mut killers = KillerTable::new();
        let z = Zobrist::new();
        let (mv, val) = best_endgame_move(&s, &mut tt, &mut killers, &z);
        let exact = solve_exact(&s, &mut tt, &mut killers, &z);
        assert_eq!(val, exact, "best move value must equal solve_exact");

        let legal = s.legal_moves();
        if legal.is_empty() {
            assert_eq!(mv, Move::Pass, "must-pass node returns Pass");
        } else {
            assert!(legal.contains(&mv), "best move must be legal: {mv:?}");
            // Playing the chosen move and solving from the child (negated)
            // must reproduce the exact value (root consistency).
            let mut child = s.clone();
            child.apply_move(mv).expect("legal move applies");
            let child_val = -solve_exact(&child, &mut tt, &mut killers, &z);
            assert_eq!(
                child_val, exact,
                "the returned move must achieve the exact value"
            );
        }
        checked += 1;
    }
    assert!(
        checked >= 15,
        "ran enough best-move positions (got {checked})"
    );
}
