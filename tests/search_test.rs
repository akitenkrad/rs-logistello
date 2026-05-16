//! Workspace integration tests for the Phase 2 search engine
//! (NegaScout/PVS + iterative deepening + transposition table + killers).
//!
//! These tests *define correctness*: the optimised search must return the
//! exact same minimax value as a dead-simple reference negamax that shares
//! the §4.5 B6 pass/terminal semantics and the same `LeafEvaluator`. Pruning,
//! the transposition table, and killer ordering are only allowed to change
//! *speed*, never the value. If any of these fail, the search is wrong — the
//! test must not be weakened.

use logistello_core::Zobrist;
use logistello_eval::DiscDiffEval;
use logistello_search::alphabeta::{INF, SearchConfig, SearchContext, negascout};
use logistello_search::iterative::{reference_negamax, search_with};
use logistello_search::killer::KillerTable;
use logistello_search::tt::TranspositionTable;
use othello_core::{GameState, Move};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha20Rng;

/// Plays `plies` random *placement* moves from the standard start, honouring
/// §4.5 B6: a forced pass is followed transparently (it does not consume a
/// "ply"); a terminal position ends the walk early.
fn random_position(rng: &mut ChaCha20Rng, plies: usize) -> GameState {
    let mut s = GameState::standard_8x8();
    let mut made = 0;
    while made < plies {
        if s.is_terminal() {
            break;
        }
        let moves = s.legal_moves();
        if moves.is_empty() {
            // Forced pass: same position progression, not counted as a ply.
            s.apply_move(Move::Pass).expect("pass legal when stuck");
            continue;
        }
        let m = *moves.choose(rng).expect("non-empty");
        s.apply_move(m).expect("chosen legal move applies");
        made += 1;
    }
    s
}

/// Full-window negascout value via the public API with explicit tables.
fn negascout_value(s: &GameState, depth: u32, config: SearchConfig) -> i32 {
    let mut tt = TranspositionTable::new();
    let mut killers = KillerTable::new();
    let z = Zobrist::new();
    let mut ctx = SearchContext {
        evaluator: &DiscDiffEval,
        tt: &mut tt,
        killers: &mut killers,
        zobrist: &z,
        config,
        nodes: 0,
    };
    negascout(&mut ctx, s, -INF, INF, depth, 0)
}

// ---------------------------------------------------------------------------
// Test 1 — Reference negamax property test (the gold standard).
// ---------------------------------------------------------------------------

#[test]
fn negascout_equals_reference_negamax_property() {
    // Seeded so failures are reproducible.
    let mut rng = ChaCha20Rng::seed_from_u64(0x00C0_FFEE_1234_5678);
    let mut checked = 0u32;
    for trial in 0..120u64 {
        let plies = (trial as usize * 7) % 41; // 0..=40 random plies
        let s = random_position(&mut rng, plies);
        for depth in 1..=6u32 {
            let want = reference_negamax(&s, depth, &DiscDiffEval);
            let got = negascout_value(&s, depth, SearchConfig::default());
            assert_eq!(
                got, want,
                "trial {trial} (plies={plies}) depth {depth}: \
                 negascout {got} != reference {want}"
            );
            checked += 1;
        }
    }
    assert!(checked >= 700, "ran enough (position, depth) checks");
}

// ---------------------------------------------------------------------------
// Test 2 — Transposition-table invariance (warm vs fresh TT).
// ---------------------------------------------------------------------------

#[test]
fn tt_does_not_change_value() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xABCD_0001);
    let z = Zobrist::new();
    for trial in 0..40u64 {
        let plies = (trial as usize * 5) % 35;
        let s = random_position(&mut rng, plies);
        for depth in 1..=6u32 {
            // (a) Fresh empty TT each call.
            let fresh = negascout_value(&s, depth, SearchConfig::default());

            // (b) A TT pre-warmed by a *prior, deeper* search of the same
            //     position, then reused for this depth — must agree.
            let mut tt = TranspositionTable::new();
            let mut killers = KillerTable::new();
            {
                let mut warm = SearchContext {
                    evaluator: &DiscDiffEval,
                    tt: &mut tt,
                    killers: &mut killers,
                    zobrist: &z,
                    config: SearchConfig::default(),
                    nodes: 0,
                };
                // Warm the table with a shallower then equal-depth search.
                let _ = negascout(&mut warm, &s, -INF, INF, depth.min(3), 0);
            }
            let warmed = {
                let mut warm = SearchContext {
                    evaluator: &DiscDiffEval,
                    tt: &mut tt,
                    killers: &mut killers,
                    zobrist: &z,
                    config: SearchConfig::default(),
                    nodes: 0,
                };
                negascout(&mut warm, &s, -INF, INF, depth, 0)
            };
            assert_eq!(
                fresh, warmed,
                "trial {trial} depth {depth}: warm-TT {warmed} != fresh {fresh}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 3 — PVS / killer-ordering invariance (pruning never changes value).
// ---------------------------------------------------------------------------

#[test]
fn killers_and_tt_do_not_change_value() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5EED_9999);
    let configs = [
        SearchConfig {
            use_tt: true,
            use_killers: true,
        },
        SearchConfig {
            use_tt: true,
            use_killers: false,
        },
        SearchConfig {
            use_tt: false,
            use_killers: true,
        },
        SearchConfig {
            use_tt: false,
            use_killers: false,
        },
    ];
    for trial in 0..50u64 {
        let plies = (trial as usize * 3) % 38;
        let s = random_position(&mut rng, plies);
        for depth in 1..=6u32 {
            let reference = reference_negamax(&s, depth, &DiscDiffEval);
            for cfg in configs {
                let got = negascout_value(&s, depth, cfg);
                assert_eq!(
                    got, reference,
                    "trial {trial} depth {depth} cfg {cfg:?}: \
                     {got} != reference {reference}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test 4 — Pass handling (B6 step 2: depth-preserving pass).
// ---------------------------------------------------------------------------

/// Walks random games until it finds a non-terminal state whose side to move
/// has no legal placement (must pass while the opponent can move).
fn find_must_pass_position(rng: &mut ChaCha20Rng) -> Option<GameState> {
    for _ in 0..4000 {
        let mut s = GameState::standard_8x8();
        while !s.is_terminal() {
            if s.must_pass() {
                debug_assert!(s.legal_moves().is_empty());
                return Some(s);
            }
            let moves = s.legal_moves();
            if moves.is_empty() {
                s.apply_move(Move::Pass).ok()?;
                continue;
            }
            let m = *moves.choose(rng).expect("non-empty");
            s.apply_move(m).expect("legal move applies");
        }
    }
    None
}

#[test]
fn pass_node_matches_reference() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x9A55_0001);
    let s = find_must_pass_position(&mut rng)
        .expect("a must-pass position should occur within the search budget");

    assert!(s.legal_moves().is_empty(), "side to move must pass");
    assert!(!s.is_terminal(), "opponent can still move");
    assert!(s.must_pass(), "exercises B6 step 2");

    for depth in 1..=6u32 {
        let want = reference_negamax(&s, depth, &DiscDiffEval);
        let got = negascout_value(&s, depth, SearchConfig::default());
        assert_eq!(
            got, want,
            "must-pass depth {depth}: negascout {got} != reference {want}"
        );
        // Iterative-deepening entry must agree too, and report Pass.
        let mut tt = TranspositionTable::new();
        let mut killers = KillerTable::new();
        let z = Zobrist::new();
        let r = search_with(
            &s,
            depth,
            &DiscDiffEval,
            &mut tt,
            &mut killers,
            &z,
            SearchConfig::default(),
        );
        assert_eq!(r.best_move, Move::Pass, "must-pass root reports Pass");
        assert_eq!(
            r.value, want,
            "iterative must-pass depth {depth}: {} != {want}",
            r.value
        );
    }
}

// ---------------------------------------------------------------------------
// Test 5 — Terminal score hand calculations (B6 formula).
// ---------------------------------------------------------------------------

#[test]
fn terminal_score_hand_built_states() {
    use logistello_search::alphabeta::terminal_score;
    use othello_core::{Color, Coord};

    // (a) Full board, Black 40 / White 24 (E = 0): diff = 16 => 2P-64 = 16.
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
    assert_eq!(terminal_score(&s), 16); // 40 - 24, E = 0
    assert_eq!(2 * 40 - 64, 16); // all-even property: 2P - 64
    s.side_to_move = Color::White;
    assert_eq!(terminal_score(&s), -16);

    // (b) Both-pass terminal with empties awarded to the winner.
    //     Black 12, White 7, E = 45. diff = +5 => 5 + 45 = 50.
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
    t.side_to_move = Color::Black;
    t.consecutive_passes = 2; // both-pass terminal
    assert!(t.is_terminal());
    assert_eq!(t.board.count(Color::Black), 12);
    assert_eq!(t.board.count(Color::White), 7);
    assert_eq!(t.board.empty_count(), 45);
    assert_eq!(terminal_score(&t), 50); // +5 + 45
    t.side_to_move = Color::White;
    assert_eq!(terminal_score(&t), -50); // loser POV: -5 - 45

    // (c) Exact draw (P == O): score is 0 regardless of empties.
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
    assert_eq!(terminal_score(&d), 0);

    // All solver values are even (empties-to-winner keeps parity).
    for v in [terminal_score(&s), terminal_score(&t), terminal_score(&d)] {
        assert_eq!(v % 2, 0, "solver value {v} must be even");
    }
}
