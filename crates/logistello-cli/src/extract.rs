//! Position extraction for evaluation-weight training (design doc §4.4 B3,
//! B5; Phase 4b).
//!
//! For every **non-terminal** position in a corpus of complete games we emit
//! one record:
//!
//! - the **B4 canonical** `(pattern_type, canonical_index)` for each of the
//!   47 features, computed with the *exact* same pack tables / key extraction
//!   `PatternEval` uses — Rust owns the canonicalisation;
//! - the [`stage`](logistello_eval::stage) (B2) from the disc count;
//! - the label `r` = the game's **terminal disc differential** projected to
//!   that position's **side-to-move** point of view (design doc §4.4 B3:
//!   "game result propagated"). White-to-move negates the Black-relative
//!   terminal margin.
//!
//! Two corpora are supported:
//!
//! - `selfplay`: `RandomPlayer` vs `RandomPlayer` games generated with the
//!   reused `othello-engine` `GameEngine` (seeded → deterministic). This
//!   makes the whole pipeline testable with **zero external dependencies**.
//! - `wthor`: real expert games parsed from `.wtb` via
//!   `othello_io::WthorReader` (passes auto-inserted by the reader, design
//!   doc §4.5 B6).
//!
//! The output is the documented columnar binary `PEX1` format (see
//! `EXTRACT_FORMAT.md`): a fixed self-describing header followed by
//! fixed-width little-endian rows, so the Python trainer can `numpy.fromfile`
//! it directly.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use logistello_eval::pattern::{FEATURES, N_FEATURES, feature_keys};
use logistello_eval::stage::stage;
use logistello_eval::weights::{PackTables, WEIGHT_TYPE_ORDER};
use othello_core::{Color, GameState, Move};
use othello_engine::{EngineConfig as GameEngineConfig, GameEngine};
use othello_io::WthorReader;
use othello_player::RandomPlayer;

/// `PEX1` file magic (`"PEX1"` little-endian).
pub const EXTRACT_MAGIC: u32 = 0x3158_4550;
/// `PEX1` format version.
pub const EXTRACT_VERSION: u32 = 1;
/// Bytes per record: `label(i16) + stage(u8) + pad(u8) + 47 * canon(u32)`.
pub const RECORD_BYTES: usize = 2 + 1 + 1 + N_FEATURES * 4;

/// `type_slot` for a [`PatternType`](logistello_eval::pattern::PatternType)
/// in `WEIGHT_TYPE_ORDER` order (0=C9 … 8=Const). Mirrors the private
/// `weights::type_slot`.
fn type_slot(ty: logistello_eval::pattern::PatternType) -> u8 {
    WEIGHT_TYPE_ORDER
        .iter()
        .position(|&t| t == ty)
        .expect("every PatternType is in WEIGHT_TYPE_ORDER") as u8
}

/// Per-`WEIGHT_TYPE_ORDER` canonical-class counts (`[u32; 9]`), built from
/// the vendored Edax pack tables. This is the **9-entry, type-order** array
/// the `PEX1` / `LGW1` formats use — distinct from
/// `weights::EVAL_PACKED_SIZE`, which is the 13-entry *accumulate-order*
/// table (S10/S8 listed multiple times).
#[must_use]
pub fn canon_sizes(tables: &PackTables) -> [u32; 9] {
    let mut out = [0u32; 9];
    for (t, &ty) in WEIGHT_TYPE_ORDER.iter().enumerate() {
        out[t] = tables.n_canonical(ty);
    }
    out
}

/// One extracted training position: label, stage, and the per-feature
/// **player-pack canonical index** (B4) in the 47-feature accumulate order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    /// Final disc differential from this position's side-to-move POV,
    /// `∈ [-64, 64]` (design doc §4.4 B3).
    pub label: i16,
    /// Evaluation stage `0..13` (design doc §4.4 B2).
    pub stage: u8,
    /// Player-pack canonical index per feature (same mapping `PatternEval`
    /// uses); length [`N_FEATURES`].
    pub canon: Vec<u32>,
}

/// Computes the canonical (B4) record for `state` given the game's terminal
/// disc differential **from Black's point of view** (`black - white` at the
/// terminal). The label is sign-flipped to `state`'s side to move.
fn record_for(state: &GameState, tables: &PackTables, black_terminal_diff: i32) -> Record {
    let st = stage(64 - state.board.empty_count());
    let keys = feature_keys(&state.board, state.side_to_move);
    let canon: Vec<u32> = (0..N_FEATURES)
        .map(|i| {
            let ty = FEATURES[i].ty;
            tables.get(ty).player[keys[i] as usize]
        })
        .collect();
    // Project Black-relative terminal margin to the side to move (B3).
    let label = match state.side_to_move {
        Color::Black => black_terminal_diff,
        Color::White => -black_terminal_diff,
    } as i16;
    Record {
        label,
        stage: st as u8,
        canon,
    }
}

/// Black-relative final disc differential (`black - white`) for a finished
/// game state. Pure disc count; the *game-result-propagated* label of B3
/// (endgame-exact refinement is the documented optional extra).
fn black_terminal_diff(end: &GameState) -> i32 {
    end.board.count(Color::Black) as i32 - end.board.count(Color::White) as i32
}

/// Extracts every non-terminal position of `n` seeded self-play
/// `RandomPlayer` vs `RandomPlayer` games. Deterministic for a given
/// `(n, seed)`; needs **no external data**.
///
/// `max_empties_skip`: if `Some(k)`, positions with `empties > k` are
/// skipped (skip the very opening); `None` keeps all.
///
/// # Errors
///
/// Propagates engine errors.
pub fn extract_selfplay(n: usize, seed: u64, max_empties_skip: Option<u32>) -> Result<Vec<Record>> {
    extract_selfplay_observed(n, seed, max_empties_skip, |_| {})
}

/// The same records, calling `on_game` once for every self-play game played.
///
/// The loop is `0..n` with no early exit, so `n` is the real total and a caller
/// can open a **bounded** stage of it.
///
/// # Errors
///
/// Propagates engine errors.
pub fn extract_selfplay_observed(
    n: usize,
    seed: u64,
    max_empties_skip: Option<u32>,
    mut on_game: impl FnMut(usize),
) -> Result<Vec<Record>> {
    let tables = PackTables::build();
    let mut out = Vec::new();
    for g in 0..n {
        // Distinct, deterministic seeds per game and per colour.
        let bseed = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(g as u64);
        let wseed = bseed ^ 0xD1B5_4A32_D192_ED03;
        let mut engine = GameEngine::new(GameEngineConfig::standard())
            .map_err(|e| anyhow::anyhow!("engine build: {e}"))?;
        let mut black = RandomPlayer::with_seed(Color::Black, bseed);
        let mut white = RandomPlayer::with_seed(Color::White, wseed);
        engine
            .run(&mut black, &mut white)
            .map_err(|e| anyhow::anyhow!("self-play game {g}: {e}"))?;

        let snaps = engine.history().snapshots();
        let end = snaps.last().expect("history has >=1 snapshot");
        debug_assert!(end.is_terminal(), "final snapshot must be terminal");
        let diff = black_terminal_diff(end);
        for s in snaps {
            if s.is_terminal() {
                continue;
            }
            if let Some(k) = max_empties_skip
                && s.board.empty_count() > k
            {
                continue;
            }
            out.push(record_for(s, &tables, diff));
        }
        // Outside the position loop, whose `continue`s skip a *position* and
        // never a game: the count is of games played.
        on_game(g);
    }
    Ok(out)
}

/// Replays one WTHOR `GameRecord`'s move list on a fresh `GameState`,
/// returning every non-terminal position visited and the game's
/// Black-relative terminal disc differential.
///
/// The reader already auto-inserts `Move::Pass` entries (design doc §4.5
/// B6); we additionally guard with `must_pass` so a faithful B6 pass is
/// applied even if a record omits it.
fn replay_wthor_game(moves: &[Move]) -> Result<(Vec<GameState>, i32)> {
    let mut state = GameState::standard_8x8();
    let mut positions = Vec::new();
    for &mv in moves {
        if state.is_terminal() {
            break;
        }
        positions.push(state.clone());
        // B6: if the side to move has no placement but the move is a
        // placement, the record omitted a pass — apply it first.
        if matches!(mv, Move::Place(_)) && state.must_pass() {
            state
                .apply_move(Move::Pass)
                .map_err(|e| anyhow::anyhow!("forced pass: {e}"))?;
            positions.push(state.clone());
        }
        state
            .apply_move(mv)
            .map_err(|e| anyhow::anyhow!("replay move {mv:?}: {e}"))?;
    }
    // Drain any trailing forced passes so the final state is terminal.
    while !state.is_terminal() && state.must_pass() {
        positions.push(state.clone());
        state
            .apply_move(Move::Pass)
            .map_err(|e| anyhow::anyhow!("trailing pass: {e}"))?;
    }
    let diff = black_terminal_diff(&state);
    Ok((positions, diff))
}

/// Like [`extract_selfplay`] but also returns each emitted record's source
/// [`GameState`] in the **same order** as the records (record `i`
/// corresponds to state `i`). Used by the GOLD interop test to evaluate
/// `PatternEval` on exactly the positions Python predicted.
///
/// # Errors
///
/// Propagates engine errors.
pub fn extract_selfplay_with_states(
    n: usize,
    seed: u64,
    max_empties_skip: Option<u32>,
) -> Result<(Vec<Record>, Vec<GameState>)> {
    let tables = PackTables::build();
    let mut recs = Vec::new();
    let mut states = Vec::new();
    for g in 0..n {
        let bseed = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(g as u64);
        let wseed = bseed ^ 0xD1B5_4A32_D192_ED03;
        let mut engine = GameEngine::new(GameEngineConfig::standard())
            .map_err(|e| anyhow::anyhow!("engine build: {e}"))?;
        let mut black = RandomPlayer::with_seed(Color::Black, bseed);
        let mut white = RandomPlayer::with_seed(Color::White, wseed);
        engine
            .run(&mut black, &mut white)
            .map_err(|e| anyhow::anyhow!("self-play game {g}: {e}"))?;
        let snaps = engine.history().snapshots();
        let end = snaps.last().expect("history has >=1 snapshot");
        let diff = black_terminal_diff(end);
        for s in snaps {
            if s.is_terminal() {
                continue;
            }
            if let Some(k) = max_empties_skip
                && s.board.empty_count() > k
            {
                continue;
            }
            recs.push(record_for(s, &tables, diff));
            states.push(s.clone());
        }
    }
    Ok((recs, states))
}

/// Extracts every non-terminal position from the `.wtb` files in `dir`
/// (optionally also the sibling `.jou`/`.trn`, which only carry names and
/// are not needed for labels).
///
/// # Errors
///
/// Returns an error if `dir` has no `.wtb` file or a record fails to replay.
pub fn extract_wthor(
    dir: &Path,
    max_games: Option<usize>,
    max_empties_skip: Option<u32>,
) -> Result<Vec<Record>> {
    extract_wthor_observed(dir, max_games, max_empties_skip, |_| {})
}

/// The same records, calling `on_game` once for every WThor game replayed.
///
/// `max_games` is a **ceiling and not a total**: the loop also ends when the
/// `.wtb` files run out, which is the usual case for the documented
/// `--games 1000000`. A caller therefore counts games in an *unbounded* stage
/// rather than inventing a denominator the run may never reach.
///
/// # Errors
///
/// Returns an error if `dir` has no `.wtb` file or a record fails to replay.
pub fn extract_wthor_observed(
    dir: &Path,
    max_games: Option<usize>,
    max_empties_skip: Option<u32>,
    mut on_game: impl FnMut(usize),
) -> Result<Vec<Record>> {
    let tables = PackTables::build();
    let mut wtb_files: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("reading WTHOR dir {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("wtb")))
        .collect();
    wtb_files.sort();
    if wtb_files.is_empty() {
        bail!("no .wtb files found in {}", dir.display());
    }
    let mut out = Vec::new();
    let mut games_done = 0usize;
    'outer: for path in wtb_files {
        let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        let mut reader = WthorReader::new(file)
            .map_err(|e| anyhow::anyhow!("WTHOR header {}: {e}", path.display()))?;
        let records = reader
            .read_all()
            .map_err(|e| anyhow::anyhow!("WTHOR read {}: {e}", path.display()))?;
        for rec in records {
            let moves: Vec<Move> = rec.moves.iter().map(|m| m.r#move).collect();
            let (positions, diff) = replay_wthor_game(&moves)?;
            for s in &positions {
                if s.is_terminal() {
                    continue;
                }
                if let Some(k) = max_empties_skip
                    && s.board.empty_count() > k
                {
                    continue;
                }
                out.push(record_for(s, &tables, diff));
            }
            games_done += 1;
            on_game(games_done);
            if let Some(mg) = max_games
                && games_done >= mg
            {
                break 'outer;
            }
        }
    }
    Ok(out)
}

/// Writes `records` to `path` in the `PEX1` columnar binary format
/// (see `EXTRACT_FORMAT.md`).
///
/// # Errors
///
/// Propagates I/O errors.
pub fn write_pex1(path: &Path, records: &[Record]) -> Result<()> {
    let f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    // Reserve header + fixed-width rows up front (RECORD_BYTES per record).
    let mut w = BufWriter::with_capacity(64 * 1024 + records.len() * RECORD_BYTES, f);
    w.write_all(&EXTRACT_MAGIC.to_le_bytes())?;
    w.write_all(&EXTRACT_VERSION.to_le_bytes())?;
    w.write_all(&(N_FEATURES as u32).to_le_bytes())?;
    w.write_all(&9u32.to_le_bytes())?; // n_types
    // 9-entry, WEIGHT_TYPE_ORDER canonical sizes (same as LGW1 canon_sizes).
    let tables = PackTables::build();
    for &sz in &canon_sizes(&tables) {
        w.write_all(&sz.to_le_bytes())?;
    }
    // feat_type[47]: type_slot per feature (self-describing).
    for def in &FEATURES {
        w.write_all(&[type_slot(def.ty)])?;
    }
    w.write_all(&(records.len() as u64).to_le_bytes())?;
    for r in records {
        w.write_all(&r.label.to_le_bytes())?;
        w.write_all(&[r.stage, 0u8])?; // stage + pad
        debug_assert_eq!(r.canon.len(), N_FEATURES);
        for &c in &r.canon {
            w.write_all(&c.to_le_bytes())?;
        }
    }
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use logistello_eval::PatternEval;
    use logistello_eval::weights::EvalWeights;

    #[test]
    fn selfplay_is_deterministic_and_well_formed() {
        let a = extract_selfplay(20, 7, None).unwrap();
        let b = extract_selfplay(20, 7, None).unwrap();
        assert_eq!(a, b, "same (n, seed) must reproduce identical records");
        assert!(!a.is_empty());
        let sizes = canon_sizes(&PackTables::build());
        for r in &a {
            assert!(r.stage < 13);
            assert!((-64..=64).contains(&r.label));
            assert_eq!(r.canon.len(), N_FEATURES);
            for (i, &c) in r.canon.iter().enumerate() {
                let ty = FEATURES[i].ty;
                assert!(
                    c < sizes[type_slot(ty) as usize],
                    "canon index out of range for feature {i}"
                );
            }
        }
    }

    #[test]
    fn canon_indices_match_pattern_eval_lookup() {
        // The emitted canonical indices must select the SAME weight slots
        // PatternEval uses internally (B4 mapping consistency).
        let mut w = EvalWeights::zeros();
        // Distinct value per canonical slot so the lookup is discriminating.
        for s in 0..logistello_eval::N_STAGES {
            for &ty in &WEIGHT_TYPE_ORDER {
                let v = w.weights_mut(s, ty);
                for (i, x) in v.iter_mut().enumerate() {
                    *x = (i as i32 % 251) - 125;
                }
            }
        }
        let ev = PatternEval::new(w.clone());
        let recs = extract_selfplay(8, 3, None).unwrap();
        // Re-derive each state to compare PatternEval.raw_sum against the
        // sum implied by the emitted canonical indices.
        let mut engine = GameEngine::new(GameEngineConfig::standard()).unwrap();
        let bseed = 3u64.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0u64);
        let wseed = bseed ^ 0xD1B5_4A32_D192_ED03;
        let mut black = RandomPlayer::with_seed(Color::Black, bseed);
        let mut white = RandomPlayer::with_seed(Color::White, wseed);
        engine.run(&mut black, &mut white).unwrap();
        let snaps: Vec<_> = engine
            .history()
            .snapshots()
            .iter()
            .filter(|s| !s.is_terminal())
            .cloned()
            .collect();
        for (idx, s) in snaps.iter().enumerate() {
            let r = &recs[idx];
            let st = r.stage as usize;
            let mut sum = 0i64;
            for (i, &c) in r.canon.iter().enumerate() {
                let ty = FEATURES[i].ty;
                sum += i64::from(w.weights(st, ty)[c as usize]);
            }
            assert_eq!(
                sum,
                ev.raw_sum(s),
                "canon-index sum must equal PatternEval raw_sum at position {idx}"
            );
        }
    }

    #[test]
    fn label_sign_follows_side_to_move() {
        // Construct a trivial known game: take a real self-play game,
        // recompute Black-relative terminal diff, and assert each record's
        // label equals that diff signed to the snapshot's side to move.
        let mut engine = GameEngine::new(GameEngineConfig::standard()).unwrap();
        let bseed = 11u64.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0u64);
        let wseed = bseed ^ 0xD1B5_4A32_D192_ED03;
        let mut b = RandomPlayer::with_seed(Color::Black, bseed);
        let mut wp = RandomPlayer::with_seed(Color::White, wseed);
        engine.run(&mut b, &mut wp).unwrap();
        let snaps = engine.history().snapshots();
        let end = snaps.last().unwrap();
        let diff = end.board.count(Color::Black) as i32 - end.board.count(Color::White) as i32;
        let recs = extract_selfplay(1, 11, None).unwrap();
        let mut i = 0;
        for s in snaps {
            if s.is_terminal() {
                continue;
            }
            let expect = match s.side_to_move {
                Color::Black => diff,
                Color::White => -diff,
            } as i16;
            assert_eq!(recs[i].label, expect, "record {i} label sign");
            i += 1;
        }
        assert_eq!(i, recs.len());
    }

    #[test]
    fn write_pex1_roundtrips_header_and_rows() {
        let recs = extract_selfplay(3, 99, Some(50)).unwrap();
        let dir = std::env::temp_dir();
        let path = dir.join(format!("pex1_test_{}.bin", std::process::id()));
        write_pex1(&path, &recs).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            EXTRACT_MAGIC
        );
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            N_FEATURES as u32
        );
        let header = 4 + 4 + 4 + 4 + 9 * 4 + N_FEATURES + 8;
        let n = u64::from_le_bytes(bytes[header - 8..header].try_into().unwrap()) as usize;
        assert_eq!(n, recs.len());
        assert_eq!(bytes.len(), header + n * RECORD_BYTES);
    }
}
