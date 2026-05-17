//! GLEM base-literal extraction (Phase 7, design doc §4.3.6 / §4.1 Phase 7).
//!
//! **Rust owns the base-literal extraction** (mirrors the Phase-4b "Rust
//! owns the B4 canonicalisation" decision; see `extract.rs`). For every
//! non-terminal position in a corpus of complete games this emits one
//! record:
//!
//! - the **sorted, deduplicated active base-literal ids** under a
//!   [`BaseFeatureSpec`], computed with the *exact* same extractor
//!   [`GlemEval`] uses at inference time;
//! - the [`stage`](logistello_eval::stage) (B2) from the disc count;
//! - the label `r` = the game's terminal disc differential projected to that
//!   position's **side-to-move** point of view (design doc §4.4 B3 — same
//!   label semantics as the Phase-4b `PEX1` extract).
//!
//! Python (`logistello_tools.glx1`) only *reads* the active-literal sets and
//! does the conjunction enumeration / support filter / linear fit / weight
//! prune — it never re-implements literal extraction. The GOLD interop test
//! proves Python's `GlemEval`-equivalent prediction equals the Rust
//! `GlemEval`'s value bit-for-bit.
//!
//! Output is the documented `GLX1` columnar binary (see `GLEM_FORMAT.md`):
//! a self-describing header (magic, version, the base-feature spec family
//! ids, n_literals) followed by **variable-length** rows (a record has a
//! per-row literal count). Both selfplay (no external data, the test
//! vehicle) and WTHOR corpora are supported, exactly like `extract.rs`.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use logistello_eval::BaseFeatureSpec;
use logistello_eval::stage::stage;
use othello_core::{Color, GameState, Move};
use othello_engine::{EngineConfig as GameEngineConfig, GameEngine};
use othello_io::WthorReader;
use othello_player::RandomPlayer;

/// `GLX1` file magic (`"GLX1"` little-endian).
pub const GLEM_EXTRACT_MAGIC: u32 = 0x3158_4C47;
/// `GLX1` format version.
pub const GLEM_EXTRACT_VERSION: u32 = 1;

/// One extracted GLEM training position: label, stage, and the sorted
/// active base-literal ids (the single source of truth shared with
/// `GlemEval`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlemRecord {
    /// Final disc differential from this position's side-to-move POV,
    /// `∈ [-64, 64]` (design doc §4.4 B3).
    pub label: i16,
    /// Evaluation stage `0..13` (design doc §4.4 B2).
    pub stage: u8,
    /// Sorted, deduplicated active global base-literal ids.
    pub literals: Vec<u32>,
}

/// Black-relative final disc differential (`black - white`) for a finished
/// game state (same faithful B3 default as `extract.rs`).
fn black_terminal_diff(end: &GameState) -> i32 {
    end.board.count(Color::Black) as i32 - end.board.count(Color::White) as i32
}

/// Builds the [`GlemRecord`] for `state` given the game's Black-relative
/// terminal disc differential. Label is sign-flipped to the side to move.
fn record_for(state: &GameState, spec: &BaseFeatureSpec, black_terminal_diff: i32) -> GlemRecord {
    let st = stage(64 - state.board.empty_count());
    let literals = spec.active_literals(state);
    let label = match state.side_to_move {
        Color::Black => black_terminal_diff,
        Color::White => -black_terminal_diff,
    } as i16;
    GlemRecord {
        label,
        stage: st as u8,
        literals,
    }
}

/// Extracts every non-terminal position of `n` seeded self-play
/// `RandomPlayer` vs `RandomPlayer` games. Deterministic for `(n, seed)`;
/// needs **no external data**. Uses the same per-game seed derivation as the
/// Phase-4b `extract::extract_selfplay` so corpora line up across pipelines.
///
/// # Errors
///
/// Propagates engine errors.
pub fn extract_selfplay(
    spec: &BaseFeatureSpec,
    n: usize,
    seed: u64,
    max_empties_skip: Option<u32>,
) -> Result<Vec<GlemRecord>> {
    let (recs, _states) = extract_selfplay_with_states(spec, n, seed, max_empties_skip)?;
    Ok(recs)
}

/// Like [`extract_selfplay`] but also returns each emitted record's source
/// [`GameState`] in the **same order**. Used by the GOLD interop test to
/// evaluate `GlemEval` on exactly the positions Python predicted.
///
/// # Errors
///
/// Propagates engine errors.
pub fn extract_selfplay_with_states(
    spec: &BaseFeatureSpec,
    n: usize,
    seed: u64,
    max_empties_skip: Option<u32>,
) -> Result<(Vec<GlemRecord>, Vec<GameState>)> {
    let mut recs = Vec::new();
    let mut states = Vec::new();
    for g in 0..n {
        // Identical seed derivation to extract::extract_selfplay.
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
            recs.push(record_for(s, spec, diff));
            states.push(s.clone());
        }
    }
    Ok((recs, states))
}

/// Replays one WTHOR move list, returning every non-terminal position and
/// the game's Black-relative terminal disc differential (same B6 forced-pass
/// discipline as `extract.rs`).
fn replay_wthor_game(moves: &[Move]) -> Result<(Vec<GameState>, i32)> {
    let mut state = GameState::standard_8x8();
    let mut positions = Vec::new();
    for &mv in moves {
        if state.is_terminal() {
            break;
        }
        positions.push(state.clone());
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
    while !state.is_terminal() && state.must_pass() {
        positions.push(state.clone());
        state
            .apply_move(Move::Pass)
            .map_err(|e| anyhow::anyhow!("trailing pass: {e}"))?;
    }
    let diff = black_terminal_diff(&state);
    Ok((positions, diff))
}

/// Extracts every non-terminal position from the `.wtb` files in `dir`.
///
/// # Errors
///
/// Returns an error if `dir` has no `.wtb` file or a record fails to replay.
pub fn extract_wthor(
    spec: &BaseFeatureSpec,
    dir: &Path,
    max_games: Option<usize>,
    max_empties_skip: Option<u32>,
) -> Result<Vec<GlemRecord>> {
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
                out.push(record_for(s, spec, diff));
            }
            games_done += 1;
            if let Some(mg) = max_games
                && games_done >= mg
            {
                break 'outer;
            }
        }
    }
    Ok(out)
}

/// Writes `records` to `path` in the `GLX1` columnar binary format
/// (see `GLEM_FORMAT.md`).
///
/// Header (all integers little-endian):
/// `magic(u32) version(u32) n_stages(u32) n_families(u32) family_ids(u8 ×F)
///  n_records(u64)`.
/// Each record: `label(i16) stage(u8) pad(u8) n_lit(u32) lit(u32 × n_lit)`.
///
/// # Errors
///
/// Propagates I/O errors.
pub fn write_glx1(path: &Path, spec: &BaseFeatureSpec, records: &[GlemRecord]) -> Result<()> {
    let f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut w = BufWriter::with_capacity(64 * 1024, f);
    w.write_all(&GLEM_EXTRACT_MAGIC.to_le_bytes())?;
    w.write_all(&GLEM_EXTRACT_VERSION.to_le_bytes())?;
    w.write_all(&(logistello_eval::stage::N_STAGES as u32).to_le_bytes())?;
    let fams = spec.families();
    w.write_all(&(fams.len() as u32).to_le_bytes())?;
    for fam in fams {
        w.write_all(&[fam.wire_id()])?;
    }
    w.write_all(&(records.len() as u64).to_le_bytes())?;
    for r in records {
        w.write_all(&r.label.to_le_bytes())?;
        w.write_all(&[r.stage, 0u8])?; // stage + pad
        w.write_all(&(r.literals.len() as u32).to_le_bytes())?;
        for &l in &r.literals {
            w.write_all(&l.to_le_bytes())?;
        }
    }
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use logistello_eval::{GlemEval, GlemModel};

    #[test]
    fn selfplay_is_deterministic_and_well_formed() {
        let spec = BaseFeatureSpec::parse("cell64,mobility,corner").unwrap();
        let a = extract_selfplay(&spec, 15, 7, None).unwrap();
        let b = extract_selfplay(&spec, 15, 7, None).unwrap();
        assert_eq!(a, b, "same (n, seed) must reproduce identical records");
        assert!(!a.is_empty());
        for r in &a {
            assert!(r.stage < 13);
            assert!((-64..=64).contains(&r.label));
            assert!(r.literals.windows(2).all(|w| w[0] < w[1]));
            for &l in &r.literals {
                assert!(l < spec.n_literals());
            }
        }
    }

    #[test]
    fn literals_match_glem_eval_active_set() {
        // The emitted active-literal set must equal the one GlemEval uses
        // internally (the cross-pipeline contract; GOLD test proves the
        // numeric prediction too).
        let spec = BaseFeatureSpec::parse("cell64,corner").unwrap();
        let (recs, states) = extract_selfplay_with_states(&spec, 6, 3, None).unwrap();
        assert_eq!(recs.len(), states.len());
        // Use a zero model so we only exercise the active-literal extractor.
        let ev = GlemEval::new(GlemModel::zeros(spec.clone()));
        for (r, s) in recs.iter().zip(&states) {
            assert_eq!(r.literals, ev.model().spec().active_literals(s));
        }
    }

    #[test]
    fn write_glx1_roundtrips_header_and_rows() {
        let spec = BaseFeatureSpec::parse("mobility,corner").unwrap();
        let recs = extract_selfplay(&spec, 4, 99, Some(50)).unwrap();
        let dir = std::env::temp_dir();
        let path = dir.join(format!("glx1_test_{}.bin", std::process::id()));
        write_glx1(&path, &spec, &recs).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            GLEM_EXTRACT_MAGIC
        );
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            GLEM_EXTRACT_VERSION
        );
        // n_families then family ids then n_records.
        let nf = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        assert_eq!(nf, 2);
        let n = u64::from_le_bytes(bytes[16 + nf..16 + nf + 8].try_into().unwrap()) as usize;
        assert_eq!(n, recs.len());
    }
}
