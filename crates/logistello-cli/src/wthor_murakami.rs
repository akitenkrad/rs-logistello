//! Murakami-1997 gold-set extraction from the raw WThor DB (design doc
//! `Logistello.md` §4.5 **B5**; Phase 9b).
//!
//! The reused `othello_io::WthorReader` parses the `.wtb` records and the
//! 16-byte header, and auto-inserts B6 passes, but it does **not** resolve
//! player ids to names (it emits `"WTHOR Player {id}"`). The real names live
//! in the sibling `WTHOR.JOU` database (a 16-byte header + 20-byte
//! NUL-padded ASCII records, the record's index = the player id used in the
//! `.wtb` `black_player_id` / `white_player_id` fields). This module owns
//! that `.jou` parsing and the "one player Logistello, the other Murakami"
//! filter, plus the algebraic encoding of the gold games.
//!
//! ## WThor move byte ↔ algebraic
//!
//! A `.wtb` move byte is `10·row + col` with `row, col ∈ 1..=8` (`11` = a1,
//! `88` = h8); `0` is end-of-game padding; passes are **not** recorded
//! (`WthorReader` re-inserts them, design doc §4.5 B6). The reused reader
//! decodes the byte to a 0-based `Coord`; [`fmt_algebraic`] re-encodes a
//! `Move` to the lowercase `f5` notation the committed JSON / `match-replay`
//! use (`pass` for a pass).
//!
//! ## Why a tiny committed JSON
//!
//! The raw FFO `.wtb`/`.jou` DB is third-party and gitignored
//! (`data/wthor/`, fetched by `scripts/fetch_wthor.sh`). The **derived**
//! 6-game Murakami set is tiny (the reproduction's gold validation target,
//! design doc §5) and *is* committed as `tests/data/murakami_1997.json` so
//! `match-replay` and the determinism tests need no network.

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, bail};
use othello_core::Move;
use othello_io::WthorReader;
use serde::{Deserialize, Serialize};

/// WThor `.jou` / `.wtb` / `.trn` shared header length (design doc §4.5 B5).
pub const WTHOR_HEADER_BYTES: usize = 16;
/// `.jou` record length: 20-byte NUL-padded ASCII player name.
pub const JOU_RECORD_BYTES: usize = 20;

/// One game of the Murakami-1997 gold set, in the committed JSON shape.
///
/// Moves are lowercase algebraic (`"f5"`, or `"pass"`), Black first, in play
/// order *with B6 passes made explicit* (so a replayer can apply them
/// verbatim). `logistello_is_black` records which colour Logistello played
/// (the other colour is Murakami); `result_black_discs` is the WThor
/// `real_score` (final Black disc count, the design-doc B5 result byte).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MurakamiGame {
    /// 0-based index of this game within `WTH_1997.wtb`.
    pub wtb_index: usize,
    /// Black player's resolved `.jou` name.
    pub black_name: String,
    /// White player's resolved `.jou` name.
    pub white_name: String,
    /// `true` iff Logistello played Black (else Logistello is White).
    pub logistello_is_black: bool,
    /// WThor `real_score`: final Black disc count `0..=64`.
    pub result_black_discs: u8,
    /// WThor `theoretical_score`: Black discs under perfect play `0..=64`.
    pub result_theoretical_black_discs: u8,
    /// Moves in play order, lowercase algebraic, B6 passes explicit.
    pub moves: Vec<String>,
}

/// The full committed gold-set document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MurakamiSet {
    /// WThor base year (1997 for the canonical set).
    pub year: u16,
    /// Source description (provenance; not a local path).
    pub source: String,
    /// The matched games (expected: 6 for 1997).
    pub games: Vec<MurakamiGame>,
}

/// Formats a `Move` as lowercase algebraic (`"f5"`) or `"pass"`.
#[must_use]
pub fn fmt_algebraic(m: Move) -> String {
    match m {
        Move::Pass => "pass".to_string(),
        Move::Place(c) => {
            let file = (b'a' + c.col) as char;
            let rank = c.row + 1;
            format!("{file}{rank}")
        }
    }
}

/// Parses a lowercase-algebraic token (`"f5"`, `"pass"`) back to a `Move`.
///
/// # Errors
///
/// Returns an error for an empty / malformed token (anything that is not
/// `pass` or `<a-h><1-8>`).
pub fn parse_algebraic(tok: &str) -> Result<Move> {
    let t = tok.trim();
    if t.eq_ignore_ascii_case("pass") {
        return Ok(Move::Pass);
    }
    let b = t.as_bytes();
    if b.len() != 2 {
        bail!("bad algebraic move {tok:?} (want <a-h><1-8> or pass)");
    }
    let file = b[0].to_ascii_lowercase();
    let rank = b[1];
    if !(b'a'..=b'h').contains(&file) || !(b'1'..=b'8').contains(&rank) {
        bail!("algebraic move {tok:?} out of range");
    }
    let col = file - b'a';
    let row = rank - b'1';
    Ok(Move::Place(othello_core::Coord::new(row, col)))
}

/// Parses a `WTHOR.JOU` byte buffer into a `id -> name` table.
///
/// Layout (design doc §4.5 B5): a 16-byte header then fixed 20-byte records,
/// each a NUL-padded ASCII (Latin-1-tolerant) player name. The record index
/// (0-based, after the header) is the id stored in the `.wtb`
/// `black_player_id` / `white_player_id` fields. Trailing partial bytes
/// (some FFO files are not an exact multiple) are ignored.
///
/// # Errors
///
/// Returns an error if the buffer is shorter than the 16-byte header.
pub fn parse_jou_names(buf: &[u8]) -> Result<Vec<String>> {
    if buf.len() < WTHOR_HEADER_BYTES {
        bail!(
            "WTHOR.JOU too short: {} bytes (< {WTHOR_HEADER_BYTES}-byte header)",
            buf.len()
        );
    }
    let body = &buf[WTHOR_HEADER_BYTES..];
    let n = body.len() / JOU_RECORD_BYTES;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let rec = &body[i * JOU_RECORD_BYTES..(i + 1) * JOU_RECORD_BYTES];
        let end = rec.iter().position(|&b| b == 0).unwrap_or(rec.len());
        // Latin-1: every byte is a valid code point — matches the FFO
        // convention (accented names) and never panics on non-UTF-8.
        let name: String = rec[..end].iter().map(|&b| b as char).collect();
        out.push(name.trim().to_string());
    }
    Ok(out)
}

/// Reads `dir/WTHOR.JOU` (any-case filename) into the id→name table.
///
/// # Errors
///
/// Returns an error if no `.jou` file is found or it cannot be parsed.
pub fn load_jou_dir(dir: &Path) -> Result<Vec<String>> {
    let mut jou: Option<std::path::PathBuf> = None;
    for e in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let p = e?.path();
        if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("jou")) {
            jou = Some(p);
            break;
        }
    }
    let jou = jou.ok_or_else(|| {
        anyhow::anyhow!(
            "no .jou (player-name DB) in {} — run scripts/fetch_wthor.sh",
            dir.display()
        )
    })?;
    let bytes = std::fs::read(&jou).with_context(|| format!("read {}", jou.display()))?;
    parse_jou_names(&bytes)
}

/// Case-insensitive substring match (the design-doc B5 filter predicate:
/// "one player name contains *Logistello*, the other *Murakami*").
fn name_contains(haystack: &str, needle: &str) -> bool {
    haystack.to_lowercase().contains(&needle.to_lowercase())
}

/// `true` iff one of `(black, white)` contains "Logistello" and the other
/// contains "Murakami" (case-insensitive) — the B5 gold-game predicate.
#[must_use]
pub fn is_logistello_vs_murakami(black: &str, white: &str) -> bool {
    let bl = name_contains(black, "logistello");
    let bm = name_contains(black, "murakami");
    let wl = name_contains(white, "logistello");
    let wm = name_contains(white, "murakami");
    (bl && wm) || (bm && wl)
}

/// Filters every `.wtb` file in `dir` for Logistello-vs-Murakami games,
/// resolving names via the sibling `.jou`, and returns them as a
/// [`MurakamiSet`] (moves in lowercase algebraic, B6 passes explicit).
///
/// The `WthorReader` already auto-inserts B6 passes into the move list and
/// exposes the raw player ids in `metadata.players.{black,white}.params`
/// (`"player_id"`); we resolve those ids through the `.jou` table.
///
/// # Errors
///
/// Returns an error if `dir` has no `.wtb` / `.jou`, or a record fails to
/// parse. Finding a number of games other than 6 is *not* an error (the
/// caller reports the honest count).
pub fn extract_murakami(dir: &Path) -> Result<MurakamiSet> {
    let names = load_jou_dir(dir)?;
    let mut wtb_files: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("reading WThor dir {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("wtb")))
        .collect();
    wtb_files.sort();
    if wtb_files.is_empty() {
        bail!(
            "no .wtb files in {} — run scripts/fetch_wthor.sh",
            dir.display()
        );
    }

    let id_name = |id: u64| -> String {
        names
            .get(id as usize)
            .cloned()
            .unwrap_or_else(|| format!("WTHOR Player {id}"))
    };

    let mut year: u16 = 0;
    let mut games = Vec::new();
    for path in &wtb_files {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mut reader = WthorReader::new(file)
            .map_err(|e| anyhow::anyhow!("WThor header {}: {e}", path.display()))?;
        year = reader.header().year;
        let records = reader
            .read_all()
            .map_err(|e| anyhow::anyhow!("WThor read {}: {e}", path.display()))?;
        for (idx, rec) in records.iter().enumerate() {
            let bid = rec.metadata.players.black.params["player_id"]
                .as_u64()
                .unwrap_or(u64::MAX);
            let wid = rec.metadata.players.white.params["player_id"]
                .as_u64()
                .unwrap_or(u64::MAX);
            let bn = id_name(bid);
            let wn = id_name(wid);
            if !is_logistello_vs_murakami(&bn, &wn) {
                continue;
            }
            let logistello_is_black = name_contains(&bn, "logistello");
            let result_black_discs = rec
                .metadata
                .result
                .as_ref()
                .map_or(0, |r| r.score.black as u8);
            // theoretical_score is not exposed by GameRecord; recover it
            // directly from the raw .wtb block below if needed. For the
            // committed set we record real (result) and re-read theoretical
            // from the raw bytes for completeness.
            let moves: Vec<String> = rec.moves.iter().map(|m| fmt_algebraic(m.r#move)).collect();
            games.push(MurakamiGame {
                wtb_index: idx,
                black_name: bn,
                white_name: wn,
                logistello_is_black,
                result_black_discs,
                result_theoretical_black_discs: theoretical_for(path, idx)
                    .unwrap_or(result_black_discs),
                moves,
            });
        }
    }
    Ok(MurakamiSet {
        year,
        source: "FFO WThor base WTH_<year>.wtb + WTHOR.JOU \
                 (ffothello.org); filter: one player Logistello, the \
                 other Murakami (design doc §4.5 B5)"
            .to_string(),
        games,
    })
}

/// Re-reads byte 7 (`theoretical_score`) of the `idx`-th 68-byte `.wtb`
/// record (after the 16-byte header). `WthorReader::GameRecord` does not
/// surface it; this is a tiny direct read so the committed JSON is complete.
fn theoretical_for(path: &Path, idx: usize) -> Option<u8> {
    let bytes = std::fs::read(path).ok()?;
    let off = WTHOR_HEADER_BYTES + idx * 68 + 7;
    bytes.get(off).copied()
}

/// Serialises a [`MurakamiSet`] to pretty JSON.
///
/// # Errors
///
/// Propagates `serde_json` errors.
pub fn to_json(set: &MurakamiSet) -> Result<String> {
    Ok(serde_json::to_string_pretty(set)?)
}

/// Loads a committed [`MurakamiSet`] JSON file.
///
/// # Errors
///
/// Propagates I/O / parse errors.
pub fn load_set(path: &Path) -> Result<MurakamiSet> {
    let s = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    Ok(serde_json::from_str(&s)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn algebraic_roundtrip_all_squares_and_pass() {
        for row in 0u8..8 {
            for col in 0u8..8 {
                let m = Move::Place(othello_core::Coord::new(row, col));
                let s = fmt_algebraic(m);
                assert_eq!(parse_algebraic(&s).unwrap(), m, "roundtrip {s}");
            }
        }
        assert_eq!(fmt_algebraic(Move::Pass), "pass");
        assert_eq!(parse_algebraic("pass").unwrap(), Move::Pass);
        assert_eq!(parse_algebraic("PASS").unwrap(), Move::Pass);
        // a1 = byte 11 = row0,col0; h8 = byte 88 = row7,col7.
        assert_eq!(
            fmt_algebraic(Move::Place(othello_core::Coord::new(0, 0))),
            "a1"
        );
        assert_eq!(
            fmt_algebraic(Move::Place(othello_core::Coord::new(7, 7))),
            "h8"
        );
        // f5 (the canonical Othello opening move) = byte 56 = row4,col5.
        assert_eq!(
            parse_algebraic("f5").unwrap(),
            Move::Place(othello_core::Coord::new(4, 5))
        );
        assert!(parse_algebraic("").is_err());
        assert!(parse_algebraic("z9").is_err());
        assert!(parse_algebraic("f55").is_err());
    }

    #[test]
    fn jou_parser_skips_header_and_decodes_records() {
        let mut buf = vec![0u8; WTHOR_HEADER_BYTES];
        // record 0
        let mut r0 = b"Murakami Takeshi".to_vec();
        r0.resize(JOU_RECORD_BYTES, 0);
        // record 1
        let mut r1 = b"Logistello (buro)".to_vec();
        r1.resize(JOU_RECORD_BYTES, 0);
        buf.extend_from_slice(&r0);
        buf.extend_from_slice(&r1);
        let names = parse_jou_names(&buf).unwrap();
        assert_eq!(names.len(), 2);
        assert_eq!(names[0], "Murakami Takeshi");
        assert_eq!(names[1], "Logistello (buro)");
        // too short -> error
        assert!(parse_jou_names(&[0u8; 4]).is_err());
    }

    #[test]
    fn predicate_matches_both_colour_orders_and_rejects_decoys() {
        assert!(is_logistello_vs_murakami(
            "Murakami Takeshi",
            "Logistello (buro)"
        ));
        assert!(is_logistello_vs_murakami(
            "logistello (buro)",
            "MURAKAMI TAKESHI"
        ));
        // Murakami Ryota is a *different* player but the substring still
        // matches "murakami" — the B5 predicate is name-substring by
        // design (the 1997 base only has Takeshi vs Logistello, verified
        // by the count assertion in the integration test).
        assert!(is_logistello_vs_murakami(
            "Logistello (buro)",
            "Murakami Ryota"
        ));
        // decoys: neither, or only one side matches.
        assert!(!is_logistello_vs_murakami("Murakami Takeshi", "Buro M."));
        assert!(!is_logistello_vs_murakami("Kasparov", "Logistello"));
        assert!(!is_logistello_vs_murakami("Foo", "Bar"));
    }

    /// Hand-built tiny `.wtb` (1 header + 3 game blocks) + matching `.jou`,
    /// driven through the *real* `WthorReader` + this module's filter:
    /// only the Logistello-vs-Murakami block must survive, with the
    /// `10·row+col` byte decode + Black-first ordering correct.
    #[test]
    fn synthetic_wtb_jou_filter_returns_only_the_gold_game() {
        let dir =
            std::env::temp_dir().join(format!("murakami_fix_{}_{}", std::process::id(), line!()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // .jou: id0=Murakami Takeshi, id1=Logistello (buro), id2=Decoy.
        let mut jou = vec![0u8; WTHOR_HEADER_BYTES];
        for nm in [
            &b"Murakami Takeshi"[..],
            &b"Logistello (buro)"[..],
            &b"Some Decoy Player"[..],
        ] {
            let mut r = nm.to_vec();
            r.resize(JOU_RECORD_BYTES, 0);
            jou.extend_from_slice(&r);
        }
        std::fs::write(dir.join("WTHOR.JOU"), &jou).unwrap();

        // .wtb header (16B): n_games=3 (LE @4), year=1997 (LE @10), size=8.
        let mut wtb = vec![0u8; WTHOR_HEADER_BYTES];
        wtb[4..8].copy_from_slice(&3u32.to_le_bytes());
        wtb[10..12].copy_from_slice(&1997u16.to_le_bytes());
        wtb[12] = 8; // board size
        wtb[13] = 0; // kind = Othello

        // A short *legal* opening line as 10·row+col bytes (1-indexed):
        // f5 d6 c3 d3  => the standard "diagonal" opening, all legal.
        // f5: row5 col6 -> 56 ; d6: row6 col4 -> 64 ;
        // c3: row3 col3 -> 33 ; d3: row3 col4 -> 34.
        let line = [56u8, 64, 33, 34];
        let mk_block = |black_id: u16, white_id: u16, real: u8, theo: u8| -> [u8; 68] {
            let mut b = [0u8; 68];
            b[0..2].copy_from_slice(&7u16.to_le_bytes()); // tournament
            b[2..4].copy_from_slice(&black_id.to_le_bytes());
            b[4..6].copy_from_slice(&white_id.to_le_bytes());
            b[6] = real;
            b[7] = theo;
            b[8..8 + line.len()].copy_from_slice(&line);
            b
        };
        // game0: decoy vs decoy (id2 vs id2)         -> excluded
        // game1: Murakami(0) black vs Logistello(1)  -> INCLUDED
        // game2: decoy(2) vs Murakami(0)             -> excluded
        wtb.extend_from_slice(&mk_block(2, 2, 32, 30));
        wtb.extend_from_slice(&mk_block(0, 1, 20, 22));
        wtb.extend_from_slice(&mk_block(2, 0, 40, 40));
        std::fs::write(dir.join("WTH_1997.wtb"), &wtb).unwrap();

        let set = extract_murakami(&dir).unwrap();
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(set.year, 1997);
        assert_eq!(
            set.games.len(),
            1,
            "exactly the one Logistello-vs-Murakami block must survive"
        );
        let g = &set.games[0];
        assert_eq!(g.wtb_index, 1);
        assert_eq!(g.black_name, "Murakami Takeshi");
        assert_eq!(g.white_name, "Logistello (buro)");
        assert!(!g.logistello_is_black, "Logistello played White here");
        assert_eq!(g.result_black_discs, 20);
        assert_eq!(g.result_theoretical_black_discs, 22);
        // 10·row+col decode: 56->f5, 64->d6, 33->c3, 34->d3.
        assert_eq!(g.moves, vec!["f5", "d6", "c3", "d3"]);
    }

    #[test]
    fn murakami_set_json_roundtrips() {
        let set = MurakamiSet {
            year: 1997,
            source: "test".to_string(),
            games: vec![MurakamiGame {
                wtb_index: 3,
                black_name: "Murakami Takeshi".to_string(),
                white_name: "Logistello (buro)".to_string(),
                logistello_is_black: false,
                result_black_discs: 16,
                result_theoretical_black_discs: 21,
                moves: vec!["f5".to_string(), "d6".to_string(), "pass".to_string()],
            }],
        };
        let js = to_json(&set).unwrap();
        let back: MurakamiSet = serde_json::from_str(&js).unwrap();
        assert_eq!(set, back);
    }
}
