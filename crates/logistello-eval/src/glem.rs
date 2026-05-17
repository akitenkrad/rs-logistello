//! GLEM — Generalized Linear Evaluation Model (Buro 1998 CG'98,
//! design doc §4.3.6, §4.1 Phase 7, §5 Objective-5).
//!
//! # What GLEM is
//!
//! GLEM builds an evaluation function from **base features** (boolean /
//! categorical literals over the board) by systematically enumerating
//! **conjunctions** (logical ANDs) of base literals up to `max_order`,
//! keeping only conjunctions whose empirical *support* (frequency in the
//! training corpus) is `>= τ_support`, fitting a per-stage linear model, and
//! pruning conjunctions whose `|weight| < τ_weight`. Per design doc §4.3.6
//! with the **B3 substitution**: the canonical Logistello-2 fit is the
//! Phase-4b linear least-squares on the *final disc differential* (GD-300 +
//! rare-config muting + adjacent-5 stage smoothing), not logistic regression
//! (logistic is the optional Logistello-1 path, done in Python).
//!
//! # Split of responsibilities (mirrors the Phase-4b "Rust owns
//! canonicalisation" decision)
//!
//! **Rust owns the base-literal extraction.** A position deterministically
//! maps to a set of *active base-literal ids* via [`BaseFeatureSpec`]. The
//! Rust CLI `glem-extract` subcommand emits, per position, the active
//! base-literal ids + label + stage in the documented `GLX1` columnar binary
//! (see `GLEM_FORMAT.md`). **Python only does** conjunction enumeration,
//! support filtering, the (reused Phase-4b) linear fit, and weight pruning;
//! it never re-implements literal extraction. The resulting [`GlemModel`]
//! (`GLM1` file) is loaded back here and evaluated by [`GlemEval`] with a
//! base-literal extractor that is **bit-identical** to the one the extractor
//! used (the GOLD interop test gates this).
//!
//! # `GLM1` model file (explicit little-endian, same discipline as `LGW1`)
//!
//! See [`GlemModel`] and `GLEM_FORMAT.md`.

use crate::eval_trait::LeafEvaluator;
use crate::stage::{N_STAGES, stage_for_state};
use othello_core::{Color, Coord, GameState};
use std::io;
use std::path::Path;

/// `GLM1` magic (ASCII `"GLM1"`, little-endian).
pub const GLEM_MAGIC: u32 = 0x314D_4C47;
/// `GLM1` format version.
pub const GLEM_VERSION: u32 = 1;

/// One base-feature *family*. Each family contributes a contiguous block of
/// base-literal ids to the global literal-id space; the block sizes and the
/// extraction logic are fixed here so Rust and Python agree bit-for-bit.
///
/// The set is intentionally **declarative / extensible**: to add a family,
/// add a variant, give it a `width()`, and extend `emit_family_literals`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BaseFamily {
    /// 64 cells × 3 ternary occupancy states, side-to-move POV.
    /// Literal `cell*3 + state`, `state ∈ {0=mine, 1=opp, 2=empty}`.
    /// Width = 192.
    Cell64,
    /// Bucketed legal-move count for the side to move.
    /// Buckets: `0 | 1..=2 | 3..=5 | 6..=9 | 10..` → 5 literals.
    Mobility,
    /// The 4 corners (A1,H1,A8,H8) × 3 ternary states, side-to-move POV.
    /// Literal `corner_index*3 + state`. Width = 12.
    Corner,
}

impl BaseFamily {
    /// Parses the canonical lowercase family name.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "cell64" => Some(BaseFamily::Cell64),
            "mobility" => Some(BaseFamily::Mobility),
            "corner" => Some(BaseFamily::Corner),
            _ => None,
        }
    }

    /// Canonical lowercase name (round-trips [`BaseFamily::parse`]).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            BaseFamily::Cell64 => "cell64",
            BaseFamily::Mobility => "mobility",
            BaseFamily::Corner => "corner",
        }
    }

    /// Stable wire id (serialized in `GLM1` / `GLX1`); never reused.
    #[must_use]
    pub fn wire_id(self) -> u8 {
        match self {
            BaseFamily::Cell64 => 0,
            BaseFamily::Mobility => 1,
            BaseFamily::Corner => 2,
        }
    }

    /// Family from its wire id.
    #[must_use]
    pub fn from_wire_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(BaseFamily::Cell64),
            1 => Some(BaseFamily::Mobility),
            2 => Some(BaseFamily::Corner),
            _ => None,
        }
    }

    /// Number of base-literal ids this family contributes.
    #[must_use]
    pub fn width(self) -> u32 {
        match self {
            BaseFamily::Cell64 => 64 * 3,
            BaseFamily::Mobility => 5,
            BaseFamily::Corner => 4 * 3,
        }
    }
}

/// Corner cell indices (A1, H1, A8, H8) as `idx = row*8 + col` (A1 = 0).
const CORNERS: [u8; 4] = [0, 7, 56, 63];

/// Mobility bucket for a legal-move count (`0 | 1-2 | 3-5 | 6-9 | 10+`).
#[inline]
#[must_use]
fn mobility_bucket(n: u32) -> u32 {
    match n {
        0 => 0,
        1..=2 => 1,
        3..=5 => 2,
        6..=9 => 3,
        _ => 4,
    }
}

/// Ternary occupancy code at board index `idx` from `side`'s POV
/// (`0 = mine, 1 = opponent, 2 = empty`) — identical convention to
/// `pattern::cell_code` / Edax `board_get_square_color`.
#[inline]
#[must_use]
fn cell_state(state: &GameState, idx: u8, side: Color) -> u32 {
    let coord = Coord::new(idx / 8, idx % 8);
    match state.board.cell(coord) {
        Some(c) if c == side => 0,
        Some(_) => 1,
        None => 2,
    }
}

/// An ordered list of base-feature families. The global base-literal id of a
/// family-local id is `offset(family) + local_id`, where `offset` is the sum
/// of the widths of the **preceding** families in this exact order. The
/// `(ordered families)` sequence **is** the "base-feature spec id" recorded
/// in `GLM1` / `GLX1` (a spec is identified by its family wire-id sequence).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseFeatureSpec {
    families: Vec<BaseFamily>,
}

/// Error parsing a [`BaseFeatureSpec`] from a comma list.
#[derive(Debug)]
pub struct SpecParseError(pub String);

impl std::fmt::Display for SpecParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "base-feature spec error: {}", self.0)
    }
}

impl std::error::Error for SpecParseError {}

impl BaseFeatureSpec {
    /// Builds a spec from an ordered family list. Duplicates are rejected so
    /// the literal-id space is unambiguous.
    ///
    /// # Errors
    ///
    /// Returns [`SpecParseError`] if `families` is empty or has duplicates.
    pub fn new(families: Vec<BaseFamily>) -> Result<Self, SpecParseError> {
        if families.is_empty() {
            return Err(SpecParseError("empty family list".into()));
        }
        for (i, f) in families.iter().enumerate() {
            if families[..i].contains(f) {
                return Err(SpecParseError(format!("duplicate family {}", f.name())));
            }
        }
        Ok(Self { families })
    }

    /// Parses a comma-separated list like `cell64,mobility,corner`.
    ///
    /// # Errors
    ///
    /// Returns [`SpecParseError`] on an unknown / duplicate / empty list.
    pub fn parse(s: &str) -> Result<Self, SpecParseError> {
        let mut fams = Vec::new();
        for tok in s.split(',') {
            let tok = tok.trim();
            if tok.is_empty() {
                continue;
            }
            let f = BaseFamily::parse(tok)
                .ok_or_else(|| SpecParseError(format!("unknown family '{tok}'")))?;
            fams.push(f);
        }
        Self::new(fams)
    }

    /// The ordered families.
    #[must_use]
    pub fn families(&self) -> &[BaseFamily] {
        &self.families
    }

    /// Canonical comma string (round-trips [`BaseFeatureSpec::parse`]).
    #[must_use]
    pub fn to_spec_string(&self) -> String {
        self.families
            .iter()
            .map(|f| f.name())
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Global base-literal-id offset of `family` in this spec (sum of widths
    /// of the families before it). `None` if the family is not in the spec.
    #[must_use]
    pub fn offset(&self, family: BaseFamily) -> Option<u32> {
        let mut off = 0u32;
        for &f in &self.families {
            if f == family {
                return Some(off);
            }
            off += f.width();
        }
        None
    }

    /// Total number of base-literal ids in this spec.
    #[must_use]
    pub fn n_literals(&self) -> u32 {
        self.families.iter().map(|f| f.width()).sum()
    }

    /// The **sorted, deduplicated** set of active global base-literal ids for
    /// `state` from its side-to-move's point of view. This is the single
    /// source of truth shared by `glem-extract` and [`GlemEval`].
    #[must_use]
    pub fn active_literals(&self, state: &GameState) -> Vec<u32> {
        let side = state.side_to_move;
        let mut out = Vec::with_capacity(self.families.len() * 8);
        let mut off = 0u32;
        for &fam in &self.families {
            emit_family_literals(fam, state, side, off, &mut out);
            off += fam.width();
        }
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// Appends the active global literal ids contributed by `fam` for `state`
/// (POV `side`), offset by `off`. Adding a family means extending this match
/// plus [`BaseFamily::width`] / parsing.
fn emit_family_literals(
    fam: BaseFamily,
    state: &GameState,
    side: Color,
    off: u32,
    out: &mut Vec<u32>,
) {
    match fam {
        BaseFamily::Cell64 => {
            for idx in 0u8..64 {
                let s = cell_state(state, idx, side);
                out.push(off + u32::from(idx) * 3 + s);
            }
        }
        BaseFamily::Mobility => {
            let n = state.board.legal_moves(side).len() as u32;
            out.push(off + mobility_bucket(n));
        }
        BaseFamily::Corner => {
            for (ci, &idx) in CORNERS.iter().enumerate() {
                let s = cell_state(state, idx, side);
                out.push(off + (ci as u32) * 3 + s);
            }
        }
    }
}

/// One selected GLEM feature: a conjunction (logical AND) of base-literal
/// ids. A position "has" the feature iff **all** member literals are active.
/// Order-1 features have a single member (GLEM then reduces to a plain
/// per-literal linear model — design doc §4.3.6 with `max_order = 1`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conjunction {
    /// Sorted, deduplicated member base-literal ids.
    pub members: Vec<u32>,
}

impl Conjunction {
    /// Builds a conjunction, normalising members (sorted + deduped).
    #[must_use]
    pub fn new(mut members: Vec<u32>) -> Self {
        members.sort_unstable();
        members.dedup();
        Self { members }
    }

    /// Conjunction order (number of distinct member literals).
    #[must_use]
    pub fn order(&self) -> usize {
        self.members.len()
    }

    /// Whether every member literal is in the (sorted) `active` set.
    #[must_use]
    pub fn satisfied(&self, active: &[u32]) -> bool {
        self.members.iter().all(|m| active.binary_search(m).is_ok())
    }
}

/// Error decoding / doing I/O on a [`GlemModel`] `GLM1` blob.
#[derive(Debug)]
pub enum GlemError {
    /// Structurally invalid byte stream.
    Format(String),
    /// Underlying I/O error.
    Io(io::Error),
}

impl std::fmt::Display for GlemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GlemError::Format(m) => write!(f, "GLM1 format error: {m}"),
            GlemError::Io(e) => write!(f, "GLM1 I/O error: {e}"),
        }
    }
}

impl std::error::Error for GlemError {}

impl From<io::Error> for GlemError {
    fn from(e: io::Error) -> Self {
        GlemError::Io(e)
    }
}

/// A loaded `GLM1` GLEM model: the base-feature spec, the selected
/// conjunctions (post support-filter + weight-prune), and a per-stage
/// (`N_STAGES` = 13) weight vector parallel to the selected-feature list.
///
/// # `GLM1` serialization contract (explicit little-endian)
///
/// Byte-identical between Rust ([`GlemModel::to_bytes`] /
/// [`GlemModel::from_bytes`]) and the Python writer
/// `logistello_tools.glm1`. Layout (all integers LE):
///
/// | field            | type       | meaning                               |
/// |------------------|------------|---------------------------------------|
/// | magic            | `u32`      | [`GLEM_MAGIC`] (`"GLM1"`)              |
/// | version          | `u32`      | [`GLEM_VERSION`] = 1                   |
/// | n_stages         | `u32`      | `13`                                  |
/// | n_families       | `u32`      | spec family count                     |
/// | family_ids       | `[u8; F]`  | spec family wire ids, in spec order   |
/// | n_features       | `u64`      | number of selected conjunctions       |
/// | per feature      | …          | `order: u32`, then `order × u32` member literal ids |
/// | weights          | `i32[]`    | `n_stages × n_features`, row-major stage→feature, 1/128-disc units |
///
/// Weights are in **1/128-disc units** with the same Edax bias rounding /
/// `±63` clamp as [`PatternEval`](crate::pattern_eval::PatternEval), so a
/// heuristic leaf can never rival an exact terminal/mate value (design doc
/// §4.5 B6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlemModel {
    spec: BaseFeatureSpec,
    features: Vec<Conjunction>,
    /// `w[stage]` is parallel to `features`. Units: 1/128 disc.
    w: Vec<Vec<i32>>,
}

impl GlemModel {
    /// Builds a model from its parts. Panics in debug if shapes disagree.
    #[must_use]
    pub fn new(spec: BaseFeatureSpec, features: Vec<Conjunction>, w: Vec<Vec<i32>>) -> Self {
        debug_assert_eq!(w.len(), N_STAGES, "GLM1 must have N_STAGES weight rows");
        for row in &w {
            debug_assert_eq!(row.len(), features.len(), "weight row != feature count");
        }
        Self { spec, features, w }
    }

    /// An all-zero model (no features): every position evaluates to `0`.
    #[must_use]
    pub fn zeros(spec: BaseFeatureSpec) -> Self {
        Self {
            spec,
            features: Vec::new(),
            w: vec![Vec::new(); N_STAGES],
        }
    }

    /// The base-feature spec.
    #[must_use]
    pub fn spec(&self) -> &BaseFeatureSpec {
        &self.spec
    }

    /// The selected conjunctions.
    #[must_use]
    pub fn features(&self) -> &[Conjunction] {
        &self.features
    }

    /// Per-stage weight row (length = number of features).
    #[must_use]
    pub fn weights(&self, stage: usize) -> &[i32] {
        &self.w[stage]
    }

    /// Number of selected conjunctions.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.features.len()
    }

    /// Raw 1/128-disc-unit weight sum (before rounding) for `state` from its
    /// side-to-move's POV: sum the stage weights of every selected
    /// conjunction satisfied by the position's active base literals.
    #[must_use]
    pub fn raw_sum(&self, state: &GameState) -> i64 {
        let active = self.spec.active_literals(state);
        let stage = stage_for_state(state);
        let row = &self.w[stage];
        let mut sum = 0i64;
        for (i, f) in self.features.iter().enumerate() {
            if f.satisfied(&active) {
                sum += i64::from(row[i]);
            }
        }
        sum
    }

    /// Serializes to the explicit little-endian `GLM1` blob (byte-identical
    /// to the Python writer; see the [`GlemModel`] doc / `GLEM_FORMAT.md`).
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&GLEM_MAGIC.to_le_bytes());
        out.extend_from_slice(&GLEM_VERSION.to_le_bytes());
        out.extend_from_slice(&(N_STAGES as u32).to_le_bytes());
        let fams = self.spec.families();
        out.extend_from_slice(&(fams.len() as u32).to_le_bytes());
        for f in fams {
            out.push(f.wire_id());
        }
        out.extend_from_slice(&(self.features.len() as u64).to_le_bytes());
        for f in &self.features {
            out.extend_from_slice(&(f.members.len() as u32).to_le_bytes());
            for &m in &f.members {
                out.extend_from_slice(&m.to_le_bytes());
            }
        }
        for row in &self.w {
            for &x in row {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out
    }

    /// Deserializes an explicit little-endian `GLM1` blob.
    ///
    /// # Errors
    ///
    /// Returns [`GlemError::Format`] on a structural mismatch (bad
    /// magic/version, wrong stage count, unknown family id, truncation, or
    /// trailing bytes).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, GlemError> {
        let err = |m: &str| GlemError::Format(m.to_string());
        let mut off = 0usize;
        let need = |off: usize, n: usize, bytes: &[u8]| -> Result<(), GlemError> {
            if off + n > bytes.len() {
                Err(GlemError::Format("truncated".into()))
            } else {
                Ok(())
            }
        };
        let rd_u32 = |off: &mut usize, bytes: &[u8]| -> Result<u32, GlemError> {
            need(*off, 4, bytes)?;
            let v = u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap());
            *off += 4;
            Ok(v)
        };
        let rd_u64 = |off: &mut usize, bytes: &[u8]| -> Result<u64, GlemError> {
            need(*off, 8, bytes)?;
            let v = u64::from_le_bytes(bytes[*off..*off + 8].try_into().unwrap());
            *off += 8;
            Ok(v)
        };

        if rd_u32(&mut off, bytes)? != GLEM_MAGIC {
            return Err(err("bad magic (expected GLM1)"));
        }
        if rd_u32(&mut off, bytes)? != GLEM_VERSION {
            return Err(err("unsupported GLM1 version"));
        }
        if rd_u32(&mut off, bytes)? as usize != N_STAGES {
            return Err(err("stage count mismatch"));
        }
        let n_fam = rd_u32(&mut off, bytes)? as usize;
        if n_fam == 0 {
            return Err(err("empty family list"));
        }
        need(off, n_fam, bytes)?;
        let mut fams = Vec::with_capacity(n_fam);
        for _ in 0..n_fam {
            let id = bytes[off];
            off += 1;
            fams.push(BaseFamily::from_wire_id(id).ok_or_else(|| err("unknown family wire id"))?);
        }
        let spec = BaseFeatureSpec::new(fams).map_err(|e| err(&e.0))?;
        let n_feat = rd_u64(&mut off, bytes)? as usize;
        let mut features = Vec::with_capacity(n_feat);
        let n_lit = spec.n_literals();
        for _ in 0..n_feat {
            let order = rd_u32(&mut off, bytes)? as usize;
            if order == 0 {
                return Err(err("conjunction of order 0"));
            }
            let mut members = Vec::with_capacity(order);
            for _ in 0..order {
                let m = rd_u32(&mut off, bytes)?;
                if m >= n_lit {
                    return Err(err("member literal id out of range for spec"));
                }
                members.push(m);
            }
            // Stored normalised; reject non-normalised input defensively.
            let c = Conjunction::new(members.clone());
            if c.members != members {
                return Err(err("conjunction members not sorted/deduped"));
            }
            features.push(c);
        }
        let mut w = Vec::with_capacity(N_STAGES);
        for _ in 0..N_STAGES {
            let mut row = Vec::with_capacity(n_feat);
            for _ in 0..n_feat {
                let x = rd_u32(&mut off, bytes)? as i32;
                row.push(x);
            }
            w.push(row);
        }
        if off != bytes.len() {
            return Err(err("trailing bytes after GLM1 body"));
        }
        Ok(Self { spec, features, w })
    }

    /// Loads a `GLM1` model file.
    ///
    /// # Errors
    ///
    /// [`GlemError::Io`] if unreadable, [`GlemError::Format`] if not a valid
    /// `GLM1` blob.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, GlemError> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    /// Writes this model as a `GLM1` file.
    ///
    /// # Errors
    ///
    /// [`GlemError::Io`] if the file cannot be written.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), GlemError> {
        std::fs::write(path, self.to_bytes())?;
        Ok(())
    }
}

/// GLEM leaf evaluator (`LeafEvaluator`): scores a position by summing the
/// per-stage weights of the selected conjunctions it satisfies, with the
/// **same** Edax bias rounding / `±63` clamp as
/// [`PatternEval`](crate::pattern_eval::PatternEval) (1/128-disc units;
/// design doc §4.5 B6). Side-to-move POV.
#[derive(Debug, Clone)]
pub struct GlemEval {
    model: GlemModel,
}

impl GlemEval {
    /// Builds an evaluator from a loaded model.
    #[must_use]
    pub fn new(model: GlemModel) -> Self {
        Self { model }
    }

    /// All-zero GLEM evaluator over `spec` (every position → `0`).
    #[must_use]
    pub fn zeros(spec: BaseFeatureSpec) -> Self {
        Self {
            model: GlemModel::zeros(spec),
        }
    }

    /// Borrow the underlying model.
    #[must_use]
    pub fn model(&self) -> &GlemModel {
        &self.model
    }

    /// Raw pre-rounding 1/128-disc weight sum (diagnostics / tests).
    #[must_use]
    pub fn raw_sum(&self, state: &GameState) -> i64 {
        self.model.raw_sum(state)
    }
}

impl LeafEvaluator for GlemEval {
    #[inline]
    fn eval(&self, state: &GameState) -> i32 {
        let sum = self.model.raw_sum(state);
        // Identical convention to PatternEval (Edax midgame.c:36-44).
        let biased = if sum > 0 { sum + 64 } else { sum - 64 };
        let score = (biased / 128) as i32;
        score.clamp(-63, 63)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use othello_core::Move;
    use rand::seq::SliceRandom;
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    fn random_position(rng: &mut ChaCha20Rng, plies: usize) -> GameState {
        let mut s = GameState::standard_8x8();
        let mut made = 0;
        while made < plies {
            if s.is_terminal() {
                break;
            }
            let moves = s.legal_moves();
            if moves.is_empty() {
                s.apply_move(Move::Pass).expect("pass legal");
                continue;
            }
            let m = *moves.choose(rng).expect("non-empty");
            s.apply_move(m).expect("legal");
            made += 1;
        }
        s
    }

    #[test]
    fn spec_parse_offsets_and_widths() {
        let spec = BaseFeatureSpec::parse("cell64,mobility,corner").unwrap();
        assert_eq!(spec.families().len(), 3);
        assert_eq!(spec.offset(BaseFamily::Cell64), Some(0));
        assert_eq!(spec.offset(BaseFamily::Mobility), Some(192));
        assert_eq!(spec.offset(BaseFamily::Corner), Some(197));
        assert_eq!(spec.n_literals(), 192 + 5 + 12);
        assert_eq!(spec.to_spec_string(), "cell64,mobility,corner");
        assert!(BaseFeatureSpec::parse("cell64,cell64").is_err());
        assert!(BaseFeatureSpec::parse("nope").is_err());
        assert!(BaseFeatureSpec::parse("").is_err());
    }

    #[test]
    fn active_literals_sorted_unique_and_in_range() {
        let spec = BaseFeatureSpec::parse("cell64,mobility,corner").unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(0x00A1_1CE5);
        for _ in 0..200 {
            let plies = (rng.next_u32() % 55) as usize;
            let s = random_position(&mut rng, plies);
            let lits = spec.active_literals(&s);
            assert!(lits.windows(2).all(|w| w[0] < w[1]), "sorted+unique");
            for &l in &lits {
                assert!(l < spec.n_literals());
            }
            // cell64 contributes exactly 64 (one state per cell), corner 4,
            // mobility 1 -> 69 active literals total (distinct id ranges).
            assert_eq!(lits.len(), 64 + 1 + 4);
        }
    }

    #[test]
    fn start_position_known_literals() {
        // Standard start, Black to move. D4=White,E4=Black,D5=Black,E5=White.
        // idx = row*8+col. D4 -> r3,c3 = 27; E4 -> r3,c4 = 28;
        // D5 -> r4,c3 = 35; E5 -> r4,c4 = 36.
        let spec = BaseFeatureSpec::parse("cell64").unwrap();
        let s = GameState::standard_8x8();
        let lits = spec.active_literals(&s);
        // Black POV: 28 (E4) and 35 (D5) are "mine" (state 0);
        // 27 (D4) and 36 (E5) are "opp" (state 1); all others empty (2).
        assert!(lits.contains(&(28 * 3)), "E4 mine");
        assert!(lits.contains(&(35 * 3)), "D5 mine");
        assert!(lits.contains(&(27 * 3 + 1)), "D4 opp");
        assert!(lits.contains(&(36 * 3 + 1)), "E5 opp");
        assert!(lits.contains(&2), "A1 empty (idx 0, state 2)");
    }

    #[test]
    fn conjunction_satisfied_semantics() {
        let active = vec![1u32, 3, 7, 10];
        assert!(Conjunction::new(vec![3, 1]).satisfied(&active));
        assert!(Conjunction::new(vec![7]).satisfied(&active));
        assert!(!Conjunction::new(vec![1, 2]).satisfied(&active));
        // Order normalised + deduped.
        let c = Conjunction::new(vec![5, 1, 5, 1]);
        assert_eq!(c.members, vec![1, 5]);
        assert_eq!(c.order(), 2);
    }

    #[test]
    fn glm1_roundtrip_bit_exact() {
        let spec = BaseFeatureSpec::parse("cell64,corner").unwrap();
        let feats = vec![
            Conjunction::new(vec![0]),
            Conjunction::new(vec![5, 191]),
            Conjunction::new(vec![193, 200, 7]),
        ];
        let mut w = vec![vec![0i32; feats.len()]; N_STAGES];
        for (s, row) in w.iter_mut().enumerate() {
            for (i, x) in row.iter_mut().enumerate() {
                *x = (s as i32) * 11 + (i as i32) * 3 - 17;
            }
        }
        let m = GlemModel::new(spec, feats, w);
        let bytes = m.to_bytes();
        let back = GlemModel::from_bytes(&bytes).expect("deserialize");
        assert_eq!(m, back, "GLM1 round-trip must be bit-exact");
        assert_eq!(back.to_bytes(), bytes);
    }

    #[test]
    fn glm1_rejects_bad_magic_and_trailing() {
        let spec = BaseFeatureSpec::parse("mobility").unwrap();
        let m = GlemModel::zeros(spec);
        let mut b = m.to_bytes();
        b[0] ^= 0xFF;
        assert!(GlemModel::from_bytes(&b).is_err());
        let mut longer = m.to_bytes();
        longer.push(0);
        assert!(GlemModel::from_bytes(&longer).is_err());
        assert!(GlemModel::from_bytes(&m.to_bytes()[..3]).is_err());
    }

    #[test]
    fn zeros_evaluates_to_zero_everywhere() {
        let spec = BaseFeatureSpec::parse("cell64,mobility,corner").unwrap();
        let ev = GlemEval::zeros(spec);
        let mut rng = ChaCha20Rng::seed_from_u64(0x0000_BEE0);
        for _ in 0..150 {
            let plies = (rng.next_u32() % 55) as usize;
            let s = random_position(&mut rng, plies);
            assert_eq!(ev.eval(&s), 0);
            assert_eq!(ev.raw_sum(&s), 0);
        }
    }

    #[test]
    fn eval_rounding_matches_pattern_eval_convention() {
        // A single always-on order-1 mobility literal with a known weight;
        // the rounded score must equal round(w/128) with Edax bias + clamp.
        let spec = BaseFeatureSpec::parse("mobility").unwrap();
        let s = GameState::standard_8x8();
        let active = spec.active_literals(&s);
        assert_eq!(active.len(), 1);
        let feat = Conjunction::new(vec![active[0]]);
        for raw in [-300i32, -64, 0, 65, 200, 8192, -8192] {
            let mut w = vec![vec![0i32]; N_STAGES];
            for row in &mut w {
                row[0] = raw;
            }
            let m = GlemModel::new(spec.clone(), vec![feat.clone()], w);
            let ev = GlemEval::new(m);
            let sum = i64::from(raw);
            let biased = if sum > 0 { sum + 64 } else { sum - 64 };
            let want = ((biased / 128) as i32).clamp(-63, 63);
            assert_eq!(ev.eval(&s), want, "raw={raw}");
        }
    }

    #[test]
    fn max_order_1_is_per_literal_linear_model() {
        // With only order-1 conjunctions, GlemEval == Σ w_l over active
        // literals l (design doc §4.3.6 max_order=1 reduces to plain linear).
        let spec = BaseFeatureSpec::parse("corner").unwrap();
        let n = spec.n_literals();
        let feats: Vec<_> = (0..n).map(|l| Conjunction::new(vec![l])).collect();
        let mut w = vec![vec![0i32; feats.len()]; N_STAGES];
        for (s, row) in w.iter_mut().enumerate() {
            for (i, x) in row.iter_mut().enumerate() {
                *x = (i as i32) - (s as i32);
            }
        }
        let m = GlemModel::new(spec.clone(), feats, w);
        let ev = GlemEval::new(m);
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        for _ in 0..50 {
            let plies = (rng.next_u32() % 50) as usize;
            let st = random_position(&mut rng, plies);
            let active = spec.active_literals(&st);
            let stg = stage_for_state(&st);
            let want: i64 = active
                .iter()
                .map(|&l| i64::from((l as i32) - (stg as i32)))
                .sum();
            assert_eq!(ev.raw_sum(&st), want);
        }
    }
}
