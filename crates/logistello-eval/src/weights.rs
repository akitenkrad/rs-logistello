//! Edax pack/unpack symmetry normalisation + learned-weight container —
//! design doc §4.4 B4.
//!
//! # Pack / unpack (Edax `src/eval.c:498-560`)
//!
//! Edax stores **one weight per canonical class**, not per raw key. Two raw
//! keys share a weight when one is the symmetry image (board reflection
//! within the pattern) of the other. The mapping is built by, for each raw
//! key `i` in `0..3^k`:
//!
//! - `j = player_feature(sym, k, i)` — the digit-permuted (symmetric) key
//!   (`f += ((l / 3^{sym[t]}) mod 3) * 3^t`, `eval.c:517-527`);
//! - if `j < i`, key `i` *shares* the canonical slot already assigned to `j`;
//!   otherwise `i` gets a fresh canonical index (`eval.c:553-558`).
//!
//! The colour-symmetry partner is recorded via `opponent_feature` with the
//! digit map `o = [1, 0, 2]` (swap player↔opponent, keep empty;
//! `eval.c:498-506`): `pack[1][opponent_feature(i)] = pack[0][i]`.
//!
//! Symmetry generators are vendored verbatim from `eval.c:581-583`:
//! `sym_S10 = {9,8,7,6,5,4,3,2,1,0}`, `sym_C10 = {9,8,7,6,4,5,3,2,1,0}`,
//! `sym_C9 = {0,2,1,4,3,5,7,6,8}`. S8/S7/S6/S5/S4 use `sym_S10`'s
//! `+2/+3/+4/+5/+6` tail slices (`eval.c:591-599`).
//!
//! # Validation
//!
//! The per-type canonical counts must equal Edax `EVAL_PACKED_SIZE`
//! (`eval.c:443`): `{10206, 29889, 29646, 29646, 3321, 3321, 3321, 3321,
//! 1134, 378, 135, 45, 1}` in accumulate order.
//!
//! # `EvalWeights` serialization contract (`LGW1`, explicit little-endian)
//!
//! See [`EvalWeights`] and `WEIGHTS_FORMAT.md`. This is the binary contract
//! the Phase 4b Python trainer writes and this crate reads. It is a fully
//! explicit little-endian byte layout (no `serde`/`bincode` internals) so
//! Rust and Python emit/consume **byte-identical** files. Weights are in
//! **1/128-disc units** (design doc B1; Edax `midgame.c:36-44`).

use crate::pattern::PatternType;
use crate::stage::N_STAGES;
use std::io;
use std::path::Path;

/// `3^k` lookup for `k = 0..=10`.
const POW3: [u32; 11] = [1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049];

// --- Edax symmetry generators (eval.c:581-583), vendored verbatim. ---
const SYM_S10: [usize; 10] = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
const SYM_C10: [usize; 10] = [9, 8, 7, 6, 4, 5, 3, 2, 1, 0];
const SYM_C9: [usize; 9] = [0, 2, 1, 4, 3, 5, 7, 6, 8];

/// Edax `EVAL_PACKED_SIZE` (`eval.c:443`), canonical-class counts per
/// pattern *type* keyed by the accumulate-order group. Order:
/// C9, C10, S10, S10, S8, S8, S8, S8, S7, S6, S5, S4, const.
pub const EVAL_PACKED_SIZE: [u32; 13] = [
    10206, 29889, 29646, 29646, 3321, 3321, 3321, 3321, 1134, 378, 135, 45, 1,
];

/// Edax `player_feature` (`eval.c:517-527`): digit permutation of the
/// base-3 key `l` of length `n` under symmetry `sym`. Returns the
/// symmetric key.
#[must_use]
fn player_feature(sym: &[usize], n: usize, l: u32) -> u32 {
    let mut f = 0u32;
    for (i, &s) in sym.iter().take(n).enumerate() {
        f += ((l / POW3[s]) % 3) * POW3[i];
    }
    f
}

/// Edax `opponent_feature` (`eval.c:498-506`): colour-swap each base-3
/// digit of `l` (length `d`) using `o = [1, 0, 2]`.
#[must_use]
fn opponent_feature(l: u32, d: usize) -> u32 {
    const O: [u32; 3] = [1, 0, 2];
    let mut f = O[(l % 3) as usize];
    if d > 1 {
        f += opponent_feature(l / 3, d - 1) * 3;
    }
    f
}

/// Symmetry generator slice for a pattern type, per `eval.c:591-599`.
/// S8/S7/S6/S5/S4 use `sym_S10` tail slices.
fn sym_for(ty: PatternType) -> &'static [usize] {
    match ty {
        PatternType::C9 => &SYM_C9,
        PatternType::C10 => &SYM_C10,
        PatternType::S10 => &SYM_S10,
        PatternType::S8 => &SYM_S10[2..], // sym_S10 + 2
        PatternType::S7 => &SYM_S10[3..], // sym_S10 + 3
        PatternType::S6 => &SYM_S10[4..], // sym_S10 + 4
        PatternType::S5 => &SYM_S10[5..], // sym_S10 + 5
        PatternType::S4 => &SYM_S10[6..], // sym_S10 + 6
        PatternType::Const => &[],
    }
}

/// Per-pattern-type pack table (Edax `unpack`, `eval.c:548-560`):
/// `raw_to_canon[player]` (pack[0]) and `raw_to_canon[opponent]` (pack[1]).
#[derive(Debug, Clone)]
pub struct PackTable {
    /// `pack[0]`: raw key (player POV) -> canonical class index.
    pub player: Vec<u32>,
    /// `pack[1]`: raw key (opponent POV) -> canonical class index.
    pub opponent: Vec<u32>,
    /// Number of distinct canonical classes (`EVAL_PACKED_SIZE` entry).
    pub n_canonical: u32,
}

impl PackTable {
    /// Builds the pack/unpack table for `ty` exactly as Edax `unpack`.
    #[must_use]
    pub fn build(ty: PatternType) -> Self {
        if ty == PatternType::Const {
            return Self {
                player: vec![0],
                opponent: vec![0],
                n_canonical: 1,
            };
        }
        let k = ty.k();
        let size = POW3[k];
        let sym = sym_for(ty);
        let mut player = vec![0u32; size as usize];
        let mut opponent = vec![0u32; size as usize];
        let mut n = 0u32;
        for i in 0..size {
            let j = player_feature(sym, k, i);
            if j < i {
                player[i as usize] = player[j as usize];
            } else {
                player[i as usize] = n;
                n += 1;
            }
            opponent[opponent_feature(i, k) as usize] = player[i as usize];
        }
        Self {
            player,
            opponent,
            n_canonical: n,
        }
    }
}

/// Cached pack tables for the 9 distinct pattern types (the 8 real symmetry
/// classes + the degenerate constant), keyed by [`PatternType`].
#[derive(Debug, Clone)]
pub struct PackTables {
    c9: PackTable,
    c10: PackTable,
    s10: PackTable,
    s8: PackTable,
    s7: PackTable,
    s6: PackTable,
    s5: PackTable,
    s4: PackTable,
    konst: PackTable,
}

impl PackTables {
    /// Builds all pack tables (one per distinct pattern type).
    #[must_use]
    pub fn build() -> Self {
        Self {
            c9: PackTable::build(PatternType::C9),
            c10: PackTable::build(PatternType::C10),
            s10: PackTable::build(PatternType::S10),
            s8: PackTable::build(PatternType::S8),
            s7: PackTable::build(PatternType::S7),
            s6: PackTable::build(PatternType::S6),
            s5: PackTable::build(PatternType::S5),
            s4: PackTable::build(PatternType::S4),
            konst: PackTable::build(PatternType::Const),
        }
    }

    /// Pack table for a pattern type.
    #[must_use]
    pub fn get(&self, ty: PatternType) -> &PackTable {
        match ty {
            PatternType::C9 => &self.c9,
            PatternType::C10 => &self.c10,
            PatternType::S10 => &self.s10,
            PatternType::S8 => &self.s8,
            PatternType::S7 => &self.s7,
            PatternType::S6 => &self.s6,
            PatternType::S5 => &self.s5,
            PatternType::S4 => &self.s4,
            PatternType::Const => &self.konst,
        }
    }

    /// Canonical class count for a pattern type (= `EVAL_PACKED_SIZE`).
    #[must_use]
    pub fn n_canonical(&self, ty: PatternType) -> u32 {
        self.get(ty).n_canonical
    }
}

/// Distinct pattern types in `EvalWeights` storage / accumulate order.
/// Each *type* owns one shared weight vector per stage; all feature
/// instances of that type index into it through the pack table.
pub const WEIGHT_TYPE_ORDER: [PatternType; 9] = [
    PatternType::C9,
    PatternType::C10,
    PatternType::S10,
    PatternType::S8,
    PatternType::S7,
    PatternType::S6,
    PatternType::S5,
    PatternType::S4,
    PatternType::Const,
];

#[inline]
fn type_slot(ty: PatternType) -> usize {
    match ty {
        PatternType::C9 => 0,
        PatternType::C10 => 1,
        PatternType::S10 => 2,
        PatternType::S8 => 3,
        PatternType::S7 => 4,
        PatternType::S6 => 5,
        PatternType::S5 => 6,
        PatternType::S4 => 7,
        PatternType::Const => 8,
    }
}

/// Magic identifying a `logistello-eval` weights blob: ASCII `LGW1`.
pub const WEIGHTS_MAGIC: u32 = 0x3157_474C; // little-endian "LGW1"

/// Learned per-stage, per-pattern-type **canonical** weight vectors.
///
/// # Storage model
///
/// For each of the [`N_STAGES`] = 13 stages and each of the 9 distinct
/// pattern types there is one `Vec<i32>` of length
/// `PackTables::n_canonical(ty)` (= the matching [`EVAL_PACKED_SIZE`]
/// entry). A raw feature key is resolved to its canonical slot through the
/// pack table, then looked up in the type's per-stage vector — so all
/// symmetry-equivalent raw keys (and, for the player-POV path, only the
/// player table) share one trainable weight, exactly as Edax (design doc
/// B4). Weights are in **1/128-disc units** (design doc B1).
///
/// # Serialization contract (explicit LE, see `WEIGHTS_FORMAT.md`)
///
/// The on-disk layout is a fixed little-endian byte stream (no `serde` /
/// `bincode`), so Rust and Python produce byte-identical files:
///
/// | offset | type      | meaning                                          |
/// |--------|-----------|--------------------------------------------------|
/// | 0      | `u32` LE  | magic = [`WEIGHTS_MAGIC`] (ASCII `LGW1`)          |
/// | 4      | `u32` LE  | version = `1`                                    |
/// | 8      | `u32` LE  | n_stages = `13`                                  |
/// | 12     | `u32` LE  | n_types = `9`                                    |
/// | 16     | `[u32;9]` | canon_sizes (C9,C10,S10,S8,S7,S6,S5,S4,Const) LE |
/// | 52     | `u64` LE  | data_len = `13 * Σ canon_sizes` = `971815`       |
/// | 60     | `i32[]`   | data, `data_len` LE `i32` weights                |
///
/// `data` is row-major over `stage (0..13)` then `type (0..9)`
/// (`WEIGHT_TYPE_ORDER`) then `canonical_index (0..canon_sizes[type])`.
/// Total file size = `60 + 4 * 971815` bytes. The Python trainer writes
/// this exact structure; loading Edax's own binary `eval.dat` is **not**
/// implemented (its packed layout differs — see `WEIGHTS_FORMAT.md`
/// "Future / optional").
#[derive(Debug, Clone)]
pub struct EvalWeights {
    /// `w[stage][type_slot]` — per-stage canonical weight vector for each of
    /// the 9 pattern types (length = that type's canonical count).
    w: [[Vec<i32>; 9]; N_STAGES],
    /// Cached pack tables (not serialized; rebuilt deterministically — and
    /// thus excluded from equality, which compares only the weights).
    tables: PackTables,
}

impl PartialEq for EvalWeights {
    fn eq(&self, other: &Self) -> bool {
        self.w == other.w
    }
}

impl Eq for EvalWeights {}

/// Fixed-size header of the explicit `LGW1` layout (everything before the
/// variable-length `data` array): magic, version, n_stages, n_types,
/// `[u32; 9]` canon_sizes, then a `u64` data length.
const LGW1_HEADER_BYTES: usize = 4 + 4 + 4 + 4 + 9 * 4 + 8;

/// Error decoding a [`EvalWeights`] `LGW1` blob (bad structure or I/O).
#[derive(Debug)]
pub enum WeightsError {
    /// The byte stream is structurally invalid (bad magic/version, wrong
    /// stage/type count, canonical-size mismatch, truncated, or trailing
    /// bytes).
    Format(String),
    /// Underlying I/O error from [`EvalWeights::load`] /
    /// [`EvalWeights::save`].
    Io(io::Error),
}

impl std::fmt::Display for WeightsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WeightsError::Format(m) => write!(f, "LGW1 format error: {m}"),
            WeightsError::Io(e) => write!(f, "LGW1 I/O error: {e}"),
        }
    }
}

impl std::error::Error for WeightsError {}

impl From<io::Error> for WeightsError {
    fn from(e: io::Error) -> Self {
        WeightsError::Io(e)
    }
}

impl EvalWeights {
    /// All-zero weights (every canonical slot = 0). Pack tables are built
    /// deterministically from the vendored arrays.
    #[must_use]
    pub fn zeros() -> Self {
        let tables = PackTables::build();
        let w = std::array::from_fn(|_| {
            std::array::from_fn(|t| {
                let ty = WEIGHT_TYPE_ORDER[t];
                vec![0i32; tables.n_canonical(ty) as usize]
            })
        });
        Self { w, tables }
    }

    /// Cached pack tables (canonical-index resolution for raw feature keys).
    #[must_use]
    pub fn tables(&self) -> &PackTables {
        &self.tables
    }

    /// Canonical class count for a pattern type.
    #[must_use]
    pub fn n_canonical(&self, ty: PatternType) -> u32 {
        self.tables.n_canonical(ty)
    }

    /// Mutable access to a stage/type weight vector (used by tests &, in
    /// future, an in-Rust loader). Indexed by canonical class.
    pub fn weights_mut(&mut self, stage: usize, ty: PatternType) -> &mut [i32] {
        &mut self.w[stage][type_slot(ty)]
    }

    /// Canonical-class weight vector for `(stage, ty)`.
    #[must_use]
    pub fn weights(&self, stage: usize, ty: PatternType) -> &[i32] {
        &self.w[stage][type_slot(ty)]
    }

    /// Looks up the canonical weight for one feature instance: resolves the
    /// raw `key` through the **player** pack table for `ty`, then indexes
    /// the `(stage, ty)` weight vector. Units: 1/128 disc.
    #[must_use]
    pub fn lookup(&self, stage: usize, ty: PatternType, key: u32) -> i32 {
        let table = self.tables.get(ty);
        let canon = table.player[key as usize] as usize;
        self.w[stage][type_slot(ty)][canon]
    }

    /// Total number of stored canonical weights (`13 * Σ EVAL_PACKED_SIZE`).
    #[must_use]
    pub fn total_weights(&self) -> usize {
        self.w.iter().flat_map(|s| s.iter()).map(|v| v.len()).sum()
    }

    /// Serializes to the explicit little-endian `LGW1` blob (see
    /// [`EvalWeights`] doc / `WEIGHTS_FORMAT.md`). This is the byte-for-byte
    /// contract the Python trainer reproduces.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut canon_sizes = [0u32; 9];
        for (t, vec) in self.w[0].iter().enumerate() {
            canon_sizes[t] = vec.len() as u32;
        }
        let data_len = self.total_weights() as u64;
        let mut out = Vec::with_capacity(LGW1_HEADER_BYTES + 4 * data_len as usize);
        out.extend_from_slice(&WEIGHTS_MAGIC.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes()); // version
        out.extend_from_slice(&(N_STAGES as u32).to_le_bytes());
        out.extend_from_slice(&9u32.to_le_bytes()); // n_types
        for &sz in &canon_sizes {
            out.extend_from_slice(&sz.to_le_bytes());
        }
        out.extend_from_slice(&data_len.to_le_bytes());
        // Row-major: stage -> type -> canonical index.
        for stage_row in &self.w {
            for vec in stage_row {
                for &x in vec {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
        out
    }

    /// Deserializes an explicit little-endian `LGW1` blob and validates the
    /// canonical sizes against the vendored pack tables
    /// (`EVAL_PACKED_SIZE`).
    ///
    /// # Errors
    ///
    /// Returns [`WeightsError::Format`] on a structural mismatch (bad
    /// magic/version, wrong stage/type count, canonical sizes that disagree
    /// with the vendored Edax pack tables, truncation, or trailing bytes).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, WeightsError> {
        let err = |m: &str| WeightsError::Format(m.to_string());
        if bytes.len() < LGW1_HEADER_BYTES {
            return Err(err("truncated header"));
        }
        let rd_u32 = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        if rd_u32(0) != WEIGHTS_MAGIC {
            return Err(err("bad magic (expected LGW1)"));
        }
        if rd_u32(4) != 1 {
            return Err(err("unsupported weights version"));
        }
        if rd_u32(8) as usize != N_STAGES || rd_u32(12) != 9 {
            return Err(err("stage/type count mismatch"));
        }
        let mut canon_sizes = [0u32; 9];
        for (t, slot) in canon_sizes.iter_mut().enumerate() {
            *slot = rd_u32(16 + 4 * t);
        }
        let tables = PackTables::build();
        for (t, &ty) in WEIGHT_TYPE_ORDER.iter().enumerate() {
            if canon_sizes[t] != tables.n_canonical(ty) {
                return Err(err("canonical size mismatch vs vendored pack tables"));
            }
        }
        let data_len = u64::from_le_bytes(bytes[52..60].try_into().unwrap()) as usize;
        let expected: usize = N_STAGES
            * WEIGHT_TYPE_ORDER
                .iter()
                .map(|&ty| tables.n_canonical(ty) as usize)
                .sum::<usize>();
        if data_len != expected {
            return Err(err("weight data length mismatch"));
        }
        let want_bytes = LGW1_HEADER_BYTES + 4 * data_len;
        if bytes.len() != want_bytes {
            return Err(err("blob size does not match declared data length"));
        }
        let mut w: [[Vec<i32>; 9]; N_STAGES] =
            std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
        let mut off = LGW1_HEADER_BYTES;
        for stage_row in &mut w {
            for (t, vec) in stage_row.iter_mut().enumerate() {
                let n = canon_sizes[t] as usize;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(i32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
                    off += 4;
                }
                *vec = v;
            }
        }
        Ok(Self { w, tables })
    }

    /// Loads an `LGW1` weights file from disk.
    ///
    /// # Errors
    ///
    /// Returns [`WeightsError::Io`] if the file cannot be read, or
    /// [`WeightsError::Format`] if its contents are not a valid `LGW1`
    /// blob (see [`from_bytes`](Self::from_bytes)).
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, WeightsError> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Writes this weight set to `path` as an `LGW1` file.
    ///
    /// # Errors
    ///
    /// Returns [`WeightsError::Io`] if the file cannot be written.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), WeightsError> {
        std::fs::write(path, self.to_bytes())?;
        Ok(())
    }
}

impl Default for EvalWeights {
    fn default() -> Self {
        Self::zeros()
    }
}

/// Phase-1 placeholder name kept for back-compat. Prefer [`EvalWeights`].
pub type Weights = EvalWeights;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern::CONST_FEATURE;

    #[test]
    fn pack_sizes_match_edax_packed_size() {
        let order = [
            PatternType::C9,
            PatternType::C10,
            PatternType::S10,
            PatternType::S10,
            PatternType::S8,
            PatternType::S8,
            PatternType::S8,
            PatternType::S8,
            PatternType::S7,
            PatternType::S6,
            PatternType::S5,
            PatternType::S4,
            PatternType::Const,
        ];
        let tables = PackTables::build();
        for (i, ty) in order.iter().enumerate() {
            assert_eq!(
                tables.n_canonical(*ty),
                EVAL_PACKED_SIZE[i],
                "type {ty:?} at slot {i} canonical count != EVAL_PACKED_SIZE"
            );
        }
    }

    #[test]
    fn unpack_weight_sharing_rule() {
        // j < i  =>  i shares j's canonical slot (Edax eval.c:555-556).
        for ty in [
            PatternType::C9,
            PatternType::C10,
            PatternType::S10,
            PatternType::S8,
            PatternType::S7,
            PatternType::S6,
            PatternType::S5,
            PatternType::S4,
        ] {
            let t = PackTable::build(ty);
            let sym = sym_for(ty);
            let k = ty.k();
            let size = POW3[k];
            let mut next = 0u32;
            for i in 0..size {
                let j = player_feature(sym, k, i);
                if j < i {
                    assert_eq!(
                        t.player[i as usize], t.player[j as usize],
                        "{ty:?}: key {i} must share canon of {j}"
                    );
                } else {
                    assert_eq!(
                        t.player[i as usize], next,
                        "{ty:?}: key {i} must get fresh canon {next}"
                    );
                    next += 1;
                }
            }
            assert_eq!(next, t.n_canonical);
        }
    }

    #[test]
    fn player_feature_is_identity_under_trivial_sym() {
        // sym = [0,1,2,...] would be identity; sym_S10 is the full reversal,
        // so applying it twice is the identity.
        let k = 10;
        for l in [0u32, 1, 2, 59048, 12345, 3 * 3 + 2] {
            let once = player_feature(&SYM_S10, k, l);
            let twice = player_feature(&SYM_S10, k, once);
            assert_eq!(twice, l, "double reversal must be identity for {l}");
        }
    }

    #[test]
    fn opponent_feature_is_involution() {
        for k in [4usize, 7, 9, 10] {
            let size = POW3[k];
            for l in (0..size).step_by(((size / 97).max(1)) as usize) {
                assert_eq!(opponent_feature(opponent_feature(l, k), k), l);
            }
        }
    }

    #[test]
    fn zeros_total_weight_count() {
        let w = EvalWeights::zeros();
        // One shared canonical vector per *distinct* type (9), not per
        // accumulate-order group: S10 and S8 each appear once in storage
        // even though EVAL_PACKED_SIZE lists them 2× / 4× (their feature
        // instances all index the same shared vector, exactly like Edax
        // reuses EVAL_S10 / EVAL_S8). So sum the 9 distinct-type sizes.
        let per_stage: u32 = WEIGHT_TYPE_ORDER.iter().map(|&ty| w.n_canonical(ty)).sum();
        assert_eq!(per_stage, 74755);
        assert_eq!(w.total_weights(), (per_stage as usize) * N_STAGES);
        // Every entry zero.
        for ty in WEIGHT_TYPE_ORDER {
            for s in 0..N_STAGES {
                assert!(w.weights(s, ty).iter().all(|&x| x == 0));
            }
        }
    }

    #[test]
    fn explicit_le_roundtrip_bit_exact() {
        let mut w = EvalWeights::zeros();
        // Poke deterministic non-trivial values across stages/types.
        for s in 0..N_STAGES {
            for (t, &ty) in WEIGHT_TYPE_ORDER.iter().enumerate() {
                let v = w.weights_mut(s, ty);
                for (i, x) in v.iter_mut().enumerate() {
                    *x = ((s as i32) * 31 + (t as i32) * 7 + i as i32) % 257 - 128;
                }
            }
        }
        let bytes = w.to_bytes();
        // Explicit LE layout: fixed header size + 4 * data_len bytes.
        assert_eq!(bytes.len(), LGW1_HEADER_BYTES + 4 * w.total_weights());
        // Header is hand-checkable little-endian.
        assert_eq!(&bytes[0..4], &WEIGHTS_MAGIC.to_le_bytes());
        assert_eq!(&bytes[4..8], &1u32.to_le_bytes());
        assert_eq!(&bytes[8..12], &(N_STAGES as u32).to_le_bytes());
        assert_eq!(&bytes[12..16], &9u32.to_le_bytes());
        assert_eq!(
            u64::from_le_bytes(bytes[52..60].try_into().unwrap()) as usize,
            w.total_weights()
        );
        let back = EvalWeights::from_bytes(&bytes).expect("deserialize");
        assert_eq!(w, back, "EvalWeights round-trip must be bit-exact");
        // Const type has exactly one canonical slot.
        assert_eq!(
            back.n_canonical(crate::pattern::FEATURES[CONST_FEATURE].ty),
            1
        );
    }

    #[test]
    fn from_bytes_rejects_bad_magic() {
        let mut bytes = EvalWeights::zeros().to_bytes();
        bytes[0] ^= 0xFF;
        assert!(EvalWeights::from_bytes(&bytes).is_err());
    }

    #[test]
    fn from_bytes_rejects_truncation_and_trailing() {
        let bytes = EvalWeights::zeros().to_bytes();
        assert!(EvalWeights::from_bytes(&bytes[..bytes.len() - 4]).is_err());
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(EvalWeights::from_bytes(&longer).is_err());
    }

    #[test]
    fn save_load_roundtrip() {
        let mut w = EvalWeights::zeros();
        for s in 0..N_STAGES {
            w.weights_mut(s, PatternType::S4)[0] = (s as i32) - 6;
        }
        let dir = std::env::temp_dir();
        let path = dir.join(format!("lgw1_test_{}.lgw1", std::process::id()));
        w.save(&path).expect("save");
        let back = EvalWeights::load(&path).expect("load");
        let _ = std::fs::remove_file(&path);
        assert_eq!(w, back);
    }

    #[test]
    fn lookup_respects_canonical_sharing() {
        // Build a weights set where a known shared pair sees the same value.
        let mut w = EvalWeights::zeros();
        let ty = PatternType::S4;
        for (i, slot) in w.weights_mut(0, ty).iter_mut().enumerate() {
            *slot = i as i32;
        }
        let table = w.tables().get(ty).clone();
        let k = ty.k();
        let size = POW3[k];
        for i in 0..size {
            let j = player_feature(sym_for(ty), k, i);
            if j < i {
                assert_eq!(
                    w.lookup(0, ty, i),
                    w.lookup(0, ty, j),
                    "shared keys must look up equal weights"
                );
                assert_eq!(table.player[i as usize], table.player[j as usize]);
            }
        }
        // Reference the cloned table so the clone is exercised.
        assert_eq!(table.n_canonical, w.n_canonical(ty));
    }
}
