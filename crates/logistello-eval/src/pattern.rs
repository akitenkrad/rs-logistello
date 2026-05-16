//! Pattern encoding (Edax 4.6 準拠) — design doc §4.4 B1.
//!
//! # Provenance
//!
//! Every square-index array below is vendored **verbatim** from
//! `edax-reversi` tag **v4.6**, commit
//! `713a434f13b3d15fb69c61b9ba3641aed82c496c`, file `src/eval.c:39-100`
//! (`static const FeatureToCoordinate EVAL_F2X[]`). Edax's square enum
//! (`src/const.h:34-44`) is `A1=0, B1=1, …, H1=7, A2=8, …, H8=63`, i.e.
//! `idx = file + 8*rank` with `A1 = 0`. This is identical to
//! `othello_core::Coord::to_bit_index_8x8 = row*8 + col`, so the indices are
//! reused unchanged.
//!
//! `EVAL_F2X[]` has 48 entries: indices `0..=45` are the **46 pattern
//! instances**, index 46 is the **constant / material term** (`n_square = 0`,
//! design doc B1 "feature 46"), and Edax's index 47 is a second `{0,{NOMOVE}}`
//! sentinel that the accumulate loop computes but never sums
//! (`eval.c:1085` only adds `… + w4[f[46]]`). We therefore model exactly the
//! **47 features** (`EVAL_N_FEATURE = 47`) the accumulator uses: features
//! `0..=45` are patterns, feature 46 is the constant.
//!
//! # Cell encoding (design doc B1 / Edax `board_get_square_color`,
//! `board.c:1483`)
//!
//! `0 = side-to-move, 1 = opponent, 2 = empty`. The key is built **MSB-first**
//! over the feature's `x[]` list: `key = key * 3 + color` (Edax
//! `eval.c:796`). Hence `0 <= key < 3^k`.

use othello_core::{Board, Color, Coord, GameState};

/// Number of evaluation features = 46 pattern instances + 1 constant term
/// (Edax `EVAL_N_FEATURE`, `eval.c:481`). Design doc B1.
pub const N_FEATURES: usize = 47;

/// Index of the constant / material term (design doc B1 "feature 46").
pub const CONST_FEATURE: usize = 46;

/// Pattern *type* (symmetry class). Drives which Edax pack/unpack table a
/// feature instance shares its weights through (design doc B4). The constant
/// term has its own degenerate type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PatternType {
    /// 3×3 corner, 9 squares, `sym_C9` (design doc B1 row C9).
    C9,
    /// Corner + 2X, 10 squares, `sym_C10` (B1 row C10).
    C10,
    /// Edge+2X / 2×5 corner, 10 squares, `sym_S10` (B1 rows S10).
    S10,
    /// Line / main-diagonal, 8 squares, `sym_S10 + 2` (B1 rows S8 + d8).
    S8,
    /// Diagonal-7, 7 squares, `sym_S10 + 3` (B1 row d7).
    S7,
    /// Diagonal-6, 6 squares, `sym_S10 + 4` (B1 row d6).
    S6,
    /// Diagonal-5, 5 squares, `sym_S10 + 5` (B1 row d5).
    S5,
    /// Diagonal-4, 4 squares, `sym_S10 + 6` (B1 row d4).
    S4,
    /// Constant / material term (feature 46), no squares.
    Const,
}

impl PatternType {
    /// Number of board squares (`k`) the pattern reads; `0` for the constant.
    #[must_use]
    pub const fn k(self) -> usize {
        match self {
            PatternType::C9 => 9,
            PatternType::C10 | PatternType::S10 => 10,
            PatternType::S8 => 8,
            PatternType::S7 => 7,
            PatternType::S6 => 6,
            PatternType::S5 => 5,
            PatternType::S4 => 4,
            PatternType::Const => 0,
        }
    }

    /// Raw (pre-pack) key space size `3^k`; `1` for the constant.
    #[must_use]
    pub const fn raw_size(self) -> u32 {
        const POW3: [u32; 11] = [1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049];
        POW3[self.k()]
    }
}

/// One feature instance: its pattern type and the verbatim Edax square-index
/// list (`EVAL_F2X[i].x[0..n_square]`), evaluated MSB-first.
#[derive(Debug, Clone, Copy)]
pub struct FeatureDef {
    /// Symmetry class / pack table this instance shares weights through.
    pub ty: PatternType,
    /// Edax `x[]` square indices (`idx = file + 8*rank`, A1 = 0). Empty for
    /// the constant term.
    pub squares: &'static [u8],
}

// ---------------------------------------------------------------------------
// Vendored EVAL_F2X[] square-index arrays.
//
// Source: edax-reversi v4.6, commit 713a434f13b3d15fb69c61b9ba3641aed82c496c,
// src/eval.c:40-98. Square names converted via const.h:34-44 enum
// (A1=0 … H8=63, idx = file + 8*rank). Verbatim: order within each array is
// preserved exactly so the MSB-first key matches Edax `eval_set`.
// ---------------------------------------------------------------------------

/// 3×3 corner (C9), 4 instances — `eval.c:40-43`.
const C9_0: [u8; 9] = [0, 1, 8, 9, 2, 16, 10, 17, 18]; // {A1,B1,A2,B2,C1,A3,C2,B3,C3}
const C9_1: [u8; 9] = [7, 6, 15, 14, 5, 23, 13, 22, 21]; // {H1,G1,H2,G2,F1,H3,F2,G3,F3}
const C9_2: [u8; 9] = [56, 48, 57, 49, 40, 58, 41, 50, 42]; // {A8,A7,B8,B7,A6,C8,B6,C7,C6}
const C9_3: [u8; 9] = [63, 55, 62, 54, 47, 61, 46, 53, 45]; // {H8,H7,G8,G7,H6,F8,G6,F7,F6}

/// Corner + 2X (C10), 4 instances — `eval.c:45-48`.
const C10_0: [u8; 10] = [32, 24, 16, 8, 0, 9, 1, 2, 3, 4]; // {A5,A4,A3,A2,A1,B2,B1,C1,D1,E1}
const C10_1: [u8; 10] = [39, 31, 23, 15, 7, 14, 6, 5, 4, 3]; // {H5,H4,H3,H2,H1,G2,G1,F1,E1,D1}
const C10_2: [u8; 10] = [24, 32, 40, 48, 56, 49, 57, 58, 59, 60]; // {A4,A5,A6,A7,A8,B7,B8,C8,D8,E8}
const C10_3: [u8; 10] = [31, 39, 47, 55, 63, 54, 62, 61, 60, 59]; // {H4,H5,H6,H7,H8,G7,G8,F8,E8,D8}

/// Edge + 2X (S10), 4 instances — `eval.c:50-53`.
const S10_E0: [u8; 10] = [9, 0, 1, 2, 3, 4, 5, 6, 7, 14]; // {B2,A1,B1,C1,D1,E1,F1,G1,H1,G2}
const S10_E1: [u8; 10] = [49, 56, 57, 58, 59, 60, 61, 62, 63, 54]; // {B7,A8,B8,C8,D8,E8,F8,G8,H8,G7}
const S10_E2: [u8; 10] = [9, 0, 8, 16, 24, 32, 40, 48, 56, 49]; // {B2,A1,A2,A3,A4,A5,A6,A7,A8,B7}
const S10_E3: [u8; 10] = [14, 7, 15, 23, 31, 39, 47, 55, 63, 54]; // {G2,H1,H2,H3,H4,H5,H6,H7,H8,G7}

/// 2×5 corner (S10), 4 instances — `eval.c:55-58`.
const S10_C0: [u8; 10] = [0, 2, 3, 10, 11, 12, 13, 4, 5, 7]; // {A1,C1,D1,C2,D2,E2,F2,E1,F1,H1}
const S10_C1: [u8; 10] = [56, 58, 59, 50, 51, 52, 53, 60, 61, 63]; // {A8,C8,D8,C7,D7,E7,F7,E8,F8,H8}
const S10_C2: [u8; 10] = [0, 16, 24, 17, 25, 33, 41, 32, 40, 56]; // {A1,A3,A4,B3,B4,B5,B6,A5,A6,A8}
const S10_C3: [u8; 10] = [7, 23, 31, 22, 30, 38, 46, 39, 47, 63]; // {H1,H3,H4,G3,G4,G5,G6,H5,H6,H8}

/// Line-2 (S8), 4 instances — `eval.c:60-63`.
const S8_L2_0: [u8; 8] = [8, 9, 10, 11, 12, 13, 14, 15]; // {A2,B2,C2,D2,E2,F2,G2,H2}
const S8_L2_1: [u8; 8] = [48, 49, 50, 51, 52, 53, 54, 55]; // {A7,B7,C7,D7,E7,F7,G7,H7}
const S8_L2_2: [u8; 8] = [1, 9, 17, 25, 33, 41, 49, 57]; // {B1,B2,B3,B4,B5,B6,B7,B8}
const S8_L2_3: [u8; 8] = [6, 14, 22, 30, 38, 46, 54, 62]; // {G1,G2,G3,G4,G5,G6,G7,G8}

/// Line-3 (S8), 4 instances — `eval.c:65-68`.
const S8_L3_0: [u8; 8] = [16, 17, 18, 19, 20, 21, 22, 23]; // {A3,B3,C3,D3,E3,F3,G3,H3}
const S8_L3_1: [u8; 8] = [40, 41, 42, 43, 44, 45, 46, 47]; // {A6,B6,C6,D6,E6,F6,G6,H6}
const S8_L3_2: [u8; 8] = [2, 10, 18, 26, 34, 42, 50, 58]; // {C1,C2,C3,C4,C5,C6,C7,C8}
const S8_L3_3: [u8; 8] = [5, 13, 21, 29, 37, 45, 53, 61]; // {F1,F2,F3,F4,F5,F6,F7,F8}

/// Line-4 (S8), 4 instances — `eval.c:70-73`.
const S8_L4_0: [u8; 8] = [24, 25, 26, 27, 28, 29, 30, 31]; // {A4,B4,C4,D4,E4,F4,G4,H4}
const S8_L4_1: [u8; 8] = [32, 33, 34, 35, 36, 37, 38, 39]; // {A5,B5,C5,D5,E5,F5,G5,H5}
const S8_L4_2: [u8; 8] = [3, 11, 19, 27, 35, 43, 51, 59]; // {D1,D2,D3,D4,D5,D6,D7,D8}
const S8_L4_3: [u8; 8] = [4, 12, 20, 28, 36, 44, 52, 60]; // {E1,E2,E3,E4,E5,E6,E7,E8}

/// Main diagonal d8 (S8), 2 instances — `eval.c:75-76`.
const S8_D8_0: [u8; 8] = [0, 9, 18, 27, 36, 45, 54, 63]; // {A1,B2,C3,D4,E5,F6,G7,H8}
const S8_D8_1: [u8; 8] = [56, 49, 42, 35, 28, 21, 14, 7]; // {A8,B7,C6,D5,E4,F3,G2,H1}

/// Diagonal d7 (S7), 4 instances — `eval.c:78-81`.
const S7_0: [u8; 7] = [1, 10, 19, 28, 37, 46, 55]; // {B1,C2,D3,E4,F5,G6,H7}
const S7_1: [u8; 7] = [15, 22, 29, 36, 43, 50, 57]; // {H2,G3,F4,E5,D6,C7,B8}
const S7_2: [u8; 7] = [8, 17, 26, 35, 44, 53, 62]; // {A2,B3,C4,D5,E6,F7,G8}
const S7_3: [u8; 7] = [6, 13, 20, 27, 34, 41, 48]; // {G1,F2,E3,D4,C5,B6,A7}

/// Diagonal d6 (S6), 4 instances — `eval.c:83-86`.
const S6_0: [u8; 6] = [2, 11, 20, 29, 38, 47]; // {C1,D2,E3,F4,G5,H6}
const S6_1: [u8; 6] = [16, 25, 34, 43, 52, 61]; // {A3,B4,C5,D6,E7,F8}
const S6_2: [u8; 6] = [5, 12, 19, 26, 33, 40]; // {F1,E2,D3,C4,B5,A6}
const S6_3: [u8; 6] = [23, 30, 37, 44, 51, 58]; // {H3,G4,F5,E6,D7,C8}

/// Diagonal d5 (S5), 4 instances — `eval.c:88-91`.
const S5_0: [u8; 5] = [3, 12, 21, 30, 39]; // {D1,E2,F3,G4,H5}
const S5_1: [u8; 5] = [24, 33, 42, 51, 60]; // {A4,B5,C6,D7,E8}
const S5_2: [u8; 5] = [4, 11, 18, 25, 32]; // {E1,D2,C3,B4,A5}
const S5_3: [u8; 5] = [31, 38, 45, 52, 59]; // {H4,G5,F6,E7,D8}

/// Diagonal d4 (S4), 4 instances — `eval.c:93-96`.
const S4_0: [u8; 4] = [3, 10, 17, 24]; // {D1,C2,B3,A4}
const S4_1: [u8; 4] = [32, 41, 50, 59]; // {A5,B6,C7,D8}
const S4_2: [u8; 4] = [4, 13, 22, 31]; // {E1,F2,G3,H4}
const S4_3: [u8; 4] = [39, 46, 53, 60]; // {H5,G6,F7,E8}

/// The 47 feature definitions in Edax accumulate order (`eval.c:1077-1085`):
/// C9×4, C10×4, S10(edge)×4, S10(2×5)×4, S8(line2)×4, S8(line3)×4,
/// S8(line4)×4, S8(d8)×2, S7×4, S6×4, S5×4, S4×4, then the constant term.
pub const FEATURES: [FeatureDef; N_FEATURES] = [
    FeatureDef {
        ty: PatternType::C9,
        squares: &C9_0,
    },
    FeatureDef {
        ty: PatternType::C9,
        squares: &C9_1,
    },
    FeatureDef {
        ty: PatternType::C9,
        squares: &C9_2,
    },
    FeatureDef {
        ty: PatternType::C9,
        squares: &C9_3,
    },
    FeatureDef {
        ty: PatternType::C10,
        squares: &C10_0,
    },
    FeatureDef {
        ty: PatternType::C10,
        squares: &C10_1,
    },
    FeatureDef {
        ty: PatternType::C10,
        squares: &C10_2,
    },
    FeatureDef {
        ty: PatternType::C10,
        squares: &C10_3,
    },
    FeatureDef {
        ty: PatternType::S10,
        squares: &S10_E0,
    },
    FeatureDef {
        ty: PatternType::S10,
        squares: &S10_E1,
    },
    FeatureDef {
        ty: PatternType::S10,
        squares: &S10_E2,
    },
    FeatureDef {
        ty: PatternType::S10,
        squares: &S10_E3,
    },
    FeatureDef {
        ty: PatternType::S10,
        squares: &S10_C0,
    },
    FeatureDef {
        ty: PatternType::S10,
        squares: &S10_C1,
    },
    FeatureDef {
        ty: PatternType::S10,
        squares: &S10_C2,
    },
    FeatureDef {
        ty: PatternType::S10,
        squares: &S10_C3,
    },
    FeatureDef {
        ty: PatternType::S8,
        squares: &S8_L2_0,
    },
    FeatureDef {
        ty: PatternType::S8,
        squares: &S8_L2_1,
    },
    FeatureDef {
        ty: PatternType::S8,
        squares: &S8_L2_2,
    },
    FeatureDef {
        ty: PatternType::S8,
        squares: &S8_L2_3,
    },
    FeatureDef {
        ty: PatternType::S8,
        squares: &S8_L3_0,
    },
    FeatureDef {
        ty: PatternType::S8,
        squares: &S8_L3_1,
    },
    FeatureDef {
        ty: PatternType::S8,
        squares: &S8_L3_2,
    },
    FeatureDef {
        ty: PatternType::S8,
        squares: &S8_L3_3,
    },
    FeatureDef {
        ty: PatternType::S8,
        squares: &S8_L4_0,
    },
    FeatureDef {
        ty: PatternType::S8,
        squares: &S8_L4_1,
    },
    FeatureDef {
        ty: PatternType::S8,
        squares: &S8_L4_2,
    },
    FeatureDef {
        ty: PatternType::S8,
        squares: &S8_L4_3,
    },
    FeatureDef {
        ty: PatternType::S8,
        squares: &S8_D8_0,
    },
    FeatureDef {
        ty: PatternType::S8,
        squares: &S8_D8_1,
    },
    FeatureDef {
        ty: PatternType::S7,
        squares: &S7_0,
    },
    FeatureDef {
        ty: PatternType::S7,
        squares: &S7_1,
    },
    FeatureDef {
        ty: PatternType::S7,
        squares: &S7_2,
    },
    FeatureDef {
        ty: PatternType::S7,
        squares: &S7_3,
    },
    FeatureDef {
        ty: PatternType::S6,
        squares: &S6_0,
    },
    FeatureDef {
        ty: PatternType::S6,
        squares: &S6_1,
    },
    FeatureDef {
        ty: PatternType::S6,
        squares: &S6_2,
    },
    FeatureDef {
        ty: PatternType::S6,
        squares: &S6_3,
    },
    FeatureDef {
        ty: PatternType::S5,
        squares: &S5_0,
    },
    FeatureDef {
        ty: PatternType::S5,
        squares: &S5_1,
    },
    FeatureDef {
        ty: PatternType::S5,
        squares: &S5_2,
    },
    FeatureDef {
        ty: PatternType::S5,
        squares: &S5_3,
    },
    FeatureDef {
        ty: PatternType::S4,
        squares: &S4_0,
    },
    FeatureDef {
        ty: PatternType::S4,
        squares: &S4_1,
    },
    FeatureDef {
        ty: PatternType::S4,
        squares: &S4_2,
    },
    FeatureDef {
        ty: PatternType::S4,
        squares: &S4_3,
    },
    FeatureDef {
        ty: PatternType::Const,
        squares: &[],
    },
];

/// Cell color code at `idx` from `side`'s point of view, per design doc B1 /
/// Edax `board_get_square_color` (`board.c:1483`): `0 = side-to-move,
/// 1 = opponent, 2 = empty`.
#[inline]
fn cell_code(board: &Board, idx: u8, side: Color) -> u32 {
    let coord = Coord::new(idx / 8, idx % 8);
    match board.cell(coord) {
        Some(c) if c == side => 0,
        Some(_) => 1,
        None => 2,
    }
}

/// Raw (pre-pack) key for one feature instance, MSB-first
/// (`key = key*3 + color`, Edax `eval.c:796`). The constant term has the
/// fixed key `0`.
#[inline]
#[must_use]
pub fn feature_key(board: &Board, def: &FeatureDef, side: Color) -> u32 {
    let mut key = 0u32;
    for &idx in def.squares {
        key = key * 3 + cell_code(board, idx, side);
    }
    key
}

/// Raw (pre-pack) keys for all 47 features from `side`'s point of view
/// (design doc B1). Feature 46 is the constant/material term and is always
/// `0`. Each key satisfies `0 <= key < 3^{k_feature}`
/// (`FEATURES[i].ty.raw_size()`).
#[must_use]
pub fn feature_keys(board: &Board, side: Color) -> [u32; N_FEATURES] {
    let mut out = [0u32; N_FEATURES];
    for (i, def) in FEATURES.iter().enumerate() {
        out[i] = feature_key(board, def, side);
    }
    out
}

/// Convenience: feature keys for a [`GameState`] from its side-to-move's
/// point of view.
#[must_use]
pub fn feature_keys_state(state: &GameState) -> [u32; N_FEATURES] {
    feature_keys(&state.board, state.side_to_move)
}

#[cfg(test)]
mod tests {
    use super::*;
    use othello_core::BoardSize;

    #[test]
    fn feature_count_is_47() {
        assert_eq!(FEATURES.len(), N_FEATURES);
        assert_eq!(N_FEATURES, 47);
        // 46 pattern instances + 1 constant.
        let patterns = FEATURES
            .iter()
            .filter(|f| f.ty != PatternType::Const)
            .count();
        assert_eq!(patterns, 46);
        assert_eq!(FEATURES[CONST_FEATURE].ty, PatternType::Const);
        assert!(FEATURES[CONST_FEATURE].squares.is_empty());
    }

    #[test]
    fn k_matches_square_count() {
        for (i, def) in FEATURES.iter().enumerate() {
            assert_eq!(
                def.squares.len(),
                def.ty.k(),
                "feature {i} square count != k"
            );
        }
    }

    #[test]
    fn b1_representative_rows() {
        // design doc B1: 3×3 corner C9 instance over {0,1,8,9,2,16,10,17,18}.
        assert_eq!(FEATURES[0].ty, PatternType::C9);
        assert_eq!(FEATURES[0].squares, &[0, 1, 8, 9, 2, 16, 10, 17, 18]);
        // design doc B1: main diag d8 = {0,9,18,27,36,45,54,63}.
        assert_eq!(FEATURES[28].squares, &[0, 9, 18, 27, 36, 45, 54, 63]);
        assert_eq!(FEATURES[28].ty, PatternType::S8);
        // B1 corner+2X C10 representative {32,24,16,8,0,9,1,2,3,4}.
        assert_eq!(FEATURES[4].ty, PatternType::C10);
        assert_eq!(FEATURES[4].squares, &[32, 24, 16, 8, 0, 9, 1, 2, 3, 4]);
        // B1 edge+2X S10 representative {9,0,1,2,3,4,5,6,7,14}.
        assert_eq!(FEATURES[8].ty, PatternType::S10);
        assert_eq!(FEATURES[8].squares, &[9, 0, 1, 2, 3, 4, 5, 6, 7, 14]);
        // B1 2×5 corner S10 representative {0,2,3,10,11,12,13,4,5,7}.
        assert_eq!(FEATURES[12].squares, &[0, 2, 3, 10, 11, 12, 13, 4, 5, 7]);
        // B1 d7 representative {1,10,19,28,37,46,55}.
        assert_eq!(FEATURES[30].ty, PatternType::S7);
        assert_eq!(FEATURES[30].squares, &[1, 10, 19, 28, 37, 46, 55]);
        // B1 d4 representative {3,10,17,24}.
        assert_eq!(FEATURES[42].ty, PatternType::S4);
        assert_eq!(FEATURES[42].squares, &[3, 10, 17, 24]);
    }

    #[test]
    fn keys_in_range_start_position() {
        let s = GameState::standard_8x8();
        let keys = feature_keys_state(&s);
        for (i, &k) in keys.iter().enumerate() {
            assert!(
                k < FEATURES[i].ty.raw_size(),
                "feature {i} key {k} >= 3^{}",
                FEATURES[i].ty.k()
            );
        }
        // Constant term is always 0.
        assert_eq!(keys[CONST_FEATURE], 0);
    }

    #[test]
    fn empty_board_all_twos() {
        // On a totally empty board every cell codes to 2, so each feature
        // key is the base-3 repunit (3^k - 1)/2 * 2 ... compute directly.
        let board = Board::new(BoardSize::STANDARD).unwrap();
        let keys = feature_keys(&board, Color::Black);
        for (i, def) in FEATURES.iter().enumerate() {
            let mut expect = 0u32;
            for _ in 0..def.ty.k() {
                expect = expect * 3 + 2;
            }
            assert_eq!(keys[i], expect, "feature {i} empty-board key");
        }
    }
}
