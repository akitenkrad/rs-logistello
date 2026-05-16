//! GOLD cross-language LGW1 interop test (Phase 4b — **non-negotiable**).
//!
//! This test proves the `LGW1` binary contract and the B4 canonical mapping
//! are consistent across Rust and Python. It MUST NOT be weakened. If it
//! fails, the cross-language weight contract is broken (diagnose; do not
//! relax the assertions).
//!
//! ## Fixtures
//!
//! Generated once by the Python helper `logistello_tools._gen_fixture`
//! (see `tools/src/logistello_tools/_gen_fixture.py`), committed under
//! `tests/data/`:
//!
//! - `fixture.lgw1` — an `LGW1` file written **by Python** from the
//!   documented deterministic weight construction
//!   `w[s][t][c] = ((s*1009 + t*131 + c*31 + 7) mod 521) - 260`.
//! - `fixture_positions.json` — for the deterministic self-play corpus
//!   (`source=selfplay games=12 seed=12345`, extracted by the Rust CLI and
//!   read back by Python), Python's own model prediction
//!   `round(sum(canonical_w) / 128)` (Edax bias rounding + +/-63 clamp) per
//!   position, in record order.
//!
//! ## What is asserted
//!
//! 1. **Byte identity (Python-write == Rust-write).** Rust builds the *same*
//!    deterministic weights, serializes via `EvalWeights::to_bytes`, and
//!    asserts the bytes equal the committed Python-written `fixture.lgw1`
//!    exactly.
//! 2. **Round-trip through Python's file.** `EvalWeights::load(fixture.lgw1)`
//!    then `.to_bytes()` reproduces the file bit-for-bit.
//! 3. **Canonical-mapping consistency.** Loading Python's `fixture.lgw1`
//!    into `PatternEval` and evaluating exactly the positions Python scored
//!    yields, for every position, the *same* value as Python's own
//!    prediction (B1 key extraction + B4 pack/unpack identical cross-lang).

use std::path::PathBuf;

use logistello_cli::extract::extract_selfplay_with_states;
use logistello_eval::weights::WEIGHT_TYPE_ORDER;
use logistello_eval::{EvalWeights, LeafEvaluator, N_STAGES, PatternEval};

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("data")
}

/// The documented deterministic construction, reproduced in Rust. Must stay
/// in lock-step with `_gen_fixture.canon_weight`.
fn canon_weight(stage: i64, type_slot: i64, canon: i64) -> i32 {
    (((stage * 1009 + type_slot * 131 + canon * 31 + 7) % 521) - 260) as i32
}

fn build_fixture_weights() -> EvalWeights {
    let mut w = EvalWeights::zeros();
    for s in 0..N_STAGES {
        for (t, &ty) in WEIGHT_TYPE_ORDER.iter().enumerate() {
            let v = w.weights_mut(s, ty);
            for (c, x) in v.iter_mut().enumerate() {
                *x = canon_weight(s as i64, t as i64, c as i64);
            }
        }
    }
    w
}

#[test]
fn gold_lgw1_bytes_identical_python_vs_rust() {
    let path = data_dir().join("fixture.lgw1");
    let py_bytes = std::fs::read(&path).expect(
        "fixture.lgw1 must exist (regenerate with \
         `uv run python -m logistello_tools._gen_fixture`)",
    );
    let rust_bytes = build_fixture_weights().to_bytes();
    assert_eq!(
        rust_bytes.len(),
        py_bytes.len(),
        "LGW1 length differs: Rust {} vs Python {}",
        rust_bytes.len(),
        py_bytes.len()
    );
    assert!(
        rust_bytes == py_bytes,
        "GOLD: Rust-serialized LGW1 is NOT byte-identical to the \
         Python-written fixture.lgw1 - cross-language contract broken"
    );

    // (2) Round-trip through Python's own file.
    let loaded = EvalWeights::load(&path).expect("load Python fixture.lgw1");
    let reser = loaded.to_bytes();
    assert!(
        reser == py_bytes,
        "GOLD: load(fixture.lgw1) -> to_bytes() must reproduce the file \
         bit-for-bit"
    );
    assert_eq!(loaded, build_fixture_weights());
}

#[test]
fn gold_pattern_eval_matches_python_predictions() {
    let dir = data_dir();
    let weights = EvalWeights::load(dir.join("fixture.lgw1")).expect("load fixture.lgw1");
    let scorer = PatternEval::new(weights);

    let meta: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("fixture_positions.json"))
            .expect("fixture_positions.json must exist"),
    )
    .expect("valid JSON");

    let games = meta["extract"]["games"].as_u64().unwrap() as usize;
    let seed = meta["extract"]["seed"].as_u64().unwrap();
    let (_recs, states) =
        extract_selfplay_with_states(games, seed, None).expect("re-extract deterministic corpus");

    let preds = meta["predictions"].as_array().expect("predictions array");
    assert_eq!(
        states.len(),
        preds.len(),
        "position count drift: Rust re-extract {} vs Python fixture {}",
        states.len(),
        preds.len()
    );
    assert!(!states.is_empty(), "fixture corpus must be non-empty");

    for (i, (state, pred)) in states.iter().zip(preds).enumerate() {
        let want = pred.as_i64().unwrap() as i32;
        let got = scorer.eval(state);
        assert_eq!(
            got, want,
            "GOLD: position {i}: Rust PatternEval = {got} but Python \
             predicted {want} - B4 canonical mapping / LGW1 contract \
             inconsistent across languages"
        );
    }
}
