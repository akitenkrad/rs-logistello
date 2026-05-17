//! GOLD cross-language GLEM interop test (Phase 7 — **non-negotiable**).
//!
//! Proves the `GLM1` binary contract, the base-literal extractor, and the
//! conjunction evaluation are consistent across Rust and Python. It MUST NOT
//! be weakened. If it fails, the cross-language GLEM contract is broken
//! (diagnose; do not relax the assertions).
//!
//! ## Fixtures
//!
//! Generated once by `logistello_tools._gen_glem_fixture` (run-once helper),
//! committed under `tests/data/`:
//!
//! - `glem_fixture.glm1` — a `GLM1` model written **by Python** from the
//!   documented deterministic feature set + weight construction
//!   (`features = [(0,),(191,),(200,),(5,191),(0,196),(12,203)]`,
//!   `w[s][f] = ((s*37 + f*53 + 11) mod 401) - 200`, spec `cell64,corner`).
//! - `glem_fixture_positions.json` — for the deterministic GLX1 corpus
//!   (`glem-extract --base-features cell64,corner --source selfplay
//!   --games 10 --seed 4242`, extracted by the Rust CLI and read back by
//!   Python), Python's own `GlemEval`-equivalent prediction
//!   `round(sum satisfied w / 128)` (Edax bias + clamp) per position.
//!
//! ## What is asserted
//!
//! 1. **Byte identity (Python-write == Rust-write).** Rust builds the same
//!    deterministic model, serializes via `GlemModel::to_bytes`, and asserts
//!    the bytes equal the committed Python-written `glem_fixture.glm1`.
//! 2. **Round-trip through Python's file.** `GlemModel::load(...)` then
//!    `.to_bytes()` reproduces the file bit-for-bit.
//! 3. **Cross-language prediction consistency.** Loading Python's
//!    `glem_fixture.glm1` into `GlemEval` and evaluating exactly the
//!    positions Python scored yields, for every position, the *same* value
//!    as Python's own prediction (base-literal extractor + conjunction eval
//!    + GLM1 contract identical cross-language).

use std::path::PathBuf;

use logistello_cli::glem_extract::extract_selfplay_with_states;
use logistello_eval::{BaseFeatureSpec, Conjunction, GlemEval, GlemModel, LeafEvaluator, N_STAGES};

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("data")
}

/// The documented deterministic construction, reproduced in Rust. Must stay
/// in lock-step with `_gen_glem_fixture`.
fn fixture_weight(stage: i64, f: i64) -> i32 {
    (((stage * 37 + f * 53 + 11) % 401) - 200) as i32
}

fn fixture_features() -> Vec<Conjunction> {
    vec![
        Conjunction::new(vec![0]),
        Conjunction::new(vec![191]),
        Conjunction::new(vec![200]),
        Conjunction::new(vec![5, 191]),
        Conjunction::new(vec![0, 196]),
        Conjunction::new(vec![12, 203]),
    ]
}

fn build_fixture_model() -> GlemModel {
    let spec = BaseFeatureSpec::parse("cell64,corner").expect("spec");
    let feats = fixture_features();
    let mut w = vec![vec![0i32; feats.len()]; N_STAGES];
    for (s, row) in w.iter_mut().enumerate() {
        for (f, x) in row.iter_mut().enumerate() {
            *x = fixture_weight(s as i64, f as i64);
        }
    }
    GlemModel::new(spec, feats, w)
}

#[test]
fn gold_glm1_bytes_identical_python_vs_rust() {
    let path = data_dir().join("glem_fixture.glm1");
    let py_bytes = std::fs::read(&path).expect(
        "glem_fixture.glm1 must exist (regenerate with \
         `uv run python -m logistello_tools._gen_glem_fixture`)",
    );
    let rust_bytes = build_fixture_model().to_bytes();
    assert_eq!(
        rust_bytes.len(),
        py_bytes.len(),
        "GLM1 length differs: Rust {} vs Python {}",
        rust_bytes.len(),
        py_bytes.len()
    );
    assert!(
        rust_bytes == py_bytes,
        "GOLD: Rust-serialized GLM1 is NOT byte-identical to the \
         Python-written glem_fixture.glm1 - cross-language contract broken"
    );

    // (2) Round-trip through Python's own file.
    let loaded = GlemModel::load(&path).expect("load Python glem_fixture.glm1");
    let reser = loaded.to_bytes();
    assert!(
        reser == py_bytes,
        "GOLD: load(glem_fixture.glm1) -> to_bytes() must reproduce the \
         file bit-for-bit"
    );
    assert_eq!(loaded, build_fixture_model());
}

#[test]
fn gold_glem_eval_matches_python_predictions() {
    let dir = data_dir();
    let model = GlemModel::load(dir.join("glem_fixture.glm1")).expect("load glm1");
    let scorer = GlemEval::new(model);

    let meta: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("glem_fixture_positions.json"))
            .expect("glem_fixture_positions.json must exist"),
    )
    .expect("valid JSON");

    let base_features = meta["extract"]["base_features"].as_str().unwrap();
    let games = meta["extract"]["games"].as_u64().unwrap() as usize;
    let seed = meta["extract"]["seed"].as_u64().unwrap();
    let spec = BaseFeatureSpec::parse(base_features).expect("spec");
    let (_recs, states) = extract_selfplay_with_states(&spec, games, seed, None)
        .expect("re-extract deterministic corpus");

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
            "GOLD: position {i}: Rust GlemEval = {got} but Python \
             predicted {want} - base-literal extractor / GLM1 contract \
             inconsistent across languages"
        );
    }
}
