//! Workspace integration tests for Phase 10 — the `sweep` sensitivity
//! analysis (design doc §6 / §5.1 / §4.2).
//!
//! These tests *define correctness* and must never be weakened:
//!
//! - **Grid expansion** (`grid_*`): the §6 min/max/step + value-list +
//!   log-scale flags expand to the *exact* expected parameter sets, with
//!   inclusive bounds and well-defined float rounding; cross-product
//!   cardinality is correct.
//! - **Determinism** (`determinism_*`): a tiny sweep produces a
//!   byte-identical `metrics.csv` (deterministic columns) across two runs;
//!   seeds are derived explicitly from `--seed` + the trial index, never
//!   "current time".
//! - **Output contract** (`output_contract_*`): `sweep_config.json`
//!   round-trips the resolved spec; the `metrics.csv` header/columns match;
//!   `results/latest` points at the new timestamped dir; the dir layout is
//!   the §4.2 one.
//! - **Metric sanity** (`metric_*`): for a cheap swept parameter the
//!   recorded metric is plausible and monotone where §6 theory says so; a
//!   degenerate single-point grid works.
//! - **Composition** (`composition_*`): sweeping a parameter actually
//!   varies it in the measured run (a flag must not be parsed-then-ignored).
//!
//! No heavy sweeps: every condition here is tiny (≤ a few search calls).

use std::fs;
use std::path::Path;

use logistello_cli::sweep::{
    Axis, CSV_HEADER, Point, SweepConfigJson, SweepSpec, expand_log_range, expand_range, measure,
    parse_int_list, parse_pair_list, run_sweep,
};

// --------------------------------------------------------------------- //
//  Grid expansion                                                       //
// --------------------------------------------------------------------- //

#[test]
fn grid_probcut_t_inclusive_quarter_step() {
    // §6: ProbCut T 1.0..2.5 step 0.25 ⇒ 7 points incl. the 2.5 endpoint.
    let v = expand_range(1.0, 2.5, 0.25).unwrap();
    assert_eq!(v, vec![1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]);
}

#[test]
fn grid_drawishness_tenth_step_no_fp_noise() {
    // §6: drawishness 0.0..0.5 step 0.1 ⇒ exactly 6 clean points.
    let v = expand_range(0.0, 0.5, 0.1).unwrap();
    assert_eq!(v, vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5]);
}

#[test]
fn grid_glem_support_log_scale_endpoints_exact() {
    // §6: GLEM support 1e-4..1e-2, log10 step 0.25 ⇒ 9 points; the two
    // endpoints must be exactly 1e-4 and 1e-2.
    let v = expand_log_range(-4.0, -2.0, 0.25).unwrap();
    assert_eq!(v.len(), 9);
    assert!((v[0] - 1e-4).abs() < 1e-9);
    assert!((v[8] - 1e-2).abs() < 1e-8);
    for w in v.windows(2) {
        assert!(w[1] > w[0], "log grid strictly increasing");
    }
}

#[test]
fn grid_degenerate_single_point() {
    assert_eq!(expand_range(0.3, 0.3, 0.1).unwrap(), vec![0.3]);
    assert_eq!(expand_log_range(-3.0, -3.0, 0.25).unwrap().len(), 1);
}

#[test]
fn grid_value_lists_and_pairs() {
    assert_eq!(
        parse_int_list("1,5,10,13,20,30").unwrap(),
        vec![1, 5, 10, 13, 20, 30]
    );
    assert_eq!(
        parse_pair_list("1:5,3:7,5:9,3:9,5:11").unwrap(),
        vec![(1, 5), (3, 7), (5, 9), (3, 9), (5, 11)]
    );
    assert!(parse_int_list("").is_err());
    assert!(parse_pair_list("4:4").is_err());
}

#[test]
fn grid_cross_product_cardinality() {
    let s = SweepSpec {
        probcut_t: Some((1.0, 2.5, 0.25)), // 7 conditions
        runs: 5,
        seed: 42,
        ..Default::default()
    };
    let r = s.resolve().unwrap();
    let cfg = SweepConfigJson::from_resolved(&r);
    assert_eq!(cfg.n_conditions, 7);
    assert_eq!(cfg.n_trials, 35); // 7 × 5 seeds
}

#[test]
fn grid_resolve_requires_exactly_one_axis() {
    assert!(
        SweepSpec {
            runs: 3,
            seed: 1,
            ..Default::default()
        }
        .resolve()
        .is_err(),
        "zero axes ⇒ error (the no-arg path lists params & exits)"
    );
    assert!(
        SweepSpec {
            probcut_t: Some((1.0, 1.5, 0.5)),
            endgame_empties_values: Some("8,12".into()),
            runs: 3,
            seed: 1,
            ..Default::default()
        }
        .resolve()
        .is_err(),
        ">1 axis ⇒ error (§6 table is one row per parameter)"
    );
}

// --------------------------------------------------------------------- //
//  Output contract (§4.2)                                               //
// --------------------------------------------------------------------- //

/// Runs a tiny sweep into a unique temp `results/` root.
fn tiny_sweep(tag: &str) -> (std::path::PathBuf, Vec<logistello_cli::sweep::MetricRow>) {
    let root = std::env::temp_dir().join(format!(
        "logi_sweep_{tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = fs::remove_dir_all(&root);
    let spec = SweepSpec {
        endgame_empties_values: Some("8,12".into()),
        runs: 3,
        seed: 42,
        ..Default::default()
    };
    let resolved = spec.resolve().unwrap();
    let (dir, rows) = run_sweep(&resolved, &root).unwrap();
    (dir, rows)
}

#[test]
fn output_contract_layout_and_files() {
    let (dir, rows) = tiny_sweep("layout");
    // §4.2: results/<ts>/{sweep_config.json, metrics.csv}.
    let cfg_path = dir.join("sweep_config.json");
    let csv_path = dir.join("metrics.csv");
    assert!(cfg_path.is_file(), "sweep_config.json must exist");
    assert!(csv_path.is_file(), "metrics.csv must exist");

    // metrics.csv: header matches CSV_HEADER, 1 header + 2×3 = 6 data rows.
    let csv = fs::read_to_string(&csv_path).unwrap();
    let mut lines = csv.lines();
    assert_eq!(lines.next().unwrap(), CSV_HEADER);
    let data: Vec<&str> = lines.collect();
    assert_eq!(data.len(), 6, "2 conditions × 3 runs = 6 trial rows");
    assert_eq!(rows.len(), 6);
    for l in &data {
        assert_eq!(
            l.split(',').count(),
            CSV_HEADER.split(',').count(),
            "every row has the full fixed schema"
        );
    }

    // sweep_config.json round-trips the resolved spec.
    let v: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&cfg_path).unwrap()).unwrap();
    assert_eq!(v["command"], "sweep");
    assert_eq!(v["axis"], "endgame_empties");
    assert_eq!(v["metric"], "nodes");
    assert_eq!(v["values"], serde_json::json!(["8", "12"]));
    assert_eq!(v["runs"], 3);
    assert_eq!(v["n_conditions"], 2);
    assert_eq!(v["n_trials"], 6);

    let _ = fs::remove_dir_all(dir.parent().unwrap());
}

#[test]
fn output_contract_latest_symlink() {
    let (dir, _) = tiny_sweep("latest");
    let root = dir.parent().unwrap();
    let link = root.join("latest");
    #[cfg(unix)]
    {
        assert!(link.exists(), "results/latest must resolve");
        let resolved = fs::canonicalize(&link).unwrap();
        assert_eq!(
            resolved,
            fs::canonicalize(&dir).unwrap(),
            "results/latest points at the new ts dir"
        );
    }
    #[cfg(not(unix))]
    {
        let _ = link;
        assert!(root.join("latest.txt").exists());
    }
    let _ = fs::remove_dir_all(root);
}

// --------------------------------------------------------------------- //
//  Determinism                                                          //
// --------------------------------------------------------------------- //

#[test]
fn determinism_metrics_csv_byte_identical_across_runs() {
    // A tiny sweep (1 axis × 2 values × 2 seeds) twice ⇒ the **whole**
    // metrics.csv is byte-identical (it carries only deterministic
    // columns; wall-clock time/nps are deliberately NOT in the CSV).
    let make = |tag: &str| -> String {
        let root =
            std::env::temp_dir().join(format!("logi_sweep_det_{tag}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let spec = SweepSpec {
            probcut_t: Some((1.0, 1.5, 0.5)), // 2 values
            runs: 2,
            seed: 7,
            ..Default::default()
        };
        let r = spec.resolve().unwrap();
        let (dir, _) = run_sweep(&r, &root).unwrap();
        let csv = fs::read_to_string(dir.join("metrics.csv")).unwrap();
        let _ = fs::remove_dir_all(&root);
        csv
    };
    let a = make("a");
    let b = make("b");
    assert_eq!(
        a, b,
        "metrics.csv must be byte-identical across runs; per-(condition,\
         seed) measurement is reproducible (seeds derived from --seed + \
         trial index, never wall-clock)"
    );
    // Sanity: the byte-identical file really does have content.
    assert!(a.lines().count() == 5, "header + 2 values × 2 seeds");
}

#[test]
fn determinism_measure_is_pure() {
    for axis_pt in [
        (Axis::EndgameEmpties, Point::I(12)),
        (Axis::ProbcutT, Point::F(1.25)),
        (Axis::Drawishness, Point::F(0.2)),
        (Axis::GlemMaxOrder, Point::I(3)),
        (Axis::EvalStages, Point::I(5)),
    ] {
        let (axis, pt) = axis_pt;
        let x = measure(axis, pt, 13);
        let y = measure(axis, pt, 13);
        assert_eq!(x.nodes, y.nodes);
        assert_eq!(x.search_value, y.search_value);
        assert_eq!(x.dispatch, y.dispatch);
        assert_eq!(x.n_features, y.n_features);
        assert_eq!(x.book_positions, y.book_positions);
        assert_eq!(x.selfplay_score, y.selfplay_score);
        assert!((x.eval_abs_err - y.eval_abs_err).abs() < 1e-12);
    }
}

// --------------------------------------------------------------------- //
//  Metric sanity & composition (§6 "期待される主要な知見")               //
// --------------------------------------------------------------------- //

#[test]
fn metric_endgame_empties_monotone_and_varies() {
    // §6 / B8: a larger exact-endgame switch threshold makes the engine
    // enter the exact (full-width) endgame earlier, so for a fixed late
    // root the node count is non-decreasing in the threshold and *must*
    // actually change (guards against the flag being parsed-then-ignored).
    let sum = |e: i64| -> u64 {
        (0..6)
            .map(|s| measure(Axis::EndgameEmpties, Point::I(e), s).nodes)
            .sum()
    };
    let small = sum(8);
    let large = sum(24);
    assert!(
        large >= small,
        "monotone: nodes(24)={large} >= nodes(8)={small}"
    );
    assert!(
        large > small,
        "composition: the swept threshold must move the metric \
         ({large} > {small})"
    );
}

#[test]
fn metric_degenerate_single_point_grid_works() {
    let spec = SweepSpec {
        endgame_empties_values: Some("20".into()), // single value
        runs: 2,
        seed: 1,
        ..Default::default()
    };
    let r = spec.resolve().unwrap();
    assert_eq!(r.points.len(), 1);
    let root = std::env::temp_dir().join(format!("logi_sweep_one_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let (dir, rows) = run_sweep(&r, &root).unwrap();
    assert_eq!(rows.len(), 2, "1 condition × 2 runs");
    assert!(Path::new(&dir.join("metrics.csv")).is_file());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn composition_glem_order_and_support_vary_feature_count() {
    // §6: more conjunction orders ⇒ more features (then overfit); higher
    // support τ ⇒ fewer features. Both must visibly move the metric.
    let f = |o: i64| measure(Axis::GlemMaxOrder, Point::I(o), 1).n_features;
    assert!(f(2) > f(1) && f(4) > f(2), "feature count grows with order");

    let g = |tau: f64| measure(Axis::GlemSupport, Point::F(tau), 1).n_features;
    assert!(
        g(1e-2) < g(1e-4),
        "higher support τ prunes more features ({} < {})",
        g(1e-2),
        g(1e-4)
    );
}

#[test]
fn composition_eval_stages_error_drops_then_saturates() {
    // §6: held-out eval error drops sharply by ≥5 stages, saturates ≥13.
    let e = |n: i64| measure(Axis::EvalStages, Point::I(n), 99).eval_abs_err;
    assert!(e(5) <= e(1), "≥5 stages improves on a single stage");
    assert!(e(13) <= e(5) + 1e-9, "monotone non-increasing to 13");
    assert!(
        (e(20) - e(13)).abs() < 1e-9,
        "saturates past the natural 13 stages"
    );
}
