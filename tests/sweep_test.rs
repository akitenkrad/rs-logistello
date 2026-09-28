//! Workspace integration tests for Phase 10 — the `sweep` sensitivity
//! analysis (design doc §6 / §5.1 / §4.2).
//!
//! These tests *define correctness* and must never be weakened:
//!
//! - **Grid expansion** (`grid_*`): the §6 min/max/step + value-list +
//!   log-scale flags expand to the *exact* expected parameter sets, with
//!   inclusive bounds and well-defined float rounding; cross-product
//!   cardinality is correct.
//! - **Determinism** (`determinism_*`): a tiny sweep records identical
//!   trial events across two runs (once the run-identifying keys are taken
//!   out); seeds are derived explicitly from `--seed` + the trial index,
//!   never "current time".
//! - **Output contract** (`output_contract_*`): the sweep is recorded as a
//!   runvault parent run holding the resolved grid plus one child run per
//!   condition, the children point back at the parent through their
//!   lineage, and each child holds one trial event per trial.
//! - **Metric sanity** (`metric_*`): for a cheap swept parameter the
//!   recorded metric is plausible and monotone where §6 theory says so; a
//!   degenerate single-point grid works.
//! - **Composition** (`composition_*`): sweeping a parameter actually
//!   varies it in the measured run (a flag must not be parsed-then-ignored).
//!
//! No heavy sweeps: every condition here is tiny (≤ a few search calls).

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use logistello_cli::sweep::{
    Axis, Point, SweepParameters, SweepSpec, expand_log_range, expand_range, measure,
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
    let cfg = SweepParameters::from_resolved(&r);
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
//  Output contract (the runvault sweep parent + its children)          //
// --------------------------------------------------------------------- //

/// Reads a run directory's `run.json`.
fn run_meta(dir: &Path) -> serde_json::Value {
    serde_json::from_str(&fs::read_to_string(dir.join("run.json")).unwrap()).unwrap()
}

/// The conditions, which live under `parameters` in the `config.json`
/// envelope rather than in `run.json`.
fn run_parameters(dir: &Path) -> serde_json::Value {
    let config: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    config["parameters"].clone()
}

/// The child runs of a sweep parent, found the way `runvault.read` finds
/// them: the parent's neighbours whose lineage points back at it.
fn children_of(parent: &Path) -> Vec<std::path::PathBuf> {
    let parent_uid = run_meta(parent)["run_uid"].as_str().unwrap().to_string();
    let experiment_dir = parent.parent().unwrap();
    let mut out: Vec<std::path::PathBuf> = fs::read_dir(experiment_dir)
        .unwrap()
        .filter_map(|e| {
            let path = e.unwrap().path();
            if path == parent || !path.join("run.json").is_file() {
                return None;
            }
            let lineage = run_meta(&path)["lineage"].clone();
            (lineage["parent_run_uid"] == serde_json::json!(parent_uid)).then_some(path)
        })
        .collect();
    out.sort();
    out
}

/// The trial events a child recorded, with the run-identifying keys (which
/// differ between two runs of the same sweep by design) taken out.
fn trial_events(child: &Path) -> Vec<serde_json::Value> {
    fs::read_to_string(child.join("events.jsonl"))
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut v: serde_json::Value = serde_json::from_str(l).unwrap();
            let object = v.as_object_mut().unwrap();
            object.remove("ts");
            object.remove("run_uid");
            v
        })
        .collect()
}

/// Runs a tiny sweep into a unique temp results root. Returns the root, the
/// sweep parent's directory and the measured rows.
fn tiny_sweep(
    tag: &str,
) -> (
    std::path::PathBuf,
    std::path::PathBuf,
    Vec<logistello_cli::sweep::MetricRow>,
) {
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
    let (parent, rows) = run_sweep(&resolved, &root, false).unwrap();
    (root, parent, rows)
}

#[test]
fn output_contract_parent_holds_the_resolved_grid() {
    let (root, parent, rows) = tiny_sweep("parent");
    assert_eq!(rows.len(), 6, "2 conditions × 3 runs = 6 trials");

    let meta = run_meta(&parent);
    assert_eq!(meta["subcommand"], "sweep");
    assert_eq!(meta["experiment"], "logistello");
    // The grid definition, and nothing measured: the parent is not a run of
    // the model, it is the statement of what its children ran.
    let params = run_parameters(&parent);
    let params = &params;
    assert_eq!(params["param"], "endgame_empties");
    assert_eq!(params["metric"], "nodes");
    assert_eq!(params["values"], serde_json::json!(["8", "12"]));
    assert_eq!(params["runs"], 3);
    assert_eq!(params["n_conditions"], 2);
    assert_eq!(params["n_trials"], 6);
    assert!(
        !parent.join("events.jsonl").exists(),
        "the parent measures nothing itself"
    );
    // A sweep parent must not claim a master seed: it is not one run of the
    // model, and the base seed is already in /parameters.seed.
    assert!(
        meta["master_seed"].is_null(),
        "no master seed on the parent"
    );

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn output_contract_one_child_per_condition_with_its_trials() {
    let (root, parent, _) = tiny_sweep("children");
    let children = children_of(&parent);
    assert_eq!(children.len(), 2, "one child run per swept value");

    let mut values: Vec<String> = Vec::new();
    for child in &children {
        let meta = run_meta(child);
        // A different subcommand from the parent's, so `runvault path
        // --subcommand` can tell a condition from the sweep it belongs to.
        assert_eq!(meta["subcommand"], "sweep-point");
        let params = run_parameters(child);
        assert_eq!(params["param"], "endgame_empties");
        assert_eq!(params["runs"], 3);
        assert_eq!(params["base_seed"], 42);
        values.push(params["value"].as_str().unwrap().to_string());

        // One event per trial, carrying the seed it actually used and only
        // the fields this axis measures.
        let events = trial_events(child);
        assert_eq!(events.len(), 3, "runs=3 trials in this condition");
        for (i, e) in events.iter().enumerate() {
            assert_eq!(e["schema"], "x.logistello.trial");
            assert_eq!(e["unit_id"], format!("trial-{i}"));
            assert!(e["nodes"].is_u64(), "the axis measures a node count");
            assert!(e["dispatch"].is_string(), "dispatch is a label");
            assert!(
                e.get("eval_abs_err").is_none() && e.get("book_positions").is_none(),
                "a column this axis never measured must not be written as 0"
            );
        }

        // The condition's one number lives in the child's metrics.csv.
        let metrics = fs::read_to_string(child.join("metrics.csv")).unwrap();
        assert!(metrics.contains(",run,mean_nodes,"), "{metrics}");
        assert!(metrics.contains(",run,n_units,"), "{metrics}");
    }
    values.sort();
    assert_eq!(values, vec!["12".to_string(), "8".to_string()]);

    let _ = fs::remove_dir_all(&root);
}

// --------------------------------------------------------------------- //
//  Determinism                                                          //
// --------------------------------------------------------------------- //

#[test]
fn determinism_trial_events_identical_across_runs() {
    // A tiny sweep (1 axis × 2 values × 2 seeds) twice ⇒ every recorded
    // trial is identical. Only the run's own identity (`run_uid`, `ts`)
    // differs, which is what makes two runs two runs.
    //
    // Keyed by (condition, trial) rather than compared as a list: the
    // children's directory names carry the start time, so which child sorts
    // first depends on whether the two starts fell in the same second.
    let make = |tag: &str| -> BTreeMap<(String, String), serde_json::Value> {
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
        let (parent, _) = run_sweep(&r, &root, false).unwrap();
        let mut trials = BTreeMap::new();
        for child in children_of(&parent) {
            let value = run_parameters(&child)["value"]
                .as_str()
                .unwrap()
                .to_string();
            for event in trial_events(&child) {
                let unit = event["unit_id"].as_str().unwrap().to_string();
                trials.insert((value.clone(), unit), event);
            }
        }
        let _ = fs::remove_dir_all(&root);
        trials
    };
    let a = make("a");
    let b = make("b");
    assert_eq!(a.len(), 4, "2 values × 2 seeds");
    assert_eq!(
        a, b,
        "per-(condition, seed) measurement is reproducible (seeds derived \
         from --seed + trial index, never wall-clock)"
    );
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
    let (parent, rows) = run_sweep(&r, &root, false).unwrap();
    assert_eq!(rows.len(), 2, "1 condition × 2 runs");
    // One condition ⇒ one child, and the child is where the numbers are:
    // the parent measures nothing.
    let children = children_of(&parent);
    assert_eq!(children.len(), 1);
    assert!(Path::new(&children[0].join("metrics.csv")).is_file());
    assert_eq!(trial_events(&children[0]).len(), 2);
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
