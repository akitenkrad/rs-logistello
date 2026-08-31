"""可視化ツールの pytest（ネットワーク・重依存なし; Agg backend）.

`visualize_sweep` / `visualize` / `show_experiment_settings` を，**runvault の
run ディレクトリ**（Python 側の `runvault.run.Run` で合成する）と，**移行前の
`results/<ts>/`**（平坦な `config.json` / `sweep_config.json` + wide な
`metrics.csv`）の両方に対して実行し，

- 期待する図ファイルが生成され，非空であること
- 依存図のシリーズ数（条件数）と概要図のサブプロット数が正しいこと
- `show-experiment-settings` が runvault の run と移行前の両方を解析できること

を検証する．Rust は呼び出さない（純粋に Python 可視化レイヤを対象）．
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg  # noqa: E402
import pytest  # noqa: E402
from runvault.read import figures_dir  # noqa: E402
from runvault.run import Run  # noqa: E402

from logistello_tools import show_experiment_settings, visualize, visualize_sweep  # noqa: E402

EXPERIMENT = "logistello"
TRIAL_EVENT = "x.logistello.trial"

# 移行前の metrics.csv ヘッダ（runvault 以前の sweep 出力）
LEGACY_CSV_HEADER = (
    "param,value,seed,nodes,search_value,dispatch,"
    "eval_abs_err,n_features,book_positions,selfplay_score"
)


def _nodes(ci: int, seed: int) -> int:
    """値・シードで決まる決定的なダミー指標（条件で単調変化）."""
    return 1000 * (ci + 1) + 7 * seed


def _write_sweep_fixture(root, *, values, runs, metric="nodes"):
    """合成スイープ（endgame_empties を nodes に対して掃いた体）を runvault の
    親 run + 条件ごとの子 run として書き，親のディレクトリを返す."""
    parent = Run.start(
        EXPERIMENT,
        "sweep",
        repo_id=EXPERIMENT,
        domain="simulation",
        results_root=root,
        parameters={
            "param": "endgame_empties",
            "metric": metric,
            "values": [str(v) for v in values],
            "runs": runs,
            "seed": 42,
            "n_conditions": len(values),
            "n_trials": len(values) * runs,
        },
        seed_pointers=["/seed"],
        sweep_parent=True,
    )
    for ci, v in enumerate(values):
        child = Run.start(
            EXPERIMENT,
            "sweep-point",
            repo_id=EXPERIMENT,
            domain="simulation",
            results_root=root,
            parameters={
                "param": "endgame_empties",
                "value": str(v),
                "runs": runs,
                "base_seed": 42,
            },
            seed_pointers=["/base_seed"],
            master_seed=42,
            replicate_index=0,
            lineage={
                "sweep_id": parent.sweep_id,
                "parent_run_uid": parent.run_uid,
            },
        )
        for seed in range(runs):
            child.log_event(
                TRIAL_EVENT,
                {
                    "unit_id": f"trial-{seed}",
                    "trial_index": seed,
                    "seed": seed,
                    "nodes": _nodes(ci, seed),
                    "search_value": 3,
                    "dispatch": "midgame",
                },
            )
        child.log_metrics(
            "run",
            {
                "n_units": float(runs),
                "mean_nodes": sum(_nodes(ci, s) for s in range(runs)) / runs,
            },
        )
        child.finish()
    return str(parent.finish())


def _write_legacy_sweep_fixture(d, *, values, runs, metric="nodes"):
    """移行前の `results/<ts>/{sweep_config.json, metrics.csv}`."""
    rows = [LEGACY_CSV_HEADER]
    for ci, v in enumerate(values):
        for seed in range(runs):
            rows.append(
                f"endgame_empties,{v},{seed},{_nodes(ci, seed)},3,"
                f"midgame,0.0000,0,0,0"
            )
    (d / "metrics.csv").write_text("\n".join(rows) + "\n")
    cfg = {
        "command": "sweep",
        "axis": "endgame_empties",
        "metric": metric,
        "values": [str(v) for v in values],
        "runs": runs,
        "seed": 42,
        "n_conditions": len(values),
        "n_trials": len(values) * runs,
    }
    (d / "sweep_config.json").write_text(json.dumps(cfg, indent=2))
    return cfg


def _write_play_fixture(root):
    """合成 `play` run（条件 + run スコープ指標 + 終端イベント）."""
    run = Run.start(
        EXPERIMENT,
        "play",
        repo_id=EXPERIMENT,
        domain="simulation",
        results_root=root,
        parameters={
            "black": "engine",
            "white": "random",
            "depth": 6,
            "endgame_empties": 20,
            "seed": 42,
        },
        seed_pointers=["/seed"],
        master_seed=42,
    )
    run.log_metrics(
        "run", {"black_score": 34.0, "white_score": 30.0, "plies": 58.0}
    )
    run.log_event("observation", {"unit_id": "game", "t": 58, "t_unit": "turn"})
    run.log_event(
        "terminal",
        {
            "unit_id": "game",
            "t": 58,
            "t_unit": "turn",
            "outcome": "black",
            "censored": False,
            "budget": 60,
            "moves": "d3 c3 f5",
        },
    )
    return str(run.finish())


def test_visualize_sweep_generates_nonempty_figures(tmp_path):
    values = [8, 12, 20]
    parent = _write_sweep_fixture(tmp_path, values=values, runs=4)

    rc = visualize_sweep.main(
        ["--results-dir", parent, "--no-grid-animation"]
    )
    assert rc == 0

    fig_dir = figures_dir(parent)
    dep = os.path.join(fig_dir, "sweep_nodes.png")
    overview = os.path.join(fig_dir, "sweep_overview.png")
    assert os.path.isfile(dep) and os.path.getsize(dep) > 0
    assert os.path.isfile(overview) and os.path.getsize(overview) > 0
    # 図は run ディレクトリの外に置く（manifest.csv は finish() が確定させる）
    assert not os.path.commonpath([fig_dir, parent]) == parent


def test_visualize_sweep_reads_the_children_trials(tmp_path):
    """親 run から条件と試行を組み直せること（子の events が試行の正本）."""
    values = [8, 12]
    runs = 3
    parent = _write_sweep_fixture(tmp_path, values=values, runs=runs)
    df = visualize_sweep.load_trials(parent)
    assert len(df) == len(values) * runs
    assert set(df["value"]) == {"8", "12"}
    # 条件列は文字列なので行順は "12" < "8"（辞書順）．どの試行がどの条件の
    # ものかは (value, trial_index) で引く．
    measured = {
        (row["value"], row["trial_index"]): row["nodes"]
        for _, row in df.iterrows()
    }
    assert measured == {
        (str(v), s): _nodes(ci, s)
        for ci, v in enumerate(values)
        for s in range(runs)
    }
    # 掃引していない軸の列は書かれていない（0 で埋めていない）
    assert "book_positions" not in df.columns


def test_visualize_sweep_dependency_has_correct_series(tmp_path):
    """依存図に「平均線 + ±1SD バンド」と全条件の個別点があること."""
    import matplotlib.pyplot as plt

    values = [8, 12, 16, 20, 24]
    runs = 5
    parent = _write_sweep_fixture(tmp_path, values=values, runs=runs)
    df = visualize_sweep.load_trials(parent)

    fig, ax = plt.subplots()
    visualize_sweep._plot_dependency(ax, df, "endgame_empties", "nodes")
    # 平均線（line2D, marker='o'）の x 長 == 条件数
    line = ax.get_lines()[0]
    assert len(line.get_xdata()) == len(values)
    # 複数シード → 個別散布点が values×runs 個（散布は PathCollection;
    # ±1SD バンドの PolyCollection は除外して厳密に数える）
    from matplotlib.collections import PathCollection

    n_scatter = sum(
        c.get_offsets().shape[0]
        for c in ax.collections
        if isinstance(c, PathCollection)
    )
    assert n_scatter == len(values) * runs
    plt.close(fig)


def test_visualize_sweep_grid_animation_gif(tmp_path):
    parent = _write_sweep_fixture(tmp_path, values=[8, 12], runs=3)
    rc = visualize_sweep.main(["--results-dir", parent])
    assert rc == 0
    gif = os.path.join(figures_dir(parent), "sweep_grid_animation.gif")
    assert os.path.isfile(gif) and os.path.getsize(gif) > 0


def test_visualize_sweep_overview_has_two_subplots(tmp_path):
    parent = _write_sweep_fixture(tmp_path, values=[8, 12, 20], runs=3)
    visualize_sweep.main(["--results-dir", parent, "--no-grid-animation"])
    img = mpimg.imread(os.path.join(figures_dir(parent), "sweep_overview.png"))
    # 1×2 概要図 → 横長（width > height）
    assert img.shape[1] > img.shape[0]


def test_visualize_sweep_reads_a_legacy_results_dir(tmp_path):
    """移行前の `results/<ts>/` も --results-dir に渡せば従来どおり読める."""
    _write_legacy_sweep_fixture(tmp_path, values=[8, 12], runs=2)
    rc = visualize_sweep.main(["--results-dir", str(tmp_path), "--no-grid-animation"])
    assert rc == 0
    dep = tmp_path / "figures" / "sweep_nodes.png"
    assert dep.is_file() and dep.stat().st_size > 0


def test_visualize_single_run(tmp_path):
    """play の run（条件 + run スコープ指標）を可視化できる."""
    run_dir = _write_play_fixture(tmp_path)
    rc = visualize.main(["--results-dir", run_dir])
    assert rc == 0
    out = os.path.join(figures_dir(run_dir), "run_summary.png")
    assert os.path.isfile(out) and os.path.getsize(out) > 0


def test_visualize_single_run_reads_run_scope_metrics(tmp_path):
    run_dir = _write_play_fixture(tmp_path)
    metrics = visualize._load_metrics(run_dir)
    assert metrics == {"black_score": 34.0, "white_score": 30.0, "plies": 58.0}
    config = visualize._load_run_config(run_dir)
    assert config["black"] == "engine" and config["seed"] == 42


def test_visualize_delegates_sweep_parent_to_visualize_sweep(tmp_path):
    parent = _write_sweep_fixture(tmp_path, values=[8, 12], runs=2)
    rc = visualize.main(["--results-dir", parent])
    assert rc == 0
    # sweep 委譲なので sweep 図ができる
    assert os.path.isfile(os.path.join(figures_dir(parent), "sweep_nodes.png"))


def test_show_experiment_settings_parses_a_sweep_parent(tmp_path, capsys):
    parent = _write_sweep_fixture(tmp_path, values=[8, 12], runs=3)
    rc = show_experiment_settings.main(["--results-dir", parent])
    assert rc == 0
    out = capsys.readouterr().out
    assert "sweep" in out
    assert "endgame_empties" in out


def test_show_experiment_settings_parses_a_run(tmp_path, capsys):
    run_dir = _write_play_fixture(tmp_path)
    rc = show_experiment_settings.main(["--results-dir", run_dir, "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "play"
    assert payload["config"]["black"] == "engine"


def test_show_experiment_settings_reads_a_legacy_run(tmp_path, capsys):
    """移行前の平坦な config.json（封筒なし）もそのまま読める."""
    (tmp_path / "config.json").write_text(
        json.dumps({"command": "play", "depth": 6, "seed": 42})
    )
    rc = show_experiment_settings.main(
        ["--results-dir", str(tmp_path), "--json"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "run"
    assert payload["config"]["command"] == "play"


def test_show_experiment_settings_missing_dir(tmp_path, capsys):
    missing = tmp_path / "nope"
    rc = show_experiment_settings.main(["--results-dir", str(missing)])
    assert rc == 1


def test_visualize_sweep_missing_trials_errors(tmp_path):
    rc = visualize_sweep.main(["--results-dir", str(tmp_path)])
    assert rc == 1


@pytest.mark.parametrize("metric", ["nodes", "eval_abs_err", "n_features"])
def test_visualize_sweep_metric_label_known(metric):
    assert metric in visualize_sweep.METRIC_LABELS
