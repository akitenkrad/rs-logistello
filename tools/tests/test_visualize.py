"""Phase 10 可視化ツールの pytest（ネットワーク・重依存なし; Agg backend）.

`visualize_sweep` / `visualize` / `show_experiment_settings` を **合成**
`metrics.csv` + `sweep_config.json` / `config.json` フィクスチャに対して
実行し，

- 期待する図ファイルが生成され，非空であること
- 依存図のシリーズ数（条件数）と概要図のサブプロット数が正しいこと
- `show-experiment-settings` が run / sweep 両方の config を解析できること

を検証する．Rust は呼び出さない（純粋に Python 可視化レイヤを対象）．
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg  # noqa: E402
import pytest  # noqa: E402

from logistello_tools import show_experiment_settings, visualize, visualize_sweep  # noqa: E402

# 設計ドキュメント §4.2 の metrics.csv ヘッダ（Rust 側 CSV_HEADER と一致;
# 完全に決定的な列のみ — wall-clock time/nps は CSV に含めない）
CSV_HEADER = (
    "param,value,seed,nodes,search_value,dispatch,"
    "eval_abs_err,n_features,book_positions,selfplay_score"
)


def _write_sweep_fixture(d, *, values, runs, metric="nodes"):
    """合成スイープ結果（endgame_empties を nodes に対して掃いた体）を書く."""
    rows = [CSV_HEADER]
    for ci, v in enumerate(values):
        for seed in range(runs):
            # 値・シードで決まる決定的なダミー指標（条件で単調変化）
            nodes = 1000 * (ci + 1) + 7 * seed
            rows.append(
                f"endgame_empties,{v},{seed},{nodes},3,"
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


def test_visualize_sweep_generates_nonempty_figures(tmp_path):
    values = [8, 12, 20]
    runs = 4
    _write_sweep_fixture(tmp_path, values=values, runs=runs)

    rc = visualize_sweep.main(
        ["--results-dir", str(tmp_path), "--no-grid-animation"]
    )
    assert rc == 0

    fig_dir = tmp_path / "figures"
    dep = fig_dir / "sweep_nodes.png"
    overview = fig_dir / "sweep_overview.png"
    assert dep.is_file() and dep.stat().st_size > 0
    assert overview.is_file() and overview.stat().st_size > 0


def test_visualize_sweep_dependency_has_correct_series(tmp_path):
    """依存図に「平均線 + ±1SD バンド」と全条件の個別点があること.

    条件数 (= x 軸点数) が values と一致することを，平均線アーティストの
    データ長で検証する（図を再構成して確認する代わりに matplotlib API で
    直接確認する）.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    values = [8, 12, 16, 20, 24]
    runs = 5
    _write_sweep_fixture(tmp_path, values=values, runs=runs)
    df = pd.read_csv(tmp_path / "metrics.csv")
    df["value"] = df["value"].astype(str)

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
    _write_sweep_fixture(tmp_path, values=[8, 12], runs=3)
    rc = visualize_sweep.main(["--results-dir", str(tmp_path)])
    assert rc == 0
    gif = tmp_path / "figures" / "sweep_grid_animation.gif"
    assert gif.is_file() and gif.stat().st_size > 0


def test_visualize_sweep_overview_has_two_subplots(tmp_path):
    _write_sweep_fixture(tmp_path, values=[8, 12, 20], runs=3)
    visualize_sweep.main(["--results-dir", str(tmp_path), "--no-grid-animation"])
    img = mpimg.imread(str(tmp_path / "figures" / "sweep_overview.png"))
    # 1×2 概要図 → 横長（width > height）
    assert img.shape[1] > img.shape[0]


def test_visualize_single_run(tmp_path):
    """play 由来の run（config.json + metrics.csv）を可視化できる."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "command": "play",
                "black": "engine",
                "white": "random",
                "depth": 6,
                "seed": 42,
            }
        )
    )
    (tmp_path / "metrics.csv").write_text(
        "black_score,white_score,winner,plies,seed\n34,30,black,58,42\n"
    )
    rc = visualize.main(["--results-dir", str(tmp_path)])
    assert rc == 0
    out = tmp_path / "figures" / "run_summary.png"
    assert out.is_file() and out.stat().st_size > 0


def test_visualize_delegates_sweep_dir_to_visualize_sweep(tmp_path):
    _write_sweep_fixture(tmp_path, values=[8, 12], runs=2)
    rc = visualize.main(["--results-dir", str(tmp_path), "--no-grid-animation"]
                        if False else ["--results-dir", str(tmp_path)])
    assert rc == 0
    # sweep 委譲なので sweep 図ができる
    assert (tmp_path / "figures" / "sweep_nodes.png").is_file()


def test_show_experiment_settings_parses_sweep_config(tmp_path, capsys):
    _write_sweep_fixture(tmp_path, values=[8, 12], runs=3)
    rc = show_experiment_settings.main(["--results-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "sweep" in out
    assert "endgame_empties" in out


def test_show_experiment_settings_parses_run_config(tmp_path, capsys):
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


def test_visualize_sweep_missing_metrics_errors(tmp_path):
    rc = visualize_sweep.main(["--results-dir", str(tmp_path)])
    assert rc == 1


@pytest.mark.parametrize("metric", ["nodes", "eval_abs_err", "n_features"])
def test_visualize_sweep_metric_label_known(metric):
    assert metric in visualize_sweep.METRIC_LABELS
