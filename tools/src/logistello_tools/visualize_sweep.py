"""sweep 結果の可視化（Phase 10; 設計ドキュメント §6）．

`logistello sweep` は sweep の親 run（掃引の格子そのもの）と，条件ごとの子 run
（試行を `x.logistello.trial` イベントとして持つ）を書く．ここでは親から格子を，
子から試行を集めて，**スイープした 1 パラメータ vs 指標** の依存図（シード試行
の平均 ± 標準偏差）と，各パラメータ値（＝条件）ごとの試行を格子に並べた合成
アニメーション `sweep_grid_animation.gif` を生成する．

```
{output_dir}/
├── sweep_<metric>.png      ← パラメータ vs 指標 依存図（平均±SD + 個別点）
├── sweep_overview.png      ← 依存図 + 試行分布の 1×2 概要
└── sweep_grid_animation.gif← 条件（パラメータ値）別 試行バーの進行アニメ
```

どの sweep を見るかは `--results-dir` を省略すれば runvault が答える
(`runvault path --experiment logistello --latest --subcommand sweep`)．図は run
ディレクトリの *隣*（`results/logistello/figures/<run_slug>/`）に置く．

移行前の `results/<YYYYMMDD_HHMMSS>/{sweep_config.json, metrics.csv}`（1 行 1
試行の wide な CSV）も `--results-dir` に直接渡せば従来どおり読める．

Usage:
    logistello-tools visualize-sweep [--results-dir DIR] [--output-dir OUT]
                                     [--no-grid-animation] [--fps FPS]
                                     [--max-frames N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # ヘッドレス / pytest（ネットワーク・GUI 不要）

import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from runvault.read import (  # noqa: E402
    config_parameters,
    figures_dir,
    load_run_meta,
    runvault_path,
    sweep_events_table,
)

#: runvault 上の実験名 (Rust 側 `record::EXPERIMENT` と同じ)．
EXPERIMENT = "logistello"

#: 試行 1 本を表す実験固有イベント (Rust 側 `record::TRIAL_EVENT` と同じ)．
TRIAL_EVENT = "x.logistello.trial"

# --------------------------------------------------------------------------- #
# 配色
# --------------------------------------------------------------------------- #
COLOR_BG = "#FAFAF8"
COLOR_MEAN = "#1565C0"
COLOR_PTS = "#90A4AE"
COLOR_BAND = "#1565C0"

# 指標 → (CSV 列, 表示名)
METRIC_LABELS: dict[str, str] = {
    "nodes": "探索ノード数",
    "eval_abs_err": "評価関数 絶対誤差",
    "n_features": "生成特徴量数",
    "book_positions": "定石局面数",
    "selfplay_score": "自己対戦スコア (石差)",
}

PARAM_LABELS: dict[str, str] = {
    "probcut_t": "ProbCut 閾値 T",
    "probcut_depth_pair": "ProbCut 深度対 (d:h)",
    "multi_stages": "Multi-ProbCut 段数 k",
    "eval_stages": "評価関数 段数",
    "glem_max_order": "GLEM 最大次数",
    "glem_support": "GLEM サポート閾値 τ",
    "max_depth": "反復深化 最大深度",
    "tt_size": "置換表サイズ (2^k)",
    "book_depth": "定石学習 深度",
    "endgame_empties": "終盤完全読み 開始空きマス",
    "drawishness": "drawishness 重み λ",
}


# --------------------------------------------------------------------------- #
# 入出力ヘルパ
# --------------------------------------------------------------------------- #
def load_sweep_config(sweep_dir: str) -> dict | None:
    """掃引の定義（軸・指標・値・試行数）を読む．

    runvault の sweep なら親 run の `parameters` がそれである．移行前の
    ディレクトリは `sweep_config.json` を読む（キー名が `axis` なので `param`
    に揃えてから返す）．
    """
    if load_run_meta(sweep_dir, required=False) is not None:
        return config_parameters(sweep_dir, required=False)
    path = os.path.join(sweep_dir, "sweep_config.json")
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        if "axis" in cfg and "param" not in cfg:
            cfg["param"] = cfg["axis"]
        return cfg
    return None


def load_trials(sweep_dir: str) -> pd.DataFrame:
    """試行を 1 行 1 試行の表にする．

    runvault の sweep では試行は子 run の `events.jsonl` にあり，条件は子の
    `parameters` にある — `sweep_events_table` が両者を突き合わせる．移行前の
    ディレクトリは 1 行 1 試行の wide な `metrics.csv` をそのまま読む．
    """
    if load_run_meta(sweep_dir, required=False) is not None:
        df = sweep_events_table(sweep_dir, ["value"], kind=TRIAL_EVENT)
    else:
        path = os.path.join(sweep_dir, "metrics.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"metrics.csv が見つかりません: {path}")
        df = pd.read_csv(path)
    df["value"] = df["value"].astype(str)
    return df


def _value_sort_key(v: str):
    """`value` 列をなるべく数値順に整列する（`d:h` は (d, h) タプル）."""
    s = str(v)
    if ":" in s:
        try:
            d, h = s.split(":")
            return (float(d), float(h))
        except ValueError:
            return (float("inf"), s)
    try:
        return (float(s),)
    except ValueError:
        return (float("inf"), s)


def _ordered_values(df: pd.DataFrame) -> list[str]:
    return sorted((str(v) for v in df["value"].unique()), key=_value_sort_key)


def _x_positions(values: list[str]) -> tuple[list[float], list[str]]:
    """カテゴリ（`d:h` 等）でも等間隔にプロットできるよう連番 x を返す."""
    return list(range(len(values))), values


# --------------------------------------------------------------------------- #
# 依存図（パラメータ vs 指標：平均 ± SD）
# --------------------------------------------------------------------------- #
def _plot_dependency(
    ax: plt.Axes,
    df: pd.DataFrame,
    param: str,
    metric: str,
) -> None:
    ax.set_facecolor(COLOR_BG)
    values = _ordered_values(df)
    xs, labels = _x_positions(values)
    n_seeds = df["seed"].nunique()

    means, stds = [], []
    for v in values:
        g = df[df["value"].astype(str) == v][metric].astype(float)
        means.append(g.mean())
        stds.append(g.std(ddof=0) if len(g) > 1 else 0.0)
        if n_seeds > 1:
            ax.scatter(
                [xs[values.index(v)]] * len(g),
                g.values,
                color=COLOR_PTS,
                alpha=0.35,
                s=22,
                zorder=2,
            )

    means_a = np.asarray(means, dtype=float)
    stds_a = np.asarray(stds, dtype=float)
    ax.plot(xs, means_a, color=COLOR_MEAN, lw=2, marker="o", ms=5, zorder=3,
            label="平均")
    if n_seeds > 1:
        ax.fill_between(
            xs,
            means_a - stds_a,
            means_a + stds_a,
            color=COLOR_BAND,
            alpha=0.18,
            zorder=1,
            label="±1 SD",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8, rotation=0)
    ax.set_xlabel(PARAM_LABELS.get(param, param))
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(
        f"{PARAM_LABELS.get(param, param)} に対する "
        f"{METRIC_LABELS.get(metric, metric)} の感度"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def save_dependency(df: pd.DataFrame, param: str, metric: str,
                    out_path: str, subtitle: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=COLOR_BG)
    fig.suptitle("Logistello 感度分析 — パラメータ依存", fontsize=13)
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", fontsize=9, color="#666")
    _plot_dependency(ax, df, param, metric)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def save_overview(df: pd.DataFrame, param: str, metric: str,
                  out_path: str, subtitle: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=COLOR_BG)
    fig.suptitle("Logistello 感度分析 — 概要", fontsize=14)
    if subtitle:
        fig.text(0.5, 0.95, subtitle, ha="center", fontsize=9, color="#666")

    _plot_dependency(axes[0], df, param, metric)

    # 右: 試行分布（条件ごとの箱ひげ）
    ax = axes[1]
    ax.set_facecolor(COLOR_BG)
    values = _ordered_values(df)
    data = [
        df[df["value"].astype(str) == v][metric].astype(float).values
        for v in values
    ]
    ax.boxplot(data, tick_labels=values)
    ax.set_xlabel(PARAM_LABELS.get(param, param))
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title("条件ごとの試行分布")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


# --------------------------------------------------------------------------- #
# 条件（パラメータ値）別 グリッドアニメーション
# --------------------------------------------------------------------------- #
def save_grid_animation(
    df: pd.DataFrame,
    param: str,
    metric: str,
    out_path: str,
    *,
    fps: int = 5,
    max_frames: int = 0,
    subtitle: str = "",
) -> bool:
    """各セル = 1 パラメータ値（条件）．フレームを進めるごとに，その条件の
    シード試行を 1 つずつ累積したバーチャートを描く合成 GIF．

    schelling の「組み合わせ別グリッドアニメーション」に対応する（盤面の
    時系列スナップショットを持たないため，sweep では「条件 × 試行」の
    進行を可視化する）．スナップショット欠落時に schelling がスキップする
    のと同様，試行が無ければ警告のみで False を返す．
    """
    values = _ordered_values(df)
    if not values:
        print("  警告: metrics.csv に条件がありません．スキップ．")
        return False

    # 各条件の試行系列（seed 昇順）
    series: list[np.ndarray] = []
    for v in values:
        g = df[df["value"].astype(str) == v].sort_values("seed")
        series.append(g[metric].astype(float).values)
    n_runs = max((len(s) for s in series), default=0)
    if n_runs == 0:
        print("  警告: 試行が 0 件です．グリッドアニメーションをスキップ．")
        return False

    n = len(values)
    n_cols = min(n, 4)
    n_rows = (n + n_cols - 1) // n_cols

    frames = list(range(1, n_runs + 1))
    if max_frames > 0 and len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in idx]

    y_max = float(np.nanmax([s.max() if len(s) else 0.0 for s in series])) or 1.0

    cell_w, cell_h = 3.0, 2.6
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(6.0, n_cols * cell_w), max(4.0, n_rows * cell_h + 1.0)),
        facecolor=COLOR_BG, squeeze=False,
    )
    fig.suptitle(
        "Logistello 感度分析 — 条件別 試行進行アニメーション\n"
        f"{subtitle}" if subtitle else
        "Logistello 感度分析 — 条件別 試行進行アニメーション",
        fontsize=12,
    )

    bars_per_cell: list = []
    for idx in range(n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        ax.set_facecolor(COLOR_BG)
        if idx >= n:
            ax.axis("off")
            bars_per_cell.append(None)
            continue
        s = series[idx]
        b = ax.bar(range(len(s)), [0.0] * len(s), color=COLOR_MEAN, alpha=0.85)
        ax.set_ylim(0, y_max * 1.1)
        ax.set_title(
            f"{PARAM_LABELS.get(param, param)}={values[idx]}", fontsize=9
        )
        ax.set_xlabel("試行 (seed 順)", fontsize=8)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=8)
        ax.tick_params(labelsize=7)
        bars_per_cell.append(b)

    def _update(f: int):
        artists = []
        for ci, bars in enumerate(bars_per_cell):
            if bars is None:
                continue
            s = series[ci]
            for j, rect in enumerate(bars):
                rect.set_height(s[j] if j < f and j < len(s) else 0.0)
                artists.append(rect)
        return artists

    ani = animation.FuncAnimation(
        fig, _update, frames=frames, blit=False,
        interval=1000 // max(fps, 1),
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    ani.save(out_path, writer="pillow", fps=fps, dpi=90)
    plt.close(fig)
    print(f"  保存: {out_path}  ({len(frames)} フレーム, {n} 条件)")
    return True


# --------------------------------------------------------------------------- #
# メイン
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="logistello-tools visualize-sweep",
        description="Logistello 感度分析スイープ 可視化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--results-dir", "--results_dir", "--sweep-dir", "--sweep_dir",
        default=None,
        help="sweep の親 run ディレクトリ (省略時は runvault が最新を答える)",
    )
    p.add_argument(
        "--results-root", "--results_root", default="results",
        help="run ディレクトリの置き場 (default: results)",
    )
    p.add_argument(
        "--output-dir", "--output_dir", default=None,
        help="図の保存先 (default: run ディレクトリの隣の figures/<run_slug>)",
    )
    p.add_argument(
        "--no-grid-animation", "--no_grid_animation", action="store_true",
        help="条件別グリッドアニメーションをスキップする",
    )
    p.add_argument("--fps", type=int, default=5,
                   help="グリッドアニメーションの FPS (default: 5)")
    p.add_argument(
        "--max-frames", "--max_frames", type=int, default=0,
        help="グリッドアニメーション最大フレーム数 (0=全件)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.results_dir is None:
        try:
            sweep_dir = runvault_path(
                EXPERIMENT,
                results_root=args.results_root,
                subcommand="sweep",
            )
        except Exception as e:  # runvault バイナリが無い / 終わった sweep が無い
            print(f"エラー: sweep を解決できません: {e}", file=sys.stderr)
            return 1
    else:
        sweep_dir = args.results_dir

    print("=== Logistello 感度分析スイープ 可視化 ===")
    print(f"スイープ結果: {sweep_dir}")

    print("[1/4] 試行を読み込み中 ...")
    try:
        df = load_trials(sweep_dir)
    except (FileNotFoundError, SystemExit) as e:
        print(f"エラー: 試行を読めません: {e}", file=sys.stderr)
        return 1
    print(f"      {len(df)} 行")

    out_dir = args.output_dir or figures_dir(sweep_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"出力先:       {out_dir}")
    print("------------------------------------------")

    print("[2/4] 掃引の定義を確認中 ...")
    config = load_sweep_config(sweep_dir) or {}
    param = str(config.get("param") or df.get("param", pd.Series(["value"])).iloc[0])
    metric = config.get("metric")
    if metric is None:
        # 定義が無い: 値の立っている指標列から推定する
        metric = next(
            (m for m in METRIC_LABELS
             if m in df.columns and df[m].astype(float).abs().sum() > 0),
            "nodes",
        )
    if metric not in df.columns:
        print(
            f"エラー: 指標 `{metric}` の列が試行にありません "
            f"(列: {list(df.columns)})",
            file=sys.stderr,
        )
        return 1
    n_seeds = df["seed"].nunique()
    n_cond = df["value"].nunique()
    subtitle = (
        f"{PARAM_LABELS.get(param, param)}: {n_cond} 値 × {n_seeds} シード"
    )
    print(f"      param={param}  metric={metric}  {subtitle}")

    print("[3/4] 依存図・概要図を保存中 ...")
    save_dependency(
        df, param, metric,
        os.path.join(out_dir, f"sweep_{metric}.png"), subtitle,
    )
    save_overview(
        df, param, metric,
        os.path.join(out_dir, "sweep_overview.png"), subtitle,
    )

    if args.no_grid_animation:
        print("[4/4] グリッドアニメーションをスキップしました")
    else:
        print("[4/4] 条件別グリッドアニメーションを生成中 ...")
        save_grid_animation(
            df, param, metric,
            os.path.join(out_dir, "sweep_grid_animation.gif"),
            fps=args.fps, max_frames=args.max_frames, subtitle=subtitle,
        )

    print("------------------------------------------")
    print("完了．出力ファイル:")
    for f in sorted(os.listdir(out_dir)):
        fp = os.path.join(out_dir, f)
        if os.path.isfile(fp):
            print(f"  {f:32s} ({os.path.getsize(fp) / 1024:7.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
