"""単発 run（`play` / `bench-search` / `match-replay` / …）結果の可視化．

runvault の run ディレクトリから条件（`config.json` の `parameters`）と run
スコープの指標（`metrics.csv` の step を持たない行）を読み，設定の要約と指標の
棒グラフを生成する．sweep の親 run が指定された場合は `visualize_sweep` に
委譲する．

どの run を見るかは `--results-dir` を省略すれば runvault が答える
(`runvault path --experiment logistello --latest --subcommand play --standalone`)．

図は run ディレクトリの *隣* (`results/logistello/figures/<run_slug>/`) に置く．
`manifest.csv` は `finish()` が確定させたもので，run が終わった後に足したものは
そこに載らないためである．

```
{output_dir}/
└── run_summary.png   ← 設定テキスト + 指標棒グラフ
```

Usage:
    logistello-tools visualize
    logistello-tools visualize --subcommand bench-search
    logistello-tools visualize --results-dir results/logistello/<run-slug>
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")  # ヘッドレス / pytest

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from runvault.read import (  # noqa: E402
    config_parameters,
    figures_dir,
    load_run_meta,
    run_scope_metrics,
    run_subcommand,
    runvault_path,
)

#: runvault 上の実験名 (Rust 側 `record::EXPERIMENT` と同じ)．
EXPERIMENT = "logistello"

COLOR_BG = "#FAFAF8"
COLOR_BAR = "#1565C0"


def _is_sweep_parent(results_dir: str) -> bool:
    """sweep の親 run か（移行前の `sweep_config.json` を持つ場合も含む）."""
    if os.path.exists(os.path.join(results_dir, "sweep_config.json")):
        return True
    if load_run_meta(results_dir, required=False) is None:
        return False
    return run_subcommand(results_dir) == "sweep"


def _load_run_config(results_dir: str) -> dict | None:
    """条件．runvault の `config.json` は封筒なので `parameters` を取り出す．"""
    return config_parameters(results_dir, required=False)


def _load_metrics(results_dir: str) -> dict[str, float] | None:
    """run を 1 つの値ずつで表す指標．

    runvault の `metrics.csv` は long 形式で，step を持たない行が run スコープ．
    移行前の wide な `metrics.csv`（1 行 1 実行）は先頭行の数値列を使う．
    """
    if load_run_meta(results_dir, required=False) is not None:
        scoped = run_scope_metrics(results_dir)
        return scoped or None
    path = os.path.join(results_dir, "metrics.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    row = df.iloc[0]
    return {
        str(k): float(v)
        for k, v in row.items()
        if isinstance(v, (int, float)) or (isinstance(v, str) and _is_num(v))
    }


def save_run_summary(config: dict | None, metrics: dict[str, float] | None,
                     out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=COLOR_BG)
    fig.suptitle("Logistello 単発実行 — 概要", fontsize=14)

    # 左: 設定テキスト
    ax = axes[0]
    ax.set_facecolor(COLOR_BG)
    ax.axis("off")
    if config:
        lines = [f"{k:<18}: {v}" for k, v in config.items()]
        ax.text(
            0.02, 0.98, "\n".join(lines), va="top", ha="left",
            family="monospace", fontsize=10, transform=ax.transAxes,
        )
    else:
        ax.text(0.5, 0.5, "(条件なし)", ha="center", va="center",
                transform=ax.transAxes, color="#999")
    ax.set_title("実行条件 (config.json の parameters)")

    # 右: run スコープ指標の棒グラフ
    ax = axes[1]
    ax.set_facecolor(COLOR_BG)
    if metrics:
        keys = list(metrics.keys())
        ax.bar(range(len(keys)), [metrics[k] for k in keys],
               color=COLOR_BAR, alpha=0.85)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_title("run スコープ指標 (metrics.csv)")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "(指標なし)", ha="center", va="center",
                transform=ax.transAxes, color="#999")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="logistello-tools visualize")
    parser.add_argument(
        "--results-dir", "--results_dir",
        default=None,
        help="run ディレクトリ (省略時は runvault が最新の run を答える)",
    )
    parser.add_argument(
        "--results-root", "--results_root", default="results",
        help="run ディレクトリの置き場 (default: results)",
    )
    parser.add_argument(
        "--subcommand", default="play",
        help="--results-dir 省略時に選ぶサブコマンド (default: play)",
    )
    parser.add_argument(
        "--output-dir", "--output_dir", default=None,
        help="図の保存先 (default: run ディレクトリの隣の figures/<run_slug>)",
    )
    args = parser.parse_args(argv)

    if args.results_dir is None:
        try:
            results_dir = runvault_path(
                EXPERIMENT,
                results_root=args.results_root,
                subcommand=args.subcommand,
                standalone=True,
            )
        except Exception as e:  # runvault バイナリが無い / 終わった run が無い
            print(f"エラー: run を解決できません: {e}", file=sys.stderr)
            return 1
    else:
        results_dir = args.results_dir

    # スイープ結果なら visualize_sweep に委譲
    if _is_sweep_parent(results_dir):
        from logistello_tools.visualize_sweep import main as sweep_main

        print("sweep の親 run を検出 → visualize-sweep に委譲します")
        return sweep_main(["--results-dir", results_dir]
                          + (["--output-dir", args.output_dir]
                             if args.output_dir else []))

    config = _load_run_config(results_dir)
    metrics = _load_metrics(results_dir)
    if config is None and metrics is None:
        print(
            f"エラー: {results_dir} に config.json / metrics.csv が"
            " 見つかりません",
            file=sys.stderr,
        )
        return 1

    out_dir = args.output_dir or figures_dir(results_dir)
    os.makedirs(out_dir, exist_ok=True)
    print("=== Logistello 単発実行 可視化 ===")
    print(f"結果: {results_dir}")
    print(f"出力先: {out_dir}")
    save_run_summary(config, metrics, os.path.join(out_dir, "run_summary.png"))
    print("完了．")
    return 0


if __name__ == "__main__":
    sys.exit(main())
