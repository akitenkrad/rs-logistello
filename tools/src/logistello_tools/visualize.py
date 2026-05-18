"""単一実行（`run` / `play`）結果の可視化（Phase 10; 設計ドキュメント §4.2）．

Rust CLI が単発実行で書き出す `results/<ts>/{config.json, metrics.csv}` を
読み，設定の要約と単発メトリクスの棒グラフを生成する．スイープ結果
（`sweep_config.json` を含む）が指定された場合は `visualize_sweep` に委譲する．

```
{output_dir}/
└── run_summary.png   ← 設定テキスト + メトリクス棒グラフ
```

Usage:
    logistello-tools visualize [--results-dir DIR] [--output-dir OUT]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # ヘッドレス / pytest

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLOR_BG = "#FAFAF8"
COLOR_BAR = "#1565C0"


def _load_run_config(results_dir: str) -> dict | None:
    path = os.path.join(results_dir, "config.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_run_summary(config: dict | None, df: pd.DataFrame | None,
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
        ax.text(0.5, 0.5, "(config.json なし)", ha="center", va="center",
                transform=ax.transAxes, color="#999")
    ax.set_title("実行設定 (config.json)")

    # 右: 数値メトリクス棒グラフ
    ax = axes[1]
    ax.set_facecolor(COLOR_BG)
    if df is not None and not df.empty:
        row = df.iloc[0]
        num = {
            k: float(v)
            for k, v in row.items()
            if isinstance(v, (int, float))
            or (isinstance(v, str) and _is_num(v))
        }
        if num:
            keys = list(num.keys())
            ax.bar(range(len(keys)), [num[k] for k in keys],
                   color=COLOR_BAR, alpha=0.85)
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")
        ax.set_title("メトリクス (metrics.csv)")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "(metrics.csv なし)", ha="center", va="center",
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
        default="results/latest",
        help="可視化対象の結果ディレクトリ (default: results/latest)",
    )
    parser.add_argument(
        "--output-dir", "--output_dir", default=None,
        help="図の保存先 (default: {results_dir}/figures)",
    )
    args = parser.parse_args(argv)

    results_dir = args.results_dir
    # スイープ結果なら visualize_sweep に委譲
    if os.path.exists(os.path.join(results_dir, "sweep_config.json")):
        from logistello_tools.visualize_sweep import main as sweep_main

        print("sweep_config.json を検出 → visualize-sweep に委譲します")
        return sweep_main(["--results-dir", results_dir]
                          + (["--output-dir", args.output_dir]
                             if args.output_dir else []))

    metrics_path = os.path.join(results_dir, "metrics.csv")
    config = _load_run_config(results_dir)
    df = None
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)

    if config is None and df is None:
        print(
            f"エラー: {results_dir} に config.json / metrics.csv が"
            " 見つかりません",
            file=sys.stderr,
        )
        return 1

    out_dir = args.output_dir or os.path.join(results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    print("=== Logistello 単発実行 可視化 ===")
    print(f"結果: {results_dir}")
    print(f"出力先: {out_dir}")
    save_run_summary(config, df, os.path.join(out_dir, "run_summary.png"))
    print("完了．")
    return 0


if __name__ == "__main__":
    sys.exit(main())
