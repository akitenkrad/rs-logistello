"""logistello-tools — 統合 CLI ディスパッチャ．

Usage:
    logistello-tools wthor-extract [...]
    logistello-tools train-eval [...]
    logistello-tools train-glem [...]
    logistello-tools visualize [...]
    logistello-tools visualize-sweep [...]
    logistello-tools show-experiment-settings [...]
"""
from __future__ import annotations

import argparse
import sys


def _wthor_extract(argv: list[str]) -> None:
    print("wthor-extract: not yet implemented (Phase 4)")


def _train_eval(argv: list[str]) -> None:
    print("train-eval: not yet implemented (Phase 4)")


def _train_glem(argv: list[str]) -> None:
    print("train-glem: not yet implemented (Phase 7)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="logistello-tools",
        description="Logistello reproduction data & visualization tools",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "wthor-extract",
        help="WTHOR データベースから学習データを抽出 (Phase 4)",
        add_help=False,
    )
    subparsers.add_parser(
        "train-eval",
        help="パターン評価関数のロジスティック/線形回帰学習 (13 stages, Phase 4)",
        add_help=False,
    )
    subparsers.add_parser(
        "train-glem",
        help="GLEM 特徴量自動生成 + 学習 (Phase 7)",
        add_help=False,
    )
    subparsers.add_parser("visualize", help="単一実行結果の可視化", add_help=False)
    subparsers.add_parser(
        "visualize-sweep",
        help="スイープ結果の可視化 (パラメータ依存図 + 組み合わせ別グリッドアニメーション)",
        add_help=False,
    )
    subparsers.add_parser(
        "show-experiment-settings",
        help="実行結果ディレクトリの設定値を表示 (config.json / sweep_config.json)",
        add_help=False,
    )

    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        parser.parse_args(argv)
        return

    command = argv[0]
    rest = argv[1:]
    if command == "wthor-extract":
        _wthor_extract(rest)
    elif command == "train-eval":
        _train_eval(rest)
    elif command == "train-glem":
        _train_glem(rest)
    elif command == "visualize":
        from logistello_tools.visualize import main as run_main
        run_main(rest)
    elif command == "visualize-sweep":
        from logistello_tools.visualize_sweep import main as run_main
        run_main(rest)
    elif command == "show-experiment-settings":
        from logistello_tools.show_experiment_settings import main as run_main
        run_main(rest)
    else:
        parser.parse_args(argv)  # 不正なサブコマンドはここでエラーになる


if __name__ == "__main__":
    main()
