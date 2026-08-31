"""logistello-tools — 統合 CLI ディスパッチャ．

Usage:
    logistello-tools wthor-extract [...]
    logistello-tools train-eval [...]
    logistello-tools train-glem [...]
    logistello-tools objective5 [...]
    logistello-tools visualize [...]
    logistello-tools visualize-sweep [...]
    logistello-tools show-experiment-settings [...]
"""
from __future__ import annotations

import argparse
import sys


def _wthor_extract(argv: list[str]) -> None:
    # Phase 4b: position extraction is owned by the Rust CLI (it owns the
    # B4 canonicalisation). Use `logistello extract --source wthor
    # --wthor-dir DIR --output FILE` (selfplay source needs no data); see
    # crates/logistello-eval/EXTRACT_FORMAT.md.
    print(
        "wthor-extract: extraction is performed by the Rust CLI "
        "(`cargo run -p logistello-cli -- extract --source wthor "
        "--wthor-dir DIR --output FILE`); see EXTRACT_FORMAT.md"
    )


def _train_eval(argv: list[str]) -> None:
    from logistello_tools.train_eval import main as run_main

    run_main(argv)


def _train_glem(argv: list[str]) -> None:
    from logistello_tools.train_glem import main as run_main

    run_main(argv)


def _objective5(argv: list[str]) -> None:
    from logistello_tools.objective5 import main as run_main

    run_main(argv)


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
    subparsers.add_parser(
        "objective5",
        help="Objective-5: GLEM 自動特徴量 vs 手動 PatternEval 比較 (Phase 7)",
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
        help="run の条件を表示 (runvault の run / 移行前の results ディレクトリ)",
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
    elif command == "objective5":
        _objective5(rest)
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
