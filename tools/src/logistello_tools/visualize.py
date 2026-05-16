"""可視化スクリプト．

Usage:
    logistello-tools visualize [--results-dir RESULTS_DIR]
"""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="logistello-tools visualize")
    parser.add_argument(
        "--results-dir",
        default="results/latest",
        help="可視化対象の結果ディレクトリ (default: results/latest)",
    )
    args = parser.parse_args(argv)
    print(f"TODO: visualize {args.results_dir}")


if __name__ == "__main__":
    main()
