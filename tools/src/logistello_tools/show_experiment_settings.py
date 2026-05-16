"""実験設定値の表示．

汎用モード:
    --results-dir <path>  既存実行の `config.json` (run) / `sweep_config.json` (sweep)
                          を読んで整形表示する．`results/latest` シンボリックリンクも解決．

論文固有モード（必要に応じて追加）:
    引数なし時に `paper_experiments()` 等の論文再現実験定義を一覧表示する．
    本テンプレートでは未実装．`reproduce_paper.py` を作成したら，
    そこから `Experiment` / `paper_experiments()` をインポートして拡張すること．

Usage:
    logistello-tools show-experiment-settings --results-dir results/latest
    logistello-tools show-experiment-settings --results-dir results/20260101_120000 --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _resolve_results_dir(arg: str) -> Path:
    """相対パスは CWD 起点で解決し，シンボリックリンクは実体に解決する．"""
    p = Path(arg)
    if not p.is_absolute():
        p = Path.cwd() / arg
    return Path(os.path.realpath(p))


def _find_config_file(results_dir: Path) -> tuple[Path, str]:
    run_cfg = results_dir / "config.json"
    sweep_cfg = results_dir / "sweep_config.json"
    if run_cfg.exists():
        return run_cfg, "run"
    if sweep_cfg.exists():
        return sweep_cfg, "sweep"
    raise FileNotFoundError(
        f"設定ファイルが見つかりません: {results_dir}\n"
        f"  期待ファイル: config.json (run) / sweep_config.json (sweep)"
    )


def render_config(cfg: dict, source: Path, kind: str) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"実行設定 ({kind})")
    lines.append("=" * 80)
    lines.append(f"設定ファイル: {source}")
    lines.append("-" * 80)
    for key, value in cfg.items():
        lines.append(f"{key:<20}: {value}")
    lines.append("=" * 80)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="logistello-tools show-experiment-settings",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", "--results_dir",
        required=True,
        help="config.json / sweep_config.json を含む実行結果ディレクトリ",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="表ではなく JSON 形式で出力する．",
    )
    args = parser.parse_args(argv)

    results_dir = _resolve_results_dir(args.results_dir)
    if not results_dir.exists():
        print(f"エラー: ディレクトリが存在しません: {results_dir}", file=sys.stderr)
        return 1

    cfg_path, kind = _find_config_file(results_dir)
    with cfg_path.open() as f:
        cfg = json.load(f)

    if args.json:
        payload = {"source": str(cfg_path), "kind": kind, "config": cfg}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(render_config(cfg, cfg_path, kind))
    return 0


if __name__ == "__main__":
    sys.exit(main())
