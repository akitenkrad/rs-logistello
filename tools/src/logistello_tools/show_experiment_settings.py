"""実験設定値の表示．

runvault の run ディレクトリから `run.json` (どの実験のどのサブコマンドか) と
`config.json` の `parameters` (条件そのもの) を読んで整形表示する．

どの run を見るかは `--results-dir` を省略すれば runvault が答える
(`runvault path --experiment logistello --latest`)．`results/` を自分で走査して
新しそうなディレクトリを当てにいくことはしない．

移行前の `results/<YYYYMMDD_HHMMSS>/` (平坦な `config.json` / `sweep_config.json`
を持つ) も `--results-dir` に直接渡せば従来どおり読める．

Usage:
    logistello-tools show-experiment-settings
    logistello-tools show-experiment-settings --subcommand sweep
    logistello-tools show-experiment-settings --results-dir results/logistello/<run-slug>
    logistello-tools show-experiment-settings --results-dir results/20260101_120000 --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from runvault.read import config_parameters, load_run_meta, runvault_path

#: runvault 上の実験名 (Rust 側 `record::EXPERIMENT` と同じ)．
EXPERIMENT = "logistello"


def _resolve_results_dir(arg: str) -> Path:
    """相対パスは CWD 起点で解決し，シンボリックリンクは実体に解決する．"""
    p = Path(arg)
    if not p.is_absolute():
        p = Path.cwd() / arg
    return Path(os.path.realpath(p))


def _legacy_sweep_config(results_dir: Path) -> dict | None:
    """移行前の `sweep_config.json`（runvault 以前の sweep 出力）．"""
    path = results_dir / "sweep_config.json"
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return None


def load_settings(results_dir: Path) -> tuple[dict, str, Path]:
    """`(設定, 種別, 読んだファイル)` を返す．

    runvault の run なら `config.json` の `parameters`．種別は `run.json` の
    `subcommand`．どちらも無い移行前のディレクトリは，平坦な `config.json`
    または `sweep_config.json` をそのまま読む．
    """
    meta = load_run_meta(results_dir, required=False)
    parameters = config_parameters(results_dir, required=False)
    if meta is not None and parameters is not None:
        return parameters, str(meta["subcommand"]), results_dir / "config.json"
    if parameters is not None:
        return parameters, "run", results_dir / "config.json"
    legacy = _legacy_sweep_config(results_dir)
    if legacy is not None:
        return legacy, "sweep", results_dir / "sweep_config.json"
    raise FileNotFoundError(
        f"設定ファイルが見つかりません: {results_dir}\n"
        f"  期待ファイル: config.json (runvault の run / 移行前の run) / "
        f"sweep_config.json (移行前の sweep)"
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
        default=None,
        help="run ディレクトリ (省略時は runvault が最新の run を答える)",
    )
    parser.add_argument(
        "--results-root", "--results_root", default="results",
        help="run ディレクトリの置き場 (default: results)",
    )
    parser.add_argument(
        "--subcommand", default=None,
        help="--results-dir 省略時に絞り込むサブコマンド (play / sweep / ...)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="表ではなく JSON 形式で出力する．",
    )
    args = parser.parse_args(argv)

    if args.results_dir is None:
        try:
            results_dir = Path(
                runvault_path(
                    EXPERIMENT,
                    results_root=args.results_root,
                    subcommand=args.subcommand,
                )
            )
        except Exception as e:  # runvault バイナリが無い / 終わった run が無い
            print(f"エラー: run を解決できません: {e}", file=sys.stderr)
            return 1
    else:
        results_dir = _resolve_results_dir(args.results_dir)

    if not results_dir.exists():
        print(f"エラー: ディレクトリが存在しません: {results_dir}", file=sys.stderr)
        return 1

    try:
        cfg, kind, cfg_path = load_settings(results_dir)
    except FileNotFoundError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1

    if args.json:
        payload = {"source": str(cfg_path), "kind": kind, "config": cfg}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(render_config(cfg, cfg_path, kind))
    return 0


if __name__ == "__main__":
    sys.exit(main())
