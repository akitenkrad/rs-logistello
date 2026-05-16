"""スイープ結果の可視化スクリプト（テンプレート骨格）．

`results/<timestamp>_sweep/sweep_summary.csv` を読み込み，パラメータと最終メトリクス
の関係（1D 折れ線 / 2D ヒートマップ）と，各組み合わせ毎のグリッド進行アニメーション
を生成する．

論文固有のメトリクス・配色・ラベルは各実装で埋める．参考実装は `replications/schelling1971/`．

## 出力（想定）

```
{output_dir}/
├── sweep_*.png                ← パラメータと最終メトリクスの関係図（1D/2D）
└── sweep_grid_animation.gif   ← パラメータ組み合わせ別のグリッド進行アニメーション
                                  （sweep を `--snapshot-interval N` (N>0) 付きで
                                  実行した場合のみ生成される）
```

## グリッドアニメーションの設計指針

- 各セルが1つのパラメータ組み合わせの run のスナップショットを再生する合成 GIF．
- 2D スイープ: 行=2nd-axis, 列=1st-axis に並べる（例: 行=probcut_t, 列=depth）．
- 1D スイープ: 横一列〜矩形に折りたたんで配置（例: 4 列までは 1 行, 超えたら折り返し）．
- 複数 seed が含まれる場合は `--grid-seed` で1つを選ぶ（既定: 先頭シード）．
- 収束ステップ数の異なる run は最終フレームを保持して同期する．
- スナップショットが無い run は空セルとして描画し，全 run でスナップショット欠落時は
  警告のみ出力してスキップする（描画失敗で全体を落とさない）．

Usage:
    logistello-tools visualize-sweep [--sweep-dir RESULTS_DIR] [--output-dir OUT]
                                     [--no-grid-animation]
                                     [--grid-seed SEED] [--fps FPS] [--max-frames N]
"""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="logistello-tools visualize-sweep",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sweep-dir", "--sweep_dir",
        default="results/latest",
        help="スイープ結果のディレクトリ (default: results/latest)",
    )
    parser.add_argument(
        "--output-dir", "--output_dir",
        default=None,
        help="図の保存先ディレクトリ (default: {sweep_dir}/figures)",
    )
    parser.add_argument(
        "--no-grid-animation", "--no_grid_animation",
        action="store_true",
        help="パラメータ組み合わせ別グリッドアニメーションの生成をスキップする",
    )
    parser.add_argument(
        "--grid-seed", "--grid_seed",
        type=int,
        default=None,
        help="グリッドアニメーションで使用する seed (default: 先頭シード)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="グリッドアニメーションの FPS (default: 5)",
    )
    parser.add_argument(
        "--max-frames", "--max_frames",
        type=int,
        default=0,
        help="グリッドアニメーションの最大フレーム数 (0=全フレーム)",
    )
    args = parser.parse_args(argv)

    # TODO: implement sweep visualization
    #   1. sweep_summary.csv をロード
    #   2. sweep_config.json を任意で参照（サブタイトル等）
    #   3. 1D / 2D を自動判別し，sweep_*.png を保存
    #   4. args.no_grid_animation が False なら，各セル run の snapshots/ を読み込んで
    #      合成 GIF (sweep_grid_animation.gif) を生成
    #   参考実装: replications/schelling1971/tools/src/schelling_tools/visualize_sweep.py
    print(
        "TODO: visualize-sweep "
        f"(sweep_dir={args.sweep_dir}, output_dir={args.output_dir}, "
        f"no_grid_animation={args.no_grid_animation}, grid_seed={args.grid_seed}, "
        f"fps={args.fps}, max_frames={args.max_frames})"
    )


if __name__ == "__main__":
    main()
