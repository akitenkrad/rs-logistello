[English](../en/sensitivity-sweep.md) | [日本語](sensitivity-sweep.md)

# 感度分析（`sweep`）+ 可視化

`sweep` は設計ドキュメント §6 の感度分析を実行する．**1 つ**の §6
パラメータをそのグリッド（両端含む `min/max/step`，GLEM サポートは対数
スケール，または明示的な値リスト）に展開し，値ごとに `--runs` の独立した
**シード付き**試行を実行する．結果は runvault が
`--results-root`（既定 `results`）の下に置く:

- **親 run**（サブコマンド `sweep`） — 解決済みのグリッドそのものを
  `config.json` の `parameters` に持つ．親自身は何も測らない．
- **条件ごとの子 run**（サブコマンド `sweep-point`） — その値を
  `parameters` に持ち，試行 1 本を `events.jsonl` の
  `x.logistello.trial` イベント 1 行として，条件の平均を `metrics.csv`
  の run スコープ指標（`mean_<metric>` と `n_units`）として持つ．

試行は時間軸を持たない 1 回の測定なので，1 行で言い尽くせる（子 run に
割る必要がない）．掃引していない軸の列は書かない — 測っていない量を 0 と
して書くと，後から «測って 0 だった» と区別できなくなる．

## 設計上の注記

- **1 実行 1 軸．** §6 表はパラメータごとに 1 行である．`sweep` は
  パラメータごとに 1 回呼ぶ必要がある．パラメータフラグなしでは
  スイープ可能なパラメータを列挙し非ゼロ終了する．
- **実時間は記録しない．** 試行ごとの RNG シードは
  `--seed + condition_index*runs + trial_index` として明示的に導出
  （実時間は使わない）するので，測定値は反復実行で同一である．実行時間は
  `status.json` の `duration_sec` が正本なので指標にはしない．

| パラメータ（`--flag`） | メトリクス | §6 期待方向 |
|---|---|---|
| `--probcut-t-min/max/step` (1.0–2.5/0.25) | 探索 `nodes` | `T` 大 ⇒ 枝刈り少 ⇒ ノード増 |
| `--probcut-depth-pairs-values d:h,…` | 探索 `nodes` | `d` 浅 ⇒ 緩く安価なプローブ |
| `--multi-stages-values 1..5` | 探索 `nodes` | `k≈3` が経験的最適 |
| `--eval-stages-values 1,5,10,13,20,30` | `eval_abs_err` | ≥5 段で急減，≥13 で飽和 |
| `--glem-max-order-values 1..4` | `n_features` | 2 次でほぼ捕捉，4 次は過学習 |
| `--glem-support-min/max/step` (1e-4–1e-2 対数) | `n_features` | τ 高 ⇒ 特徴少 |
| `--max-depth-values 6,8,10,12` | 探索 `nodes` | 深さに対し概ね指数的に増加 |
| `--tt-size-values 20,22,24,26` (=2^k) | 探索 `nodes` | TT 大 ⇒ ノード少（≥2^24 で飽和） |
| `--book-depth-values 12,18,24,30` | `book_positions` | 深 ⇒ 定石が大きく安定 |
| `--endgame-empties-values 8,10,16,20,24` | 探索 `nodes` | 大 ⇒ 厳密終盤が早く開始 ⇒ ノード増 |
| `--drawishness-min/max/step` (0.0–0.5/0.1) | `selfplay_score` | `λ` が自己対局結果を動かす |

`--runs` は条件あたりの独立シード付き試行数（設計ドキュメント §6: 30;
Edax ベース条件 ≥50）．`--seed` は基底 RNG シード（既定 `42`）．

## 小規模デモ（軽量．いつでも安全に実行可）

```bash
cargo run --release -p logistello-cli -- sweep \
    --endgame-empties-values 10,20 --runs 3 --seed 42

# どの run を見るかは runvault が答える（--results-dir 省略時）
uv run logistello-tools show-experiment-settings --subcommand sweep
uv run logistello-tools visualize-sweep
#   -> results/logistello/figures/<run_slug>/{sweep_<metric>.png,
#      sweep_overview.png, sweep_grid_animation.gif}
```

1 ゲームの実行も run として記録されるので，`visualize` で可視化できる．

```bash
cargo run --release -p logistello-cli -- play \
    --black engine --white random --depth 6 --seed 42
uv run logistello-tools visualize          # 最新の play run
```

## Python 可視化ツール

| ツール | 目的 |
|---|---|
| `visualize [--subcommand play]` | 単発 run の可視化（条件 + run スコープ指標） |
| `visualize-sweep` | スイープのパラメータ対メトリクス図 + 条件別グリッドアニメ（`--fps`，`--max-frames`，`--no-grid-animation`） |
| `show-experiment-settings [--subcommand …]` | 解決済みの条件を表示（`--json` で JSON） |

`--results-dir` を省略すると `runvault path --latest` が run を解決する
（`results/` を自分で走査しない）．図は run ディレクトリの *隣*
（`results/logistello/figures/<run_slug>/`）に置く — `manifest.csv` は
`finish()` が確定させたもので，後から足したものはそこに載らないためである．
移行前の `results/<YYYYMMDD_HHMMSS>/` は `--results-dir` に直接渡せば
従来どおり読める．フル §6 スイープ（`--runs 30`，Edax
≥50）は[フルスケール再現](full-scale-reproduction.md)に記載．
