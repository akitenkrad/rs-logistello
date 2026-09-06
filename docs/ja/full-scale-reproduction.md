[English](../en/full-scale-reproduction.md) | [日本語](full-scale-reproduction.md)

# フルスケール再現（重い．ユーザ実行）

> **これらのコマンドはテストスイートや限定デモでは意図的に実行されない．**
> 各々が多数の CPU 時間を要する．設計ドキュメント §5 の論文値を再現する
> ときに自分で実行すること．他ドキュメントの限定／中規模の簡易実行は
> 数百局コーパスを使い，§5 目標より弱いのは*仕様どおり*である．§5 ≥80%
> ／歴史的水準は本番規模学習向けである．

## フル WThor コーパス学習

FFO が公開する全基準年 1977..=2023:

```bash
for y in $(seq 1977 2023); do WTHOR_YEAR=$y bash scripts/fetch_wthor.sh; done
cargo run --release -p logistello-cli -- extract --source wthor \
    --wthor-dir data/wthor --games 1000000 \
    --output results/wthor_full.pex1
uv run logistello-tools train-eval \
    --positions results/wthor_full.pex1 \
    --output results/eval_weights/wthor_full.lgw1
```

## 本番規模重みでのフル村上戦リプレイ

本番規模重み（+ `probcut-fit --mpc-cascade` の任意の Multi-ProbCut
パラメータ，[ProbCut チューニング](probcut-tuning.md)を参照）で:

```bash
cargo run --release -p logistello-cli -- match-replay \
    --games tests/data/murakami_1997.json \
    --eval-weights results/eval_weights/wthor_full.lgw1 \
    --depth 12 --mpc-params results/mpc_params.json
```

## Edax 比フル Elo スイープ

設計ドキュメント §5.1: レベル 5,10,15,20 × 30 局（Edax セットアップは
[Edax と Elo](edax-elo.md)を参照）:

```bash
cargo run --release -p logistello-cli -- elo-vs-edax \
    --edax-path .edax/edax --edax-levels 5,10,15,20 \
    --num-games-per-level 30 \
    --eval-weights results/eval_weights/wthor_full.lgw1 \
    --depth 12
```

## フル §6 感度スイープ

設計ドキュメント §6: 条件あたり 30 試行，Edax ベースは ≥50．1 呼び出し
1 §6 軸（設計上の注記は[感度分析スイープ](sensitivity-sweep.md)を参照）:

```bash
# §5.1 例: ProbCut-T グリッド，条件あたり 30 独立試行．
cargo run --release -p logistello-cli -- sweep \
    --probcut-t-min 1.0 --probcut-t-max 2.5 --probcut-t-step 0.25 \
    --runs 30 --seed 42

# 他の全 §6 行，1 呼び出しに 1 つ，--runs 30:
cargo run --release -p logistello-cli -- sweep \
    --probcut-depth-pairs-values 1:5,3:7,5:9,3:9,5:11 --runs 30 --seed 42
cargo run --release -p logistello-cli -- sweep \
    --multi-stages-values 1,2,3,4,5 --runs 30 --seed 42
cargo run --release -p logistello-cli -- sweep \
    --eval-stages-values 1,5,10,13,20,30 --runs 30 --seed 42
cargo run --release -p logistello-cli -- sweep \
    --glem-max-order-values 1,2,3,4 --runs 30 --seed 42
cargo run --release -p logistello-cli -- sweep \
    --glem-support-min -4 --glem-support-max -2 --glem-support-step 0.25 \
    --runs 30 --seed 42
cargo run --release -p logistello-cli -- sweep \
    --max-depth-values 6,8,10,12 --runs 30 --seed 42
cargo run --release -p logistello-cli -- sweep \
    --tt-size-values 20,22,24,26 --runs 30 --seed 42
cargo run --release -p logistello-cli -- sweep \
    --book-depth-values 12,18,24,30 --runs 30 --seed 42
cargo run --release -p logistello-cli -- sweep \
    --endgame-empties-values 8,10,16,20,24 --runs 30 --seed 42
cargo run --release -p logistello-cli -- sweep \
    --drawishness-min 0.0 --drawishness-max 0.5 --drawishness-step 0.1 \
    --runs 30 --seed 42

# Edax ベース条件は --runs 50 を使う（設計ドキュメント §6 の統計的信頼性
# 要件）．Edax 対エンジンのスイープメトリクスが配線されたら．
uv run logistello-tools visualize-sweep
```
