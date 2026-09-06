[English](../../README.md) | [日本語](README.md)

# logistello

Michael Buro の Othello プログラム **Logistello**（Buro 1994-1999）の，
Rust + Python による忠実な再現実装．

## 概要

Logistello は長年世界最強の Othello プログラムであり，1997 年には人間の世界
チャンピオン村上健に **6-0** で勝利した．本プロジェクトは Buro の一連の論文から
その中核アイデアを再現する．

再現する中核アイデアは次のとおり．棋譜から学習する統計的パターン評価関数
（Buro 1995 JAIR / 1997 NEC TR），選択的 α-β 枝刈り **ProbCut** /
**Multi-ProbCut**（Buro 1995 ICCA / 1997），一般化線形評価モデル **GLEM**
（Buro 1998 CG'98），オープニングブック学習（Buro 1999）．

本実装はゼロからのエンジン開発ではなく**再現研究**である．bitboard・合法手
生成・WTHOR 入出力・Edax 連携は再利用可能ライブラリ
**[`rs-othello-sim`](https://github.com/akitenkrad/rs-othello-sim)**（git
依存・リビジョン固定）に委譲し，Logistello 固有の貢献（Zobrist ハッシュ，
NegaScout/PVS 探索，パターン評価，ProbCut/Multi-ProbCut，定石学習）を本
リポジトリの `crates/` に実装する．

## インストール

前提: **Rust 1.85+ / edition 2024**，Python ツール用に
[`uv`](https://docs.astral.sh/uv/)，および（任意で）`scripts/setup_edax.sh`
による Edax v4.6．

本リポジトリは
[`social-simulation-replications`](https://github.com/akitenkrad/social-simulation-replications)
の Submodule なので，Submodule 込みで取得する．

```bash
git clone --recurse-submodules <repo-url>
# 既存クローン内なら:
git submodule update --init
```

ビルド・Python セットアップ・サニティチェック（リポジトリルートから実行）．

```bash
cargo build --release
uv sync
cargo run --release -p logistello-cli -- perft --depth 6   # => perft(depth=6) = 8200
```

`perft --depth 6` が `8200` を出力すれば，再利用した合法手生成とローカル
perft がエンドツーエンドで正しく接続されていることを確認できる．

## ドキュメント

ユースケース別の詳細ガイド（各ドキュメントは EN/JA バイリンガル）．

- [はじめに](getting-started.md) — ビルド，`perft`，`play`，`bench-search`，`selfplay`．
- [評価関数の学習](evaluation-training.md) — `extract` + Python `train-eval` → 学習済み `PatternEval`．
- [ProbCut チューニング](probcut-tuning.md) — `probcut-fit`，単一 ProbCut と Multi-ProbCut．
- [GLEM 特徴量](glem-features.md) — `glem-extract` + `train-glem` 自動生成連言特徴．
- [定石学習](opening-book.md) — `learn-book` 自己対局による定石学習．
- [村上戦リプレイ](murakami-replay.md) — 1997 年 6-0 のゴールド検証セット．
- [Edax と Elo](edax-elo.md) — Edax v4.6 オラクル，`elo-vs-edax`，`eval-correlation-edax`．
- [感度分析スイープ](sensitivity-sweep.md) — `sweep` + 可視化．
- [フルスケール再現](full-scale-reproduction.md) — 重い，ユーザ実行の論文規模コマンド．

データ形式リファレンス仕様:
[`WEIGHTS_FORMAT.md`](../../crates/logistello-eval/WEIGHTS_FORMAT.md)，
[`EXTRACT_FORMAT.md`](../../crates/logistello-eval/EXTRACT_FORMAT.md)，
[`GLEM_FORMAT.md`](../../crates/logistello-eval/GLEM_FORMAT.md)，
[`BOOK_FORMAT.md`](../../crates/logistello-book/BOOK_FORMAT.md)，
[`EDAX_SETUP.md`](../../EDAX_SETUP.md)．

## ステータス

探索，パターン評価，ProbCut/Multi-ProbCut，GLEM，定石学習，および
村上／Edax 検証ハーネスを含むパイプライン全体が，エンドツーエンドで
実装・検証済みである．コミットされた簡易実行は
*中規模* の WThor 学習重みを使うため，着手一致率や Elo は設計ドキュメント §5
／歴史的目標より弱い．フルスケール再現は意図的にユーザ実行とする
（[フルスケール再現](full-scale-reproduction.md) を参照）．

## 参考文献

- Buro, M. (1995). "Statistical Feature Combination for the Evaluation of Game Positions." *JAIR* 3: 373-382.
- Buro, M. (1995). "ProbCut: An Effective Selective Extension of the αβ Algorithm." *ICCA Journal* 18(2): 71-76.
- Buro, M. (1997). "An Evaluation Function for Othello Based on Statistics." NEC Research Institute TR 31.
- Buro, M. (1997). "Experiments with Multi-ProbCut and a New High-Quality Evaluation Function for Othello." NEC Research Institute TR.
- Buro, M. (1997). "The Othello Match of the Year: Takeshi Murakami vs. Logistello." *ICCA Journal* 20(3): 189-193.
- Buro, M. (1998). "From Simple Features to Sophisticated Evaluation Functions." In *Computers and Games (CG'98)*, Springer LNCS 1558, 126-145.
- Buro, M. (1999). "Toward Opening Book Learning." *ICCA Journal* 22(2): 98-102.
- Buro, M. (2002). "Improving Heuristic Mini-Max Search by Supervised Learning." *Artificial Intelligence* 134: 85-99.
- Delorme, R. (2003-). *Edax — A Strong Othello/Reversi Program.* Logistello の後継オープンソース実装．本実装ではオラクルとして使用．
- Fédération Française d'Othello. *WTHOR Database.* 評価関数学習に用いる世界選手権棋譜アーカイブ．

## ライセンス

MIT．[LICENSE](../../LICENSE) を参照．
