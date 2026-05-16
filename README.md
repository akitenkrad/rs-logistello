# replication-logistello

A faithful Rust + Python reproduction of Michael Buro's **Logistello** Othello
program (Buro, 1994-1999).

## Purpose

Logistello was, for years, the strongest Othello program in the world; in 1997
it defeated the human world champion Takeshi Murakami 6-0. This project
reproduces its key ideas from Buro's series of papers:

- statistical pattern-based evaluation functions learned from game records
  (Buro 1995 JAIR; 1997 NEC TR);
- **ProbCut** and **Multi-ProbCut** selective alpha-beta pruning
  (Buro 1995 ICCA; 1997);
- **GLEM** generalized linear evaluation models (Buro 1998 CG'98);
- opening-book learning (Buro 1999).

This is a reproduction study, not a from-scratch engine: the bitboard,
move generation, WTHOR I/O, and Edax integration come from the reusable
**[`rs-othello-sim`](https://github.com/akitenkrad/rs-othello-sim)** library,
which this project depends on as a pinned git dependency. The Logistello-specific
contributions (Zobrist hashing, NegaScout/PVS search, pattern evaluation,
ProbCut/Multi-ProbCut, opening-book learning) live in the `crates/` here.

## Workspace structure

```
replication-logistello/
├── Cargo.toml                  Rust virtual workspace
├── pyproject.toml              Python (uv) workspace
├── crates/
│   ├── logistello-core/        Zobrist hashing, perft, board constants
│   ├── logistello-eval/        pattern evaluation (inference side)
│   ├── logistello-search/      NegaScout/PVS, TT, killers, (Multi-)ProbCut
│   ├── logistello-book/        opening-book learning + drawishness
│   └── logistello-cli/         unified CLI binary `logistello`
├── tools/
│   └── src/logistello_tools/   Python: WTHOR extract, eval/GLEM training,
│                               visualization, experiment settings
├── tests/
│   ├── perft_test.rs           known-value perft integration test
│   └── data/                   fixtures
└── results/                    run outputs (gitignored)
```

`Cargo.lock` is committed (this is an application/binary workspace).

## Build & run

All commands are run from the repository root.

### Rust

```bash
cargo build --release
cargo run --release -p logistello-cli -- perft --depth 6
cargo test
```

`perft --depth 6` enumerates the search tree from the standard starting
position and should print `8200` (the standard Othello perft value), confirming
the reused move generator and the local perft are wired correctly end-to-end.

### Python

```bash
uv sync
uv run logistello-tools --help
```

The Python tools (`wthor-extract`, `train-eval`, `train-glem`, `visualize`,
`visualize-sweep`, `show-experiment-settings`) are scaffolded as stubs and will
be filled in per implementation phase.

## Status

Project scaffold. Only `perft` is wired end-to-end; every other Rust subcommand
and Python tool is a Phase-tagged placeholder.

## References

- Buro, M. (1995). "Statistical Feature Combination for the Evaluation of Game
  Positions." *Journal of Artificial Intelligence Research* 3: 373-382.
- Buro, M. (1995). "ProbCut: An Effective Selective Extension of the αβ
  Algorithm." *ICCA Journal* 18(2): 71-76.
- Buro, M. (1997). "An Evaluation Function for Othello Based on Statistics."
  NEC Research Institute Technical Report 31, Princeton, NJ.
- Buro, M. (1997). "Experiments with Multi-ProbCut and a New High-Quality
  Evaluation Function for Othello." NEC Research Institute Technical Report.
- Buro, M. (1997). "The Othello Match of the Year: Takeshi Murakami vs.
  Logistello." *ICCA Journal* 20(3): 189-193.
- Buro, M. (1998). "From Simple Features to Sophisticated Evaluation
  Functions." In *Computers and Games (CG'98)*, Springer LNCS 1558, 126-145.
- Buro, M. (1999). "Toward Opening Book Learning." *ICCA Journal* 22(2): 98-102.
- Buro, M. (2002). "Improving Heuristic Mini-Max Search by Supervised
  Learning." *Artificial Intelligence* 134: 85-99.
- Delorme, R. (2003-). *Edax — A Strong Othello/Reversi Program.*
  Direct open-source successor to Logistello; used here as an oracle.
- Fédération Française d'Othello. *WTHOR Database.* World-championship game
  archive used for evaluation-function learning.

## License

MIT. See [LICENSE](LICENSE).

---
# replication-logistello（日本語）

Michael Buro の Othello プログラム **Logistello** (Buro 1994-1999) の，
Rust + Python による忠実な再現実装．

## 目的

Logistello は長年世界最強の Othello プログラムであり，1997 年には人間の世界
チャンピオン村上健に 6-0 で勝利した．本プロジェクトは Buro の一連の論文から
その中核アイデアを再現する．

- 棋譜から学習する統計的パターン評価関数（Buro 1995 JAIR / 1997 NEC TR）
- 選択的 α-β 枝刈り **ProbCut** / **Multi-ProbCut**（Buro 1995 ICCA / 1997）
- 一般化線形評価モデル **GLEM**（Buro 1998 CG'98）
- オープニングブック学習（Buro 1999）

本実装はゼロからのエンジン開発ではなく**再現研究**である．bitboard・合法手
生成・WTHOR 入出力・Edax 連携は再利用可能ライブラリ
**[`rs-othello-sim`](https://github.com/akitenkrad/rs-othello-sim)**（git
依存・リビジョン固定）に委譲し，Logistello 固有の貢献（Zobrist ハッシュ，
NegaScout/PVS 探索，パターン評価，ProbCut/Multi-ProbCut，定石学習）を本
リポジトリの `crates/` に実装する．

## ワークスペース構成

英語節のツリーを参照．`Cargo.lock` はコミットする（アプリケーション
ワークスペースのため）．

## ビルド & 実行

全コマンドはリポジトリルートから実行する．

```bash
# Rust
cargo build --release
cargo run --release -p logistello-cli -- perft --depth 6   # => 8200
cargo test

# Python
uv sync
uv run logistello-tools --help
```

`perft --depth 6` は標準初期局面からの探索木を列挙し，標準 Othello perft 値
`8200` を出力する．これにより，再利用した合法手生成とローカル perft が
エンドツーエンドで正しく接続されていることを確認できる．

## ステータス

プロジェクトの骨格．エンドツーエンドで動作するのは `perft` のみ．他の Rust
サブコマンドおよび Python ツールは Phase 番号付きのプレースホルダ．

## ライセンス

MIT．[LICENSE](LICENSE) を参照．

---
*This file was generated by Claude Code.*
