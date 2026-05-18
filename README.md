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

### Phase 4b: evaluation-weight training pipeline

The evaluation function is trained from games end-to-end: Rust extracts
training positions (it owns the Edax-faithful B4 canonicalisation), Python
fits the per-stage linear model (design doc §4.4 B3), and Rust loads the
learned weights for play. The on-disk weight format is `LGW1`, a fully
explicit little-endian layout that Rust and Python write/read **byte-
identically** (see `crates/logistello-eval/WEIGHTS_FORMAT.md`); the extract
format is the columnar `PEX1` (`crates/logistello-eval/EXTRACT_FORMAT.md`).

```bash
# 1. Extract training positions.
#    a) self-play (deterministic, no external data — the test vehicle):
cargo run --release -p logistello-cli -- extract \
    --source selfplay --games 300 --seed 1 --output /tmp/pos.bin
#    b) OR real WTHOR expert games (manual download, design doc §4.5 B5):
#       curl -L -o WTH_2004.ZIP \
#         https://www.ffothello.org/wthor/base_zip/WTH_2004.ZIP
#       unzip WTH_2004.ZIP -d data/wthor/
cargo run --release -p logistello-cli -- extract \
    --source wthor --wthor-dir data/wthor/ --output /tmp/pos.bin

# 2. Train the per-stage linear eval (faithful B3 GD-300 by default;
#    --method ridge|sgd for the scikit-learn §7 comparison paths).
uv run logistello-tools train-eval \
    --positions /tmp/pos.bin --output /tmp/w.lgw1 --method gd

# 3. Play using the learned pattern evaluator.
cargo run --release -p logistello-cli -- play \
    --black engine --white random --depth 4 --seed 42 \
    --eval-weights /tmp/w.lgw1
```

Without `--eval-weights`, `play` uses the Phase 3 `BasicEval` (disc count +
mobility); with it, the engine uses the learned Edax-style `PatternEval`.

### Python

```bash
uv sync
uv run logistello-tools --help
uv run logistello-tools train-eval --positions FILE --output FILE.lgw1
```

`train-eval` is implemented (Phase 4b). The remaining Python tools
(`train-glem`, `visualize`, `visualize-sweep`, `show-experiment-settings`)
are scaffolded as stubs and will be filled in per implementation phase;
`wthor-extract` is intentionally delegated to the Rust `extract` subcommand
(Rust owns the B4 canonicalisation).

### Edax external engine (optional; Phase 9)

Edax v4.6 is the fixed-strength external opponent / non-playing oracle. It is
a platform binary plus multi-MB evaluation weights and is therefore **never
committed** (`.edax/` is gitignored). To build it and fetch the weights
locally:

```bash
bash scripts/setup_edax.sh   # builds Edax v4.6 + fetches eval.dat into .edax/
```

The script is idempotent and self-contained; full reproducible steps,
the macOS portability patch, and the GTP driver protocol are documented in
[`EDAX_SETUP.md`](EDAX_SETUP.md). Without `.edax/`, the Edax integration
tests (`cargo test --test edax_smoke_test`,
`cargo test --test murakami_replay_test`) skip and pass, so the build
stays green on machines without Edax.

### Phase 9b: Murakami-1997 replay, Edax-Elo & eval-correlation

The 1997 Takeshi Murakami vs. Logistello match (the historical 6-0 sweep)
is the reproduction's **gold validation set** (design doc §4.5 B5, §5).
The raw FFO WThor database is third-party and gitignored (`data/`); the
*derived* 6-game JSON is tiny and committed
(`tests/data/murakami_1997.json`).

```bash
# 1. Fetch the raw WThor 1997 base + the shared player/tournament name DBs
#    into the gitignored data/wthor/ (idempotent; FORCE=1 to re-download).
bash scripts/fetch_wthor.sh

# 2. Filter the six Logistello-vs-Murakami games into the committed gold
#    JSON (reports the count + names honestly; expects 6 for 1997).
cargo run --release -p logistello-cli -- murakami-extract \
    --wthor-dir data/wthor --output tests/data/murakami_1997.json

# 3. Train medium-scale real-WThor pattern weights (bounded; see §commands
#    below for the documented full-scale corpus).
cargo run --release -p logistello-cli -- extract --source wthor \
    --wthor-dir data/wthor --games 400 --max-empties-skip 50 \
    --output results/wthor_medium.pex1
uv run logistello-tools train-eval \
    --positions results/wthor_medium.pex1 \
    --output results/eval_weights/wthor_medium.lgw1

# 4. Replay the 6 gold games and report the move-match rate (overall +
#    "main positions" ply window, design doc §4.3.8 / §5 ≥80% target).
cargo run --release -p logistello-cli -- match-replay \
    --games tests/data/murakami_1997.json \
    --eval-weights results/eval_weights/wthor_medium.lgw1 \
    --depth 8 --output results/murakami_replay.csv

# 5. Estimate Elo vs Edax (BOUNDED demo: levels 1,3 × 2 games). The
#    Edax level→absolute strength mapping is only qualitative (§5).
cargo run --release -p logistello-cli -- elo-vs-edax \
    --edax-levels 1,3 --num-games-per-level 2 \
    --eval-weights results/eval_weights/wthor_medium.lgw1 \
    --output results/elo_vs_edax.csv

# 6. Correlate our static eval against Edax's solved value (bounded).
cargo run --release -p logistello-cli -- eval-correlation-edax \
    --edax-level 6 --positions 24 \
    --eval-weights results/eval_weights/wthor_medium.lgw1 \
    --output results/eval_correlation_edax.csv
```

`match-replay` / `elo-vs-edax` / `eval-correlation-edax` also mirror their
CSV into a timestamped `results/<YYYYMMDD_HHMMSS>/` directory with a
`results/latest` symlink (design doc §4.2). The Edax-driving subcommands
**skip and print a notice** (no fabrication) when `.edax/` is absent.

> **Caveat (honest):** the committed convenience runs use *medium-scale*
> WThor-trained weights (a few-hundred-game corpus, minutes of training),
> not the production Logistello-2 corpus. Move-match and Elo numbers from
> medium weights are weaker than the §5 ≥80% / historical targets — that
> bar is for production-scale training. The full-scale commands are below.

### Phase 9 / full-scale reproduction (heavy; user-run)

These reproduce the design-doc §5 targets at full scale. They are
**deliberately not run by the test suite or the bounded demo** (each is
many CPU-hours); run them yourself when reproducing the paper numbers:

```bash
# Full WThor corpus: every base year 1977..=2023 the FFO publishes.
for y in $(seq 1977 2023); do WTHOR_YEAR=$y bash scripts/fetch_wthor.sh; done
cargo run --release -p logistello-cli -- extract --source wthor \
    --wthor-dir data/wthor --games 1000000 \
    --output results/wthor_full.pex1
uv run logistello-tools train-eval \
    --positions results/wthor_full.pex1 \
    --output results/eval_weights/wthor_full.lgw1

# Full Murakami replay with the production-scale weights (+ optional
# Multi-ProbCut params from `probcut-fit --mpc-cascade`).
cargo run --release -p logistello-cli -- match-replay \
    --games tests/data/murakami_1997.json \
    --eval-weights results/eval_weights/wthor_full.lgw1 \
    --depth 12 --mpc-params results/probcut_params.json \
    --output results/murakami_replay.csv

# Full Elo sweep vs Edax (design doc §5.1: levels 5,10,15,20 × 30 games).
cargo run --release -p logistello-cli -- elo-vs-edax \
    --edax-path .edax/edax --edax-levels 5,10,15,20 \
    --num-games-per-level 30 \
    --eval-weights results/eval_weights/wthor_full.lgw1 \
    --depth 12 --output results/elo_vs_edax.csv
```

## Status

Phases 0-9b are wired end-to-end: `perft`, full game `play` (basic /
learned pattern / GLEM evaluators), Phase 4b position `extract`
(self-play / WTHOR) + the Python `train-eval` linear-regression pipeline
(cross-language byte-identical `LGW1`), `probcut-fit` (single +
Multi-ProbCut), `learn-book` (Phase 8), the Edax v4.6 integration, and
Phase 9b's `murakami-extract` / `match-replay` / `elo-vs-edax` /
`eval-correlation-edax` (the design-doc §4.3.8 metrics). The remaining
Python visualisation tools and `sweep` (Phase 10) are Phase-tagged
placeholders.

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
