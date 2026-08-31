[English](getting-started.md) | [日本語](../ja/getting-started.md)

# Getting started

All commands are run from the repository root. See the
[root README](../../README.md) for installation.

## Build & test

```bash
cargo build --release
cargo test            # runs tests/perft_test.rs
cargo clippy
cargo fmt --check
```

## `perft` — correctness anchor

`perft` enumerates the search tree from the standard starting position and
verifies it against known Othello perft values. It is the correctness anchor
for the reused move generator.

```bash
cargo run --release -p logistello-cli -- perft --depth 6   # => perft(depth=6) = 8200
```

| Flag | Meaning | Default |
|---|---|---|
| `--depth` | Plies to enumerate from the standard starting position | `6` |

## `play` — play a single game

`play` plays a single game between two players to the terminal. Each side is
one of `engine` (Logistello: basic eval + iterative deepening + exact
endgame), `random`, or `greedy`. It is deterministic for a given seed (the
random opponent is seeded; the engine itself is deterministic).

```bash
cargo run --release -p logistello-cli -- play \
    --black engine --white random --depth 6 --seed 42
```

| Flag | Meaning | Default |
|---|---|---|
| `--black` / `--white` | `engine`, `random`, or `greedy` | `engine` / `random` |
| `--depth` | Selective-midgame iterative-deepening max depth (plies) | `8` |
| `--endgame-empties` | Switch to exact endgame once empties ≤ this (B8; Edax-parity 10) | `20` |
| `--seed` | Seed for the random opponent | `42` |
| `--eval-weights` | Optional `LGW1` learned weights → `PatternEval` (see [evaluation training](evaluation-training.md)) | — |
| `--glem-model` | Optional `GLM1` GLEM model → `GlemEval` (see [GLEM features](glem-features.md)) | — |
| `--probcut` / `--no-probcut` / `--probcut-t` / `--probcut-params` | Single ProbCut (see [ProbCut tuning](probcut-tuning.md)) | off / `1.5` |
| `--mpc` / `--mpc-params` | Multi-ProbCut (see [ProbCut tuning](probcut-tuning.md)) | off |
| `--book` | Optional learned opening book `OPB1` (see [opening book](opening-book.md)) | — |

Without `--eval-weights`, `play` uses the built-in `BasicEval` (disc count +
mobility); with it, the engine uses the learned Edax-style `PatternEval`. A
single full game is recorded as a runvault run and can be visualised with
`visualize` (see [sensitivity sweep](sensitivity-sweep.md)).

## What a run records (runvault)

The subcommands that **measure** something (`play`, `bench-search`,
`match-replay`, `elo-vs-edax`, `eval-correlation-edax`, `sweep`) record the
invocation into a [runvault](https://github.com/akitenkrad/rs-runvault) run
directory, `<results-root>/logistello/<run-slug>/`. runvault owns the
placement and the naming, so nothing here creates a timestamped directory or
a `latest` link. `--results-root` (default `results`) is a global flag.

| File | What it holds |
|---|---|
| `run.json` | experiment, subcommand, domain, seed, lineage, the paper being reproduced |
| `config.json` | the conditions (`parameters`) and the content hashes of the files that decide the result (`data`) |
| `metrics.csv` | numbers: the quantities that describe the whole run with one value each (long form) |
| `events.jsonl` | the rows that cannot be metrics — observations identified by a series, and labels such as an outcome |
| `status.json` | state and elapsed time (`duration_sec`, the record of time) |

The subcommands that only **produce data or an artefact** (`extract`,
`glem-extract`, `probcut-fit`, `murakami-extract`, `learn-book`, `perft`)
create no run. What they write is an input to a later run, and that run
carries it in `config.json`'s `data` by content hash — because the contents,
not the path, decide the result.

```bash
# Read a run's conditions / draw its figures (latest run when --results-dir is omitted)
uv run logistello-tools show-experiment-settings
uv run logistello-tools visualize
```

## `bench-search` — search benchmark

`bench-search` benchmarks the search engine from the standard opening using
the trivial disc-difference evaluator. `--probcut` / `--mpc` run the same
nominal depth with single ProbCut / Multi-ProbCut enabled so the speedup vs
the exact search is observable (`probcut_speedup`).

```bash
cargo run --release -p logistello-cli -- bench-search --depth 8
cargo run --release -p logistello-cli -- bench-search --depth 8 --speedup
```

| Flag | Meaning | Default |
|---|---|---|
| `--depth` | Maximum iterative-deepening depth (plies) | `8` |
| `--probcut` / `--no-probcut` / `--probcut-t` / `--probcut-params` | Single ProbCut at the same nominal depth | off / `1.5` |
| `--mpc` / `--mpc-params` | Multi-ProbCut at the same nominal depth | off |
| `--speedup` | Full-vs-single-ProbCut-vs-MPC comparison at `--depth` (the §5 comparisons) | off |
| `--from-plies` | Plies of seeded random play before benchmarking (0 = standard opening) | `0` |
| `--seed` | Seed for the `--from-plies` random walk | `42` |
| `--glem-model` | Optional `GLM1` GLEM model as the leaf evaluator | — |

## `selfplay` — self-play games

`selfplay` runs self-play games for data generation. It takes no options; for
the training-position pipeline use `extract` (see
[evaluation training](evaluation-training.md)).

```bash
cargo run --release -p logistello-cli -- selfplay
```

---
*This file was generated by Claude Code.*
