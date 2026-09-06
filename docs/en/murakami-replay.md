[English](murakami-replay.md) | [日本語](../ja/murakami-replay.md)

# Murakami-1997 replay — the gold validation set

The 1997 Takeshi Murakami vs. Logistello match (the historical 6-0 sweep)
is the reproduction's **gold validation set** (design doc §4.5 B5, §5). The
raw FFO WThor database is third-party and gitignored (`data/`); the
*derived* 6-game JSON is tiny and committed
(`tests/data/murakami_1997.json`).

## 1. Fetch the raw WThor DB

`scripts/fetch_wthor.sh` fetches the raw WThor 1997 base + the shared
player/tournament name DBs into the gitignored `data/wthor/` (idempotent;
`FORCE=1` to re-download).

```bash
bash scripts/fetch_wthor.sh
```

## 2. Extract the 6 gold games

`murakami-extract` parses every `.wtb` in `--wthor-dir`, resolves player ids
via the sibling `.jou`, keeps games where one player is "Logistello" and the
other "Murakami" (case-insensitive), and writes the tiny committed JSON gold
set. It reports the count + names found honestly (expects 6 for 1997).

```bash
cargo run --release -p logistello-cli -- murakami-extract \
    --wthor-dir data/wthor --output tests/data/murakami_1997.json
```

| Flag | Meaning | Default |
|---|---|---|
| `--wthor-dir` | Directory with the FFO `.wtb` + `WTHOR.JOU` | `data/wthor` |
| `--output` | Output JSON (the committed gold set) | `tests/data/murakami_1997.json` |

## 3. Replay & report the move-match rate

`match-replay` replays a recorded match and reports our engine's
move-match rate (overall + a "main positions" ply window, design doc §4.3.8
`move_match_rate_murakami`, §5 ≥80% on main positions).

```bash
# Medium-scale weights (see evaluation-training.md for training):
cargo run --release -p logistello-cli -- match-replay \
    --games tests/data/murakami_1997.json \
    --eval-weights results/eval_weights/wthor_medium.lgw1 \
    --depth 8
```

| Flag | Meaning | Default |
|---|---|---|
| `--games` | The committed gold-set JSON (`murakami-extract` output) | `tests/data/murakami_1997.json` |
| `--eval-weights` | `LGW1` weights for `PatternEval`; omit to use `BasicEval` | — |
| `--book` | Optional learned opening book (`OPB1`) | — |
| `--depth` | Selective-midgame iterative-deepening depth (plies) | `8` |
| `--endgame-empties` | Exact-endgame switch threshold (empties; B8) | `20` |
| `--mpc-params` | Optional Multi-ProbCut params JSON (a `probcut-fit` output) | — |
| `--main-lo` / `--main-hi` | Inclusive "main position" ply window (design doc §5) | `10` / `50` |
| `--output` | Optional extra copy of the decision table as a CSV, outside the run | — |

The invocation is recorded as a runvault run (under `--results-root`,
default `results`). Every scored decision is an `observation` event
(`unit_id` = the game, `t` = the ply); the aggregates are run-scope metrics
in `metrics.csv` (`scored_decisions`, `matched`, `overall_match_rate`,
`main_decisions`, `main_matched`, `main_match_rate`, `n_units`). The gold
games, the learned weights, the book and the MPC coefficients go into
`config.json`'s `data` **by content hash** — a path is not a condition, so
the same file in a different place is still the same condition.

The run draws no random numbers (recorded games are replayed
deterministically), so its domain is `analysis`. `--output` is a copy for
whoever wants the flat table; it lives outside the run and is not the
record.

> **Caveat (honest):** the committed convenience runs use *medium-scale*
> WThor-trained weights (a few-hundred-game corpus, minutes of training),
> not the production Logistello-2 corpus. Move-match numbers from medium
> weights are weaker than the §5 ≥80% / historical target — that bar is for
> production-scale training. See
> [full-scale reproduction](full-scale-reproduction.md) for the production
> commands.
