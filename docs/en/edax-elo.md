[English](edax-elo.md) | [日本語](../ja/edax-elo.md)

# Edax oracle, Elo & eval correlation

Edax v4.6 is the fixed-strength external opponent / non-playing oracle. It
is a platform binary plus multi-MB evaluation weights and is therefore
**never committed** (`.edax/` is gitignored). The Edax-driving subcommands
**skip and print a notice** (no fabrication) when `.edax/` is absent, so the
build stays green on machines without Edax.

## Edax setup

```bash
bash scripts/setup_edax.sh   # builds Edax v4.6 + fetches eval.dat into .edax/
```

The script is idempotent and self-contained. Full reproducible steps are in
[`EDAX_SETUP.md`](../../EDAX_SETUP.md). Two honest findings are documented
there:

- **macOS `aligned_alloc` fix:** Edax's hash-table allocation calls C11
  `aligned_alloc` with a size that is not a multiple of the alignment;
  macOS's `aligned_alloc` strictly rejects that (glibc is lenient). The
  setup script applies the upstream-style `adjust_size` patch automatically
  and idempotently; it is harmless on Linux.
- **GTP driver finding:** the pinned `rs-othello-sim` `GtpProtocol` /
  `ExternalEnginePlayer` path is unusable with the pinned revision (it
  re-`play`s Edax's own move with the wrong colour). Edax is therefore
  driven via a **direct-GTP `EdaxGtpSession`** that never re-plays Edax's
  own move (the `? wrong color` fix).

## `elo-vs-edax` — estimate Elo vs Edax

`elo-vs-edax` estimates Elo versus Edax at various levels via the direct-GTP
full-game driver (design doc §4.3.8 `elo_vs_edax_level_N`). The Elo delta is
`Δ = -400·log10(1/p - 1)` from the observed score `p`. It is **bounded by
default** (levels 1,3 × 2 games); the full sweep is documented in
[full-scale reproduction](full-scale-reproduction.md). The Edax
level→absolute strength mapping is only qualitative (§5).

```bash
cargo run --release -p logistello-cli -- elo-vs-edax \
    --edax-levels 1,3 --num-games-per-level 2 \
    --eval-weights results/eval_weights/wthor_medium.lgw1
```

| Flag | Meaning | Default |
|---|---|---|
| `--edax-path` | Path to the Edax binary | `.edax/edax` |
| `--edax-levels` | Comma list of Edax `-level N` strengths | `1,3` |
| `--num-games-per-level` | Games per level (colour-balanced; rounded up to even) | `2` |
| `--eval-weights` | `LGW1` weights for our `PatternEval` (else `BasicEval`) | — |
| `--depth` | Our engine's selective-midgame depth (kept small for the demo) | `6` |
| `--endgame-empties` | Our engine's exact-endgame switch threshold (empties) | `16` |
| `--seed` | Recorded, but decides nothing (see below) | `42` |
| `--output` | Optional extra copy of the per-level table as a CSV, outside the run | — |

The invocation is recorded as a runvault run. One level is one
`x.logistello.edax_level` event (the old `elo_vs_edax.csv` row: `games`,
`wins`, `draws`, `losses`, `score_rate`, `win_rate`, `elo_delta`), and
`metrics.csv` carries only `n_units` — the number of observed units, here
levels. Edax goes into `data` as **two** content hashes, the binary and its
`eval.dat` weights: change either and the numbers change.

`--seed` is accepted and recorded but **decides nothing**: our engine is
deterministic and Edax runs single-threaded with its book off. It is
therefore kept out of `config_hash` (so two runs of one condition stay one
condition), and the domain is `analysis` rather than `simulation` — a run
that draws no random numbers should not claim a master seed.

## `eval-correlation-edax` — correlate our eval against Edax

`eval-correlation-edax` reports the Pearson correlation of our `PatternEval`
value vs Edax's evaluation on a bounded position set (design doc §4.3.8
`eval_correlation_edax`, Edax as ground truth).

```bash
cargo run --release -p logistello-cli -- eval-correlation-edax \
    --edax-level 6 --positions 24 \
    --eval-weights results/eval_weights/wthor_medium.lgw1
```

| Flag | Meaning | Default |
|---|---|---|
| `--edax-path` | Path to the Edax binary | `.edax/edax` |
| `--edax-level` | Edax fixed strength used for its `genmove`-based evaluation | `6` |
| `--eval-weights` | `LGW1` weights for our `PatternEval` (else `BasicEval`) | — |
| `--positions` | Number of positions to sample (bounded) | `24` |
| `--seed` | Seed for the deterministic position walk | `42` |
| `--output` | Optional extra copy of the sample table as a CSV, outside the run | — |

One sampled position is one `x.logistello.eval_sample` event (`our_score`,
`edax_score`); Pearson's `r` and the sample count `n_units` are run-scope
metrics. `r` is **undefined** for `n < 2` or a flat series, and an
undefined value is not zero, so that row is simply not written. The sampled
positions come from a seeded random walk, so this run's domain is
`simulation` with `master_seed` = `--seed`.

> **Caveat (honest):** the committed convenience runs use *medium-scale*
> WThor-trained weights, so the Elo numbers are weaker than the §5 /
> historical targets. See
> [full-scale reproduction](full-scale-reproduction.md).

---
*This file was generated by Claude Code.*
