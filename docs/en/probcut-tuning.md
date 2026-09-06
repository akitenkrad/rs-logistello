[English](probcut-tuning.md) | [日本語](../ja/probcut-tuning.md)

# ProbCut & Multi-ProbCut tuning

> **What ProbCut is (honest):** ProbCut / Multi-ProbCut are **unsound
> forward prunes**. They are *statistical*, not exact: a shallow probe
> search at depth `d` predicts the deep value at depth `h` via an OLS model
> `v_h = a·v_d + b` (with residual σ), and a per-σ confidence `T` decides
> whether to cut. Cuts can therefore be wrong — this trades exactness for
> speed, exactly as in Buro's papers. The fit quality (R²) depends on the
> evaluator: with the trivial `BasicEval` R² ≈ 0.83, but with the learned
> pattern evaluator the design-doc §4.5 B7 target is R² > 0.96 (B7).

## `probcut-fit` — fit `(a, b, σ)` coefficients

For every sampled position `probcut-fit` measures the true `v_x =
NegaScout(x)` for each needed depth under the production TT discipline
(ProbCut/MPC OFF), stratifies by disc phase (`< 36` vs `≥ 36`), and fits OLS
`v_h = a·v_d + b` per group.

```bash
# Single ProbCut: one (d, h) pair (design doc §4.5 B7 single ProbCut = 4:8)
cargo run --release -p logistello-cli -- probcut-fit \
    --single-pair 4:8 --samples 5000 --seed 42 \
    --output results/probcut_params.json

# Multi-ProbCut cascade: an independent OLS per (disc-phase, h, d) cell
cargo run --release -p logistello-cli -- probcut-fit \
    --mpc-cascade 3:1,4:2,5:1,6:2,7:3,8:4,9:3:5,10:4:6,11:3:5,12:4,13:5 \
    --samples 5000 --seed 42 --output results/mpc_params.json
```

| Flag | Meaning | Default |
|---|---|---|
| `--single-pair` | Shallow:deep pair `d:h` (used when `--mpc-cascade` absent) | `4:8` |
| `--mpc-cascade` | Cascade spec `h:d1[:d2],...`; runs the `(disc-phase, h, d)` fit instead | — |
| `--source` | `selfplay` (seeded, no external data) or `wthor` | `selfplay` |
| `--wthor-dir` | Directory with `.wtb` files (required for `--source wthor`) | — |
| `--samples` | Number of sample positions to fit on | `5000` |
| `--seed` | Self-play RNG seed (deterministic) | `42` |
| `--eval-weights` | Optional `LGW1`: fit with `PatternEval` instead of `BasicEval` (B7 R² > 0.96) | — |
| `--probcut-t` | Per-σ confidence `T` written into the single `ProbCutConfig` (ignored for `--mpc-cascade`, which carries the canonical 2-phase 1.0 / 1.4) | `1.5` |
| `--output` | Output JSON (`ProbCutConfig` or `MultiProbCutConfig`) | required |

## Using the fitted params in `play` / `bench-search`

A `--probcut` / `--mpc` run is only meaningful with the matching fitted
params JSON (the built-in default has no fitted coefficients, so ProbCut
just falls through):

```bash
# Single ProbCut
cargo run --release -p logistello-cli -- play \
    --black engine --white random --depth 8 \
    --probcut --probcut-params results/probcut_params.json

# Multi-ProbCut (supersedes single ProbCut at the cascade heights 3..=13)
cargo run --release -p logistello-cli -- bench-search \
    --depth 10 --mpc --mpc-params results/mpc_params.json

# Observe the speedup vs exact search
cargo run --release -p logistello-cli -- bench-search --depth 10 --speedup \
    --probcut-params results/probcut_params.json \
    --mpc-params results/mpc_params.json
```

See [getting started](getting-started.md) for the full `play` /
`bench-search` flag tables.
