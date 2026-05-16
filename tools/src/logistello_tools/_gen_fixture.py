"""GOLD interop fixture generator (Phase 4b, run-once helper — NOT a test).

Produces the two fixtures the Rust GOLD interop test
(``tests/eval_interop_test.rs``) consumes:

1. ``tests/data/fixture.lgw1`` — an ``LGW1`` file written **by Python** from
   a documented deterministic weight construction (see ``canon_weight``).
   The Rust test builds the *same* construction, serializes it, and asserts
   the bytes equal this file → proves Python-write == Rust-write byte for
   byte (and, with the Rust round-trip, Rust↔Python is byte-identical).

2. ``tests/data/fixture_positions.json`` — for a deterministic PEX1 corpus
   (selfplay, fixed seed/params, extracted by the Rust CLI), Python's own
   model prediction ``round(Σ canonical_w / 128)`` with Edax bias rounding +
   ``±63`` clamp, per position. The Rust test re-extracts the same corpus,
   loads ``fixture.lgw1`` into ``PatternEval`` and asserts equality →
   proves the B4 canonical mapping is consistent across languages.

Usage::

    uv run python -m logistello_tools._gen_fixture \\
        --pex1 PATH --out-dir tests/data
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from logistello_tools.lgw1 import N_STAGES, N_TYPES, EvalWeights
from logistello_tools.pex1 import load_pex1

# Deterministic, language-agnostic weight construction. The Rust GOLD test
# reproduces this exact formula; keep the two in lock-step.
#
#   w[stage][type_slot][canon] =
#       ((stage*1009 + type_slot*131 + canon*31 + 7) mod 521) - 260
#
# Range is [-260, 260], comfortably inside i32; sign varies across slots.
FIXTURE_SEED_PARAMS = {"source": "selfplay", "games": 12, "seed": 12345}


def canon_weight(stage: int, type_slot: int, canon: int) -> int:
    return ((stage * 1009 + type_slot * 131 + canon * 31 + 7) % 521) - 260


def build_fixture_weights() -> EvalWeights:
    w = EvalWeights()
    for s in range(N_STAGES):
        for t in range(N_TYPES):
            n = w.canon_sizes[t]
            idx = np.arange(n, dtype=np.int64)
            w.w[s][t] = (
                (s * 1009 + t * 131 + idx * 31 + 7) % 521
            ) - 260
    return w


def edax_round_clamp(raw_sum: int) -> int:
    """``round(sum/128)`` with Edax bias rounding + ±63 clamp (midgame.c)."""
    biased = raw_sum + 64 if raw_sum > 0 else raw_sum - 64
    score = int(biased // 128) if biased >= 0 else -int((-biased) // 128)
    return max(-63, min(63, score))


def predict(ex, weights: EvalWeights) -> list[int]:
    """Per-position ``round(Σ canonical_w / 128)`` (same as PatternEval)."""
    preds: list[int] = []
    for r in range(ex.n_records):
        st = int(ex.stage[r])
        raw = 0
        for i in range(ex.canon.shape[1]):
            ty = int(ex.feat_type[i])
            c = int(ex.canon[r, i])
            raw += int(weights.w[st][ty][c])
        preds.append(edax_round_clamp(raw))
    return preds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gen_fixture")
    parser.add_argument("--pex1", required=True, help="input PEX1 corpus")
    parser.add_argument("--out-dir", required=True, help="tests/data dir")
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    weights = build_fixture_weights()
    lgw1_path = os.path.join(args.out_dir, "fixture.lgw1")
    weights.save(lgw1_path)

    ex = load_pex1(args.pex1)
    preds = predict(ex, weights)
    meta = {
        "construction": (
            "w[s][t][c] = ((s*1009 + t*131 + c*31 + 7) mod 521) - 260"
        ),
        "extract": FIXTURE_SEED_PARAMS,
        "n_positions": ex.n_records,
        "predictions": preds,
    }
    json_path = os.path.join(args.out_dir, "fixture_positions.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    print(
        f"wrote {lgw1_path} ({os.path.getsize(lgw1_path)} bytes) and "
        f"{json_path} ({ex.n_records} positions)"
    )


if __name__ == "__main__":
    main()
