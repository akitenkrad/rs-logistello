"""GOLD GLEM interop fixture generator (Phase 7, run-once helper — NOT a
test).

Produces the two fixtures the Rust GOLD interop test
(``tests/glem_interop_test.rs``) consumes:

1. ``tests/data/glem_fixture.glm1`` — a ``GLM1`` model written **by
   Python** from a documented deterministic feature set + weight
   construction (see below). The Rust test builds the *same* construction,
   serializes it, asserts byte-identity, AND round-trips Python's file.

2. ``tests/data/glem_fixture_positions.json`` — for a deterministic GLX1
   corpus (``glem-extract --base-features cell64,corner --source selfplay
   --games 10 --seed 4242``, extracted by the Rust CLI), Python's own
   ``GlemEval``-equivalent prediction
   ``round(Σ satisfied w / 128)`` (Edax bias + ±63 clamp) per position.
   The Rust test re-extracts the same corpus, loads the GLM1 into
   ``GlemEval`` and asserts equality → proves the base-literal extractor +
   conjunction eval + GLM1 contract agree across languages.

The feature set is a small, fully-deterministic set of order-1 and order-2
conjunctions over the ``cell64,corner`` spec (no training randomness — the
GOLD test is about the *contract*, not the fitted numbers). cell64 owns
literal ids ``0..192``, corner owns ``192..204`` (n_literals = 204):

    features =
      [ (0,), (191,), (200,),                      # order-1
        (5, 191), (0, 196), (12, 203) ]            # order-2
    w[stage][f] = ((stage*37 + f*53 + 11) mod 401) - 200

Usage::

    uv run python -m logistello_tools._gen_glem_fixture \\
        --glx1 PATH --out-dir tests/data
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from logistello_tools.glem import predict
from logistello_tools.glm1 import N_STAGES, GlemModel
from logistello_tools.glx1 import load_glx1

FIXTURE_SEED_PARAMS = {
    "base_features": "cell64,corner",
    "source": "selfplay",
    "games": 10,
    "seed": 4242,
}

FIXTURE_FEATURES = [
    [0],
    [191],
    [200],
    [5, 191],
    [0, 196],
    [12, 203],
]


def fixture_weight(stage: int, f: int) -> int:
    return ((stage * 37 + f * 53 + 11) % 401) - 200


def build_fixture_model() -> GlemModel:
    n_feat = len(FIXTURE_FEATURES)
    w = np.zeros((N_STAGES, n_feat), dtype=np.int64)
    for s in range(N_STAGES):
        for f in range(n_feat):
            w[s, f] = fixture_weight(s, f)
    return GlemModel(
        family_names=["cell64", "corner"],
        features=[list(f) for f in FIXTURE_FEATURES],
        w=w,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gen_glem_fixture")
    parser.add_argument("--glx1", required=True, help="input GLX1 corpus")
    parser.add_argument("--out-dir", required=True, help="tests/data dir")
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    model = build_fixture_model()
    glm1_path = os.path.join(args.out_dir, "glem_fixture.glm1")
    model.save(glm1_path)

    ex = load_glx1(args.glx1)
    preds = [int(v) for v in predict(model, ex)]
    meta = {
        "construction": (
            "features = [(0,),(191,),(200,),(5,191),(0,196),(12,205)]; "
            "w[s][f] = ((s*37 + f*53 + 11) mod 401) - 200"
        ),
        "extract": FIXTURE_SEED_PARAMS,
        "n_positions": ex.n_records,
        "predictions": preds,
    }
    json_path = os.path.join(args.out_dir, "glem_fixture_positions.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    print(
        f"wrote {glm1_path} ({os.path.getsize(glm1_path)} bytes) and "
        f"{json_path} ({ex.n_records} positions)"
    )


if __name__ == "__main__":
    main()
