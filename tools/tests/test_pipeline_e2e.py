"""End-to-end Phase 4b pipeline test (no external deps, seeded).

`logistello extract --source selfplay` (Rust) -> `train_eval.train`
(faithful GD-300) -> `EvalWeights` -> evaluate on a **held-out** self-play
set and assert the trained model's Pearson correlation with the true
terminal disc differential is meaningfully positive AND beats the all-zeros
baseline (which has zero/undefined correlation).

The model evaluation here uses the *same* canonical-sum + Edax bias rounding
the Rust `PatternEval` uses (proven byte-identical by the Rust GOLD interop
test), so this is a faithful surrogate for "load LGW1 into PatternEval and
measure correlation" with zero external dependencies.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from logistello_tools._gen_fixture import edax_round_clamp
from logistello_tools.lgw1 import EvalWeights
from logistello_tools.pex1 import Extract, load_pex1
from logistello_tools.train_eval import train

REPO = Path(__file__).resolve().parents[2]


def _find_cli() -> str | None:
    for prof in ("release", "debug"):
        p = REPO / "target" / prof / "logistello"
        if p.exists():
            return str(p)
    return shutil.which("logistello")


def _extract(cli: str, out: Path, games: int, seed: int) -> None:
    subprocess.run(
        [
            cli,
            "extract",
            "--source",
            "selfplay",
            "--games",
            str(games),
            "--seed",
            str(seed),
            "--output",
            str(out),
        ],
        check=True,
        capture_output=True,
    )


def _predict(ex: Extract, w: EvalWeights) -> np.ndarray:
    """Vectorised round(sum(canonical_w)/128) with Edax bias rounding."""
    # Map (feature -> type_slot) is constant; gather per-position raw sums.
    raw = np.zeros(ex.n_records, dtype=np.int64)
    for i in range(ex.canon.shape[1]):
        ty = int(ex.feat_type[i])
        col = ex.canon[:, i]
        # Per stage, gather weights[stage][ty][col].
        for s in range(13):
            m = ex.stage == s
            if np.any(m):
                raw[m] += w.w[s][ty][col[m]]
    return np.array([edax_round_clamp(int(v)) for v in raw], dtype=np.float64)


@pytest.mark.skipif(
    _find_cli() is None,
    reason="logistello CLI binary not built (run `cargo build`)",
)
def test_e2e_trained_beats_zeros_correlation(tmp_path):
    cli = _find_cli()
    assert cli is not None

    train_bin = tmp_path / "train.pex1"
    test_bin = tmp_path / "test.pex1"
    # Disjoint seeds -> disjoint deterministic corpora (train vs held-out).
    _extract(cli, train_bin, games=300, seed=1)
    _extract(cli, test_bin, games=60, seed=999)

    weights = train(str(train_bin), method="gd")
    lgw1 = tmp_path / "w.lgw1"
    weights.save(str(lgw1))
    # Round-trips through the LGW1 contract.
    reloaded = EvalWeights.load(str(lgw1))

    held = load_pex1(str(test_bin))
    y_true = held.label.astype(np.float64)

    pred_trained = _predict(held, reloaded)
    zeros = EvalWeights()  # all-zero baseline
    pred_zeros = _predict(held, zeros)

    # Zeros eval is identically 0 -> no signal.
    assert np.allclose(pred_zeros, 0.0)
    # Variation in the true label exists (sanity).
    assert y_true.std() > 1.0

    r_trained = np.corrcoef(pred_trained, y_true)[0, 1]
    # Concrete, non-flaky, seeded threshold: the faithful GD model must
    # capture a clear positive relationship on held-out self-play data and
    # is, by construction, far better than the zero-signal baseline.
    assert r_trained > 0.30, f"trained Pearson r too low: {r_trained:.4f}"
    assert pred_trained.std() > 0.0, "trained model must produce signal"

    print(
        f"\n[e2e] held-out Pearson r: trained={r_trained:.4f} "
        f"zeros=undefined(0 signal) "
        f"n_test={held.n_records} n_train_pos="
        f"{load_pex1(str(train_bin)).n_records}"
    )


if __name__ == "__main__":  # manual run convenience
    sys.exit(pytest.main([os.path.abspath(__file__), "-s", "-q"]))
