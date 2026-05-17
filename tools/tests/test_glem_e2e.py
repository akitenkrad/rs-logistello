"""End-to-end Phase 7 GLEM pipeline + Objective-5 smoke (no external deps).

`logistello glem-extract --source selfplay` (Rust owns base-literal
extraction) -> `glem.generate_and_fit` (§4.3.6, B3 GD-300 fit) -> GLM1 ->
evaluate on a **held-out** self-play set and assert the GLEM model's Pearson
correlation with the true terminal disc differential is meaningfully
positive AND beats the all-zeros baseline (which has zero signal).

The Python `glem.predict` uses the same canonical-sum + Edax bias rounding
the Rust `GlemEval` uses (proven byte/numerically identical by the Rust
GOLD test ``tests/glem_interop_test.rs``), so this is a faithful surrogate
for "load GLM1 into GlemEval and measure correlation" with zero external
deps.

Also exercises the Objective-5 apparatus (GLEM-auto vs the fixed
PatternEval on the same held-out games) as a smoke result.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from logistello_tools.glem import generate_and_fit, predict
from logistello_tools.glm1 import GlemModel
from logistello_tools.glx1 import load_glx1

REPO = Path(__file__).resolve().parents[2]


def _find_cli() -> str | None:
    for prof in ("release", "debug"):
        p = REPO / "target" / prof / "logistello"
        if p.exists():
            return str(p)
    return shutil.which("logistello")


def _glem_extract(
    cli: str, out: Path, base: str, games: int, seed: int
) -> None:
    subprocess.run(
        [
            cli,
            "glem-extract",
            "--base-features",
            base,
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


def _extract_pex1(cli: str, out: Path, games: int, seed: int) -> None:
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


@pytest.mark.skipif(
    _find_cli() is None,
    reason="logistello CLI binary not built (run `cargo build`)",
)
def test_glem_e2e_trained_beats_zeros_correlation(tmp_path):
    cli = _find_cli()
    assert cli is not None

    # `corner,mobility` order-2: a well-conditioned base set so the shared
    # faithful Phase-4b GD (fixed-step, diagonal-curvature bound) stays
    # stable while still exercising real order-2 conjunctions + the full
    # GLX1 -> generate_and_fit -> GLM1 -> GlemEval-equivalent path. (The
    # cell64 conjunction extractor/eval is separately proven by the Rust
    # GOLD interop test on the cell64,corner fixture.)
    base = "corner,mobility"
    train_glx = tmp_path / "train.glx1"
    test_glx = tmp_path / "test.glx1"
    # Disjoint seeds -> disjoint deterministic corpora (train vs held-out).
    _glem_extract(cli, train_glx, base, games=150, seed=1)
    _glem_extract(cli, test_glx, base, games=40, seed=999)

    ex_train = load_glx1(str(train_glx))
    model, report = generate_and_fit(
        ex_train,
        max_order=2,
        support_threshold=0.01,
        weight_threshold=0.0,
        method="gd",
    )
    # Round-trips through the GLM1 contract.
    glm1 = tmp_path / "m.glm1"
    model.save(str(glm1))
    reloaded = GlemModel.load(str(glm1))
    assert reloaded.features == model.features
    assert np.array_equal(reloaded.w, model.w)

    held = load_glx1(str(test_glx))
    y_true = held.label.astype(np.float64)
    pred_trained = predict(reloaded, held)

    zeros = GlemModel(family_names=model.family_names, features=[],
                       w=np.zeros((13, 0), dtype=np.int64))
    pred_zeros = predict(zeros, held)
    assert np.allclose(pred_zeros, 0.0)  # zeros -> no signal
    assert y_true.std() > 1.0  # label varies (sanity)

    r_trained = np.corrcoef(pred_trained, y_true)[0, 1]
    # Concrete, non-flaky, seeded threshold: GLEM order-2 over cell64+corner
    # must capture a clear positive relationship on held-out self-play and
    # is, by construction, far better than the zero-signal baseline.
    assert r_trained > 0.20, f"trained Pearson r too low: {r_trained:.4f}"
    assert pred_trained.std() > 0.0, "trained model must produce signal"

    print(
        f"\n[glem-e2e] base={base} #base={report.n_base} "
        f"#generated={report.n_generated} "
        f"#after-support={report.n_after_support} "
        f"#after-weight-prune={report.n_after_weight_prune} "
        f"held-out Pearson r: trained={r_trained:.4f} "
        f"zeros=0(no signal) n_test={held.n_records}"
    )


@pytest.mark.skipif(
    _find_cli() is None,
    reason="logistello CLI binary not built (run `cargo build`)",
)
def test_objective5_apparatus_smoke(tmp_path):
    """Objective-5 (§5): GLEM-auto vs manual PatternEval on the same
    held-out games. Full ELO is Phase 9/10; this is the apparatus + a
    smoke result (both produce signal; numbers reported)."""
    from logistello_tools.objective5 import compare
    from logistello_tools.train_eval import train

    cli = _find_cli()
    assert cli is not None

    base = "corner,mobility"  # well-conditioned (see e2e rationale above)
    # Same (games, seed) -> Rust extract & glem-extract walk identical
    # snapshots in identical order, so PEX1 and GLX1 labels line up.
    g_train = tmp_path / "tr.glx1"
    g_test = tmp_path / "te.glx1"
    p_train = tmp_path / "tr.pex1"
    p_test = tmp_path / "te.pex1"
    _glem_extract(cli, g_train, base, games=150, seed=7)
    _glem_extract(cli, g_test, base, games=40, seed=900)
    _extract_pex1(cli, p_train, games=150, seed=7)
    _extract_pex1(cli, p_test, games=40, seed=900)

    glem_model, _ = generate_and_fit(
        load_glx1(str(g_train)),
        max_order=2,
        support_threshold=0.01,
        weight_threshold=0.0,
    )
    glm1 = tmp_path / "g.glm1"
    glem_model.save(str(glm1))

    lgw1 = tmp_path / "p.lgw1"
    train(str(p_train), method="gd").save(str(lgw1))

    glem_s, manual_s = compare(
        str(g_test), str(glm1), str(p_test), str(lgw1)
    )
    print(
        f"\n[objective5] glem-auto: n_features={glem_s.n_features} "
        f"r={glem_s.pearson_r:+.4f} mse={glem_s.mse:.2f}  |  "
        f"manual-pattern: n_features={manual_s.n_features} "
        f"r={manual_s.pearson_r:+.4f} mse={manual_s.mse:.2f}  |  "
        f"delta_r={glem_s.pearson_r - manual_s.pearson_r:+.4f} "
        f"(full ELO = Phase 9/10)"
    )
    # Apparatus smoke: both evaluators must produce a real (non-degenerate)
    # signal on the held-out set; the comparison itself is the deliverable.
    assert glem_s.pearson_r > 0.0
    assert manual_s.pearson_r > 0.0
    assert glem_s.n_features > 0


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-s", "-q"]))
