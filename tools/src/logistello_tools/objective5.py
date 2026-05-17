"""Objective-5: GLEM auto-generated features vs the manual PatternEval
(design doc §5 "GLEM 自動特徴量 vs 手動", §3 Objective 5).

Reports, on the **same held-out corpus**, the held-out
correlation / MSE of:

* **GLEM-auto**  — a trained ``GLM1`` model scored on a ``GLX1`` extract;
* **manual**     — the fixed Phase-4 ``PatternEval`` (``LGW1``) scored on a
  ``PEX1`` extract of the *same games* (the Rust ``extract`` and
  ``glem-extract`` walk identical seeded self-play snapshots in the same
  order, so the label column lines up position-for-position).

Full ELO is Phase 9/10; this module is just the reproducible **apparatus**
plus a smoke result (correlation / MSE of each evaluator's prediction
against the true terminal disc differential). It is callable as
``logistello-tools objective5`` and from pytest.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from logistello_tools._gen_fixture import edax_round_clamp
from logistello_tools.glm1 import GlemModel
from logistello_tools.glx1 import load_glx1
from logistello_tools.lgw1 import EvalWeights
from logistello_tools.pex1 import load_pex1


@dataclass
class EvalScore:
    """One evaluator's held-out fit."""

    name: str
    n_features: int
    pearson_r: float
    mse: float


def _pattern_predict(pex_path: str, lgw1_path: str) -> tuple[
    np.ndarray, np.ndarray, int
]:
    ex = load_pex1(pex_path)
    w = EvalWeights.load(lgw1_path)
    raw = np.zeros(ex.n_records, dtype=np.int64)
    for i in range(ex.canon.shape[1]):
        ty = int(ex.feat_type[i])
        col = ex.canon[:, i]
        for s in range(13):
            m = ex.stage == s
            if np.any(m):
                raw[m] += w.w[s][ty][col[m]]
    pred = np.array(
        [edax_round_clamp(int(v)) for v in raw], dtype=np.float64
    )
    n_feat = int(sum(len(v) for v in w.w[0]))
    return pred, ex.label.astype(np.float64), n_feat


def _glem_predict(glx_path: str, glm1_path: str) -> tuple[
    np.ndarray, np.ndarray, int
]:
    from logistello_tools.glem import predict

    ex = load_glx1(glx_path)
    model = GlemModel.load(glm1_path)
    pred = predict(model, ex)
    return pred, ex.label.astype(np.float64), model.n_features


def _score(name: str, n_feat: int, pred: np.ndarray, y: np.ndarray) -> EvalScore:
    if pred.std() == 0.0 or y.std() == 0.0:
        r = 0.0
    else:
        r = float(np.corrcoef(pred, y)[0, 1])
    mse = float(np.mean((pred - y) ** 2))
    return EvalScore(name=name, n_features=n_feat, pearson_r=r, mse=mse)


def compare(
    glx1: str, glm1: str, pex1: str, lgw1: str
) -> tuple[EvalScore, EvalScore]:
    """GLEM-auto vs manual PatternEval on the same held-out games."""
    g_pred, g_y, g_nf = _glem_predict(glx1, glm1)
    p_pred, p_y, p_nf = _pattern_predict(pex1, lgw1)
    glem = _score("glem-auto", g_nf, g_pred, g_y)
    manual = _score("manual-pattern", p_nf, p_pred, p_y)
    return glem, manual


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="logistello-tools objective5",
        description=(
            "Objective-5 apparatus: GLEM-auto vs manual PatternEval on "
            "the same held-out corpus (design doc §5)"
        ),
    )
    parser.add_argument("--glx1", required=True, help="held-out GLX1 extract")
    parser.add_argument("--glm1", required=True, help="trained GLM1 model")
    parser.add_argument("--pex1", required=True, help="held-out PEX1 extract")
    parser.add_argument("--lgw1", required=True, help="trained LGW1 weights")
    args = parser.parse_args(argv)

    glem, manual = compare(args.glx1, args.glm1, args.pex1, args.lgw1)
    print("objective5 (GLEM-auto vs manual PatternEval, same held-out set)")
    for s in (glem, manual):
        print(
            f"  {s.name:<16} n_features={s.n_features:<8} "
            f"pearson_r={s.pearson_r:+.4f} mse={s.mse:.3f}"
        )
    print(
        f"  delta(glem-manual): r={glem.pearson_r - manual.pearson_r:+.4f} "
        f"mse={glem.mse - manual.mse:+.3f}  "
        f"(full ELO is Phase 9/10; this is the §5 smoke apparatus)"
    )


if __name__ == "__main__":
    main()
