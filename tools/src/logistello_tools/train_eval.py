"""Phase 4b: pattern evaluation-weight training (design doc §4.4 B3).

Reads a Rust-extracted ``PEX1`` corpus, builds a per-stage sparse design over
the **B4 canonical** indices, and fits the Logistello-2 **linear** evaluation
model ``e(p) = Σ_i w_i f_i(p)`` with **MSE** loss, per stage (13).

Faithful primary path (``--method gd``, the default): gradient descent for
300 iterations, ``w⁰ = 0``, with

* **rare-configuration muting** — every canonical feature-config ``i`` gets a
  fixed multiplier ``min{1, N_i/50} / N_i`` (``N_i`` = #training positions in
  which that config is active; 0 if unseen), applied to its gradient;
* **adjacent-5 smoothing** — a position in stage ``s`` contributes to the
  training set of stages ``s, s±1, s±2`` (clamped to ``[0, 12]``), exactly
  the B2/B3 smoothing window.

Alternative scikit-learn paths are provided for §7 comparison:
``--method ridge`` (``sklearn.linear_model.Ridge``) and ``--method sgd``
(``SGDRegressor``); both still use the adjacent-5 stage smoothing. The GD
path is the faithful default.

Output is the cross-language ``LGW1`` weight file (explicit little-endian,
byte-identical to the Rust reader; see ``WEIGHTS_FORMAT.md``). Weights are
emitted in **1/128-disc units** (the leaf score is ``round(sum/128)``).
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp

from logistello_tools.lgw1 import (
    EVAL_PACKED_SIZE,
    N_STAGES,
    N_TYPES,
    EvalWeights,
)
from logistello_tools.pex1 import load_pex1

# Weights stored in 1/128-disc units (design doc B1; Edax midgame.c:36-44).
DISC_UNIT = 128
# B3: rare-config muting denominator floor.
MUTE_FLOOR = 50.0
# B3: faithful gradient-descent iteration count.
GD_ITERS = 300
# Adjacent-5 stage smoothing half-window (s ± SMOOTH).
SMOOTH = 2


def muting_weights(active_counts: np.ndarray) -> np.ndarray:
    """B3 rare-config muting multiplier ``min{1, N_i/50} / N_i`` per column.

    ``active_counts[i]`` = number of training positions in which canonical
    config ``i`` is active. Columns never seen (``N_i == 0``) get ``0`` (they
    have no gradient signal and stay at the ``w⁰ = 0`` initial value).
    """
    n = active_counts.astype(np.float64)
    out = np.zeros_like(n)
    nz = n > 0
    out[nz] = np.minimum(1.0, n[nz] / MUTE_FLOOR) / n[nz]
    return out


def _column_offsets(canon_sizes: list[int]) -> np.ndarray:
    """Start column of each type block in the concatenated design space."""
    off = np.zeros(N_TYPES + 1, dtype=np.int64)
    off[1:] = np.cumsum(canon_sizes)
    return off


def build_design(
    canon: np.ndarray, feat_type: np.ndarray, canon_sizes: list[int]
) -> sp.csr_matrix:
    """Sparse design ``X`` (n_pos × M) over concatenated canonical columns.

    Column ``= type_offset[type(i)] + canon_index``. Several feature
    instances of the same type may land on the same canonical column; their
    contributions **accumulate** (a count), exactly as Edax sums one weight
    lookup per feature instance.
    """
    n_pos, n_feat = canon.shape
    offsets = _column_offsets(canon_sizes)
    m = int(offsets[-1])
    col = offsets[feat_type][None, :] + canon  # (n_pos, n_feat)
    rows = np.repeat(np.arange(n_pos), n_feat)
    cols = col.reshape(-1)
    data = np.ones(rows.shape[0], dtype=np.float64)
    x = sp.csr_matrix(
        (data, (rows, cols)), shape=(n_pos, m), dtype=np.float64
    )
    x.sum_duplicates()
    return x


def fit_stage_gd(
    x: sp.csr_matrix, y: np.ndarray, iters: int = GD_ITERS
) -> np.ndarray:
    """Faithful B3 GD: ``w⁰=0``, MSE, rare-config muting; ``iters`` steps.

    Returns weights in disc units (NOT yet ×128). The muting multiplier acts
    as a fixed positive diagonal preconditioner; with ``w⁰=0`` and the
    step bounded by the muted curvature the MSE is non-increasing.
    """
    n, m = x.shape
    w = np.zeros(m, dtype=np.float64)
    if n == 0:
        return w
    # N_i = #positions where column i is active.
    active = np.asarray((x > 0).sum(axis=0)).ravel()
    mute = muting_weights(active)
    xt = x.T.tocsr()
    # Per-column curvature bound: diag(XᵀX) = Σ x_ij². With the muted
    # preconditioner the safe constant step is < 1 / max(mute_i * H_ii).
    h = np.asarray(x.multiply(x).sum(axis=0)).ravel()  # Σ x_ij² per column
    denom = float(np.max(mute * h)) if m else 0.0
    lr = 0.9 / denom if denom > 0 else 0.0
    two_over_n = 2.0 / n
    for _ in range(iters):
        resid = x.dot(w) - y  # (n,)
        grad = xt.dot(resid) * two_over_n  # (m,)
        w -= lr * mute * grad
    return w


def fit_stage_sklearn(
    x: sp.csr_matrix, y: np.ndarray, method: str
) -> np.ndarray:
    """§7 alternative: scikit-learn ridge / SGD least-squares fit."""
    if x.shape[0] == 0:
        return np.zeros(x.shape[1], dtype=np.float64)
    if method == "ridge":
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=1.0, fit_intercept=False, solver="sparse_cg")
    elif method == "sgd":
        from sklearn.linear_model import SGDRegressor

        model = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=1e-5,
            fit_intercept=False,
            max_iter=GD_ITERS,
            tol=1e-4,
            random_state=0,
        )
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"unknown sklearn method {method}")
    model.fit(x, y)
    return np.asarray(model.coef_, dtype=np.float64).ravel()


def train(positions: str, method: str = "gd") -> EvalWeights:
    """Trains all 13 stages from a PEX1 corpus and returns ``EvalWeights``.

    Adjacent-5 smoothing: stage ``t``'s training set is every position whose
    own stage is within ``±2`` of ``t`` (clamped to ``[0, 12]``).
    """
    ex = load_pex1(positions)
    if ex.canon_sizes != EVAL_PACKED_SIZE:
        raise ValueError(
            f"PEX1 canon_sizes {ex.canon_sizes} disagree with the LGW1 "
            f"contract {EVAL_PACKED_SIZE} — extractor/contract drift"
        )
    weights = EvalWeights()
    offsets = _column_offsets(ex.canon_sizes)
    for t in range(N_STAGES):
        lo, hi = max(0, t - SMOOTH), min(N_STAGES - 1, t + SMOOTH)
        mask = (ex.stage >= lo) & (ex.stage <= hi)
        if not np.any(mask):
            continue
        x = build_design(
            ex.canon[mask], ex.feat_type, ex.canon_sizes
        )
        y = ex.label[mask].astype(np.float64)
        if method == "gd":
            w = fit_stage_gd(x, y)
        else:
            w = fit_stage_sklearn(x, y, method)
        # Disc units -> 1/128-disc integer units, round to nearest.
        w_units = np.rint(w * DISC_UNIT).astype(np.int64)
        for ty in range(N_TYPES):
            a, b = int(offsets[ty]), int(offsets[ty + 1])
            weights.w[t][ty] = w_units[a:b]
    return weights


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="logistello-tools train-eval",
        description=(
            "Phase 4b: linear pattern-eval weight training "
            "(design doc §4.4 B3) -> LGW1"
        ),
    )
    parser.add_argument(
        "--positions",
        required=True,
        help="input PEX1 extract file (from `logistello extract`)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="output LGW1 weight file",
    )
    parser.add_argument(
        "--method",
        choices=["gd", "ridge", "sgd"],
        default="gd",
        help=(
            "gd = faithful B3 GD-300 with rare-config muting (default); "
            "ridge / sgd = scikit-learn least-squares (§7 comparison)"
        ),
    )
    args = parser.parse_args(argv)
    weights = train(args.positions, method=args.method)
    weights.save(args.output)
    print(
        f"train-eval method={args.method} positions={args.positions} "
        f"stages={N_STAGES} weights={weights.total_weights} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
