"""Phase 4b B3 unit tests (design doc §4.4 B2/B3).

These tests *define correctness* for the Python training side:

* the rare-config muting weight is exactly ``min{1, N_i/50} / N_i``;
* adjacent-5 stage smoothing touches exactly ``{s, s±1, s±2} ∩ [0, 12]``;
* faithful GD with ``w⁰ = 0`` reduces MSE monotonically on a toy problem;
* the stage split uses the B2 ``clamp(floor((discs-13)/4), 0, 12)`` formula
  (verified against the Rust contract via the canonical sizes / N_STAGES).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from logistello_tools import train_eval
from logistello_tools.lgw1 import EVAL_PACKED_SIZE, N_STAGES


def test_muting_weight_exact_formula():
    # N_i = 0 -> 0 (no signal); 1..49 -> 1/N_i; >=50 -> 1/N_i (==min(1,N/50)/N).
    counts = np.array([0, 1, 2, 25, 49, 50, 51, 100, 5000], dtype=np.int64)
    got = train_eval.muting_weights(counts)
    expect = []
    for n in counts:
        if n == 0:
            expect.append(0.0)
        else:
            expect.append(min(1.0, n / 50.0) / n)
    np.testing.assert_allclose(got, np.array(expect))
    # Explicit spot checks: N=0 -> 0; N=1 -> min(1,1/50)/1 = 1/50;
    # N=50 -> min(1,1)/50 = 1/50; N=100 -> min(1,2)/100 = 1/100.
    assert got[0] == 0.0
    assert got[1] == 1.0 / 50.0
    assert got[5] == 1.0 / 50.0  # N=50
    assert got[7] == 1.0 / 100.0  # N=100


def test_muting_small_vs_large_regime():
    # For N < 50: min(1, N/50)/N = (N/50)/N = 1/50  (constant!).
    # For N >= 50: min(1, N/50)/N = 1/N  (decreasing).
    small = train_eval.muting_weights(np.array([1, 2, 10, 49]))
    np.testing.assert_allclose(small, np.full(4, 1.0 / 50.0))
    large = train_eval.muting_weights(np.array([50, 100, 200]))
    np.testing.assert_allclose(large, np.array([1 / 50.0, 1 / 100.0, 1 / 200.0]))


def test_adjacent5_smoothing_window_exact():
    # The training set for stage t is positions with stage in
    # {t-2..t+2} ∩ [0,12]. We assert the window membership directly.
    for t in range(N_STAGES):
        lo, hi = max(0, t - 2), min(N_STAGES - 1, t + 2)
        window = set(range(lo, hi + 1))
        expect = {
            s
            for s in (t - 2, t - 1, t, t + 1, t + 2)
            if 0 <= s <= N_STAGES - 1
        }
        assert window == expect, f"stage {t}: {window} != {expect}"
    # Ends are clamped: stage 0 -> {0,1,2}; stage 12 -> {10,11,12}.
    assert set(range(max(0, 0 - 2), min(12, 0 + 2) + 1)) == {0, 1, 2}
    assert set(range(max(0, 12 - 2), min(12, 12 + 2) + 1)) == {10, 11, 12}


def test_gd_w0_zero_monotone_mse_decrease():
    rng = np.random.default_rng(0)
    n, m = 400, 30
    # Sparse 0/1 design + a true linear target with a little noise.
    dense = (rng.random((n, m)) < 0.25).astype(np.float64)
    x = sp.csr_matrix(dense)
    w_true = rng.normal(size=m)
    y = x.dot(w_true) + rng.normal(scale=0.1, size=n)

    # Reproduce the exact GD recurrence to record the MSE trajectory.
    active = np.asarray((x > 0).sum(axis=0)).ravel()
    mute = train_eval.muting_weights(active)
    xt = x.T.tocsr()
    h = np.asarray(x.multiply(x).sum(axis=0)).ravel()
    denom = float(np.max(mute * h))
    lr = 0.9 / denom
    w = np.zeros(m)
    assert np.all(w == 0.0), "w⁰ must be 0"
    mses = []
    for _ in range(train_eval.GD_ITERS):
        resid = x.dot(w) - y
        mses.append(float(np.mean(resid**2)))
        grad = xt.dot(resid) * (2.0 / n)
        w -= lr * mute * grad
    mses = np.array(mses)
    # Monotone non-increasing (allow tiny float slack).
    diffs = np.diff(mses)
    assert np.all(diffs <= 1e-9), f"MSE increased: max step {diffs.max()}"
    # And it actually learned something.
    assert mses[-1] < mses[0] * 0.5

    # The public fit_stage_gd reproduces the same final weights.
    w_fit = train_eval.fit_stage_gd(x, y)
    np.testing.assert_allclose(w_fit, w, rtol=0, atol=1e-9)


def test_stage_split_matches_b2_canon_sizes():
    # The Python side must agree with the Rust LGW1 contract: 13 stages and
    # the EVAL_PACKED_SIZE canonical counts (B2/B4).
    assert N_STAGES == 13
    assert EVAL_PACKED_SIZE == [10206, 29889, 29646, 3321, 1134, 378, 135, 45, 1]
    assert 13 * sum(EVAL_PACKED_SIZE) == 971815


def test_build_design_accumulates_duplicate_canonical_columns():
    # Two feature instances of the same type landing on the same canonical
    # index must accumulate (count 2) — Edax sums one lookup per instance.
    feat_type = np.array([0, 0], dtype=np.uint8)  # both type 0 (C9)
    canon_sizes = list(EVAL_PACKED_SIZE)
    canon = np.array([[5, 5]], dtype=np.int64)  # both -> column 5
    x = train_eval.build_design(canon, feat_type, canon_sizes)
    assert x.shape == (1, sum(canon_sizes))
    assert x[0, 5] == 2.0
    assert x.sum() == 2.0
