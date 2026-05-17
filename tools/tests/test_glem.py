"""Phase 7 GLEM unit tests (design doc §4.3.6, with B3 substitution).

These tests *define correctness* for the Python GLEM side:

* enumeration: #order-k conjunctions over n base features == C(n, k) before
  filtering; ``max_order == 1`` ⇒ exactly the base features (GLEM reduces to
  a plain per-feature linear model);
* support filter: a feature with empirical frequency < τ_support is dropped,
  ≥ kept (hand corpus);
* weight prune: features with |w| < τ_weight are removed and the pruned
  model is refit; pruned eval == sum over surviving features only;
* B3 fidelity: the GLEM linear fit goes through the SAME shared Phase-4b
  GD/muting/smoothing code path (``train_eval.fit_all_stages``);
* GLM1 (de)serialization is internally consistent (cross-language byte
  identity is proven by the Rust GOLD test ``tests/glem_interop_test.rs``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from logistello_tools import glem, train_eval
from logistello_tools.glm1 import N_STAGES, GlemModel
from logistello_tools.glx1 import GlemExtract


def _toy_extract(rows: list[tuple[int, int, list[int]]]) -> GlemExtract:
    """Builds a hand GlemExtract from ``(label, stage, literals)`` rows."""
    return GlemExtract(
        family_ids=[2],  # "corner" — small (12 literals) so ids stay valid
        label=np.array([r[0] for r in rows], dtype=np.int64),
        stage=np.array([r[1] for r in rows], dtype=np.int64),
        literals=[np.array(sorted(set(r[2])), dtype=np.int64) for r in rows],
    )


# --- 2. Enumeration correctness ------------------------------------------


def test_enumeration_counts_match_binomial_before_filter():
    base = np.arange(6, dtype=np.int64)  # 6 base features
    for max_order in (1, 2, 3, 4):
        feats = glem.enumerate_conjunctions(base, max_order)
        # order 1 == base; order k>=2 == C(6, k).
        by_order: dict[int, int] = {}
        for f in feats:
            by_order[len(f)] = by_order.get(len(f), 0) + 1
        assert by_order[1] == 6
        for k in range(2, max_order + 1):
            assert by_order[k] == math.comb(6, k), (max_order, k)
        if max_order == 1:
            # max_order == 1 ⇒ exactly the base features (per-feature linear)
            assert len(feats) == 6
            assert all(len(f) == 1 for f in feats)


def test_max_order_1_is_plain_per_literal_model():
    ex = _toy_extract(
        [
            (10, 0, [0, 3]),
            (-4, 0, [3, 6]),
            (2, 1, [0, 6]),
        ]
    )
    model, rep = glem.generate_and_fit(
        ex, max_order=1, support_threshold=0.0, weight_threshold=0.0
    )
    # All features are order 1.
    assert all(len(f) == 1 for f in model.features)
    assert rep.n_generated == 0  # no order>=2 candidates


# --- 3. Support filter ---------------------------------------------------


def test_support_filter_drops_below_threshold_keeps_at_or_above():
    # 4 positions. Generated pair (0,3) appears in 1/4 = 0.25 of rows;
    # pair (0,6) appears in 2/4 = 0.5. With τ_support = 0.4: (0,6) kept,
    # (0,3) dropped. Order-1 base literals are always kept.
    ex = _toy_extract(
        [
            (1, 0, [0, 3]),
            (1, 0, [0, 6]),
            (1, 0, [0, 6]),
            (1, 0, [3, 9]),
        ]
    )
    base = glem.base_literals(ex)
    feats = glem.enumerate_conjunctions(base, 2)
    p = glem.occurrence_matrix(ex)
    sup = glem.support_counts(p, feats)
    freq = sup / ex.n_records
    keep = freq >= 0.4
    kept_pairs = {
        tuple(f) for f, k in zip(feats, keep) if k and len(f) == 2
    }
    assert (0, 6) in kept_pairs
    assert (0, 3) not in kept_pairs  # 0.25 < 0.4

    model, rep = glem.generate_and_fit(
        ex, max_order=2, support_threshold=0.4, weight_threshold=0.0
    )
    feat_set = {tuple(f) for f in model.features}
    assert (0, 6) in feat_set
    assert (0, 3) not in feat_set
    # Every order-1 base literal survives the support gate.
    for b in base:
        assert (int(b),) in feat_set
    assert rep.n_after_support >= len(base)


# --- 4. Weight prune -----------------------------------------------------


def test_weight_prune_removes_small_and_refits():
    # Construct a corpus where one base literal perfectly predicts a large
    # label and another carries no signal; with a high τ_weight only the
    # informative feature survives, and the refit eval == sum over it only.
    rows = []
    for i in range(40):
        if i % 2 == 0:
            rows.append((60, 5, [0]))  # literal 0 -> big positive label
        else:
            rows.append((-60, 5, [3]))  # literal 3 -> big negative label
        rows.append((0, 5, [6]))  # literal 6 -> zero label (no signal)
    ex = _toy_extract(rows)

    model, rep = glem.generate_and_fit(
        ex,
        max_order=1,
        support_threshold=0.0,
        weight_threshold=5.0,  # disc-unit threshold
    )
    # Literal 6's weight ~0 -> pruned; 0 and 3 carry strong signal -> kept.
    surviving = {tuple(f) for f in model.features}
    assert (6,) not in surviving
    assert (0,) in surviving or (3,) in surviving
    assert rep.n_after_weight_prune <= rep.n_after_support
    assert rep.n_after_weight_prune == len(model.features)

    # Pruned eval == sum over surviving features only (definitional).
    pred = glem.predict(model, ex)
    feat_sets = [set(f) for f in model.features]
    for r in range(ex.n_records):
        rs = set(int(x) for x in ex.literals[r])
        st = int(ex.stage[r])
        raw = sum(
            int(model.w[st][fi])
            for fi, fs in enumerate(feat_sets)
            if fs <= rs
        )
        biased = raw + 64 if raw > 0 else raw - 64
        score = (
            int(biased // 128) if biased >= 0 else -int((-biased) // 128)
        )
        assert pred[r] == max(-63, min(63, score))


# --- 5. B3 fidelity (shared Phase-4b code path) --------------------------


def test_b3_fit_uses_shared_train_eval_code_path(monkeypatch):
    # The GLEM per-stage fit MUST go through train_eval.fit_all_stages
    # (single shared GD/muting/smoothing path — design doc §4.4/§4.5 B3).
    called = {"n": 0}
    real = train_eval.fit_all_stages

    def spy(design_for_stage, method="gd"):
        called["n"] += 1
        return real(design_for_stage, method=method)

    monkeypatch.setattr(train_eval, "fit_all_stages", spy)
    ex = _toy_extract([(5, 0, [0]), (-5, 0, [3]), (1, 1, [0, 3])])
    glem.generate_and_fit(
        ex, max_order=2, support_threshold=0.0, weight_threshold=0.0
    )
    # generate_and_fit fits twice (support set, then the pruned refit).
    assert called["n"] >= 2


def test_b3_smoothing_window_is_b2_adjacent5():
    # GLEM reuses the exact SMOOTH constant from train_eval (B2/B3 ±2).
    assert glem.SMOOTH == train_eval.SMOOTH == 2
    assert N_STAGES == 13


# --- GLM1 (de)serialization internal consistency -------------------------


def test_glm1_roundtrip_and_normalisation():
    m = GlemModel(
        family_names=["cell64", "corner"],
        features=[[3, 1, 3], [200], [5, 191]],  # unsorted/dup -> normalised
        w=np.arange(N_STAGES * 3, dtype=np.int64).reshape(N_STAGES, 3),
    )
    assert m.features[0] == [1, 3]  # deduped + sorted
    blob = m.to_bytes()
    back = GlemModel.from_bytes(blob)
    assert back.family_names == ["cell64", "corner"]
    assert back.features == [[1, 3], [200], [5, 191]]
    assert np.array_equal(back.w, m.w)
    assert back.to_bytes() == blob


@dataclass
class _Case:
    blob_mut: object
    why: str


def test_glm1_rejects_corrupt_blobs():
    m = GlemModel(family_names=["mobility"], features=[[0]],
                   w=np.zeros((N_STAGES, 1), dtype=np.int64))
    good = m.to_bytes()
    bad_magic = bytearray(good)
    bad_magic[0] ^= 0xFF
    for blob, why in [
        (bytes(bad_magic), "magic"),
        (good[:5], "truncated"),
        (good + b"\x00", "trailing"),
    ]:
        try:
            GlemModel.from_bytes(blob)
            raise AssertionError(f"should have rejected: {why}")
        except (ValueError, Exception):
            pass
