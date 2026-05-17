"""GLEM feature generation + fit (Buro 1998 CG'98; design doc §4.3.6,
§4.1 Phase 7, §5 Objective-5).

Implements the §4.3.6 ``GenerateFeatures`` procedure **exactly**, with the
B3 substitution (design doc §4.4 B3 / §4.5 B3): the canonical Logistello-2
fit is the Phase-4b **linear least-squares** on the final disc differential
(reused GD-300 + rare-config muting + adjacent-5 stage smoothing), not
logistic regression. ``--method gd`` (default) is faithful; ``ridge`` /
``sgd`` are the §7 scikit-learn alternatives; the optional Logistello-1
``logistic`` path is provided for Objective-5 only.

::

    GenerateFeatures(base_features, max_order):
        features  <- base_features                       # order-1 literals
        for order = 2..max_order:
            for combo in Combinations(base_features, order):
                f_new   <- Conjunction(combo)
                support <- FrequencyInTrainingData(f_new)
                if support >= τ_support: keep f_new
        w               <- LinearLeastSquares(features, training_data)  # B3
        features_pruned <- { f : |w_f| >= τ_weight }
        return features_pruned (+ refit weights on the pruned set)

**Rust owns the base-literal extraction** (mirrors the Phase-4b "Rust owns
canonicalisation" decision). This module consumes the per-position active
base-literal id sets from a GLX1 corpus (``glx1.load_glx1``); it never
re-implements literal extraction. The fitted model is written as the
cross-language ``GLM1`` file (byte-identical to the Rust ``GlemModel``;
the GOLD interop test proves it).

The stage split is the B2 13 stages (the GLX1 ``stage`` column, identical
to ``logistello_eval::stage``); adjacent-5 smoothing reuses ``train_eval``.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from logistello_tools import train_eval
from logistello_tools.glm1 import N_STAGES, GlemModel
from logistello_tools.glx1 import GlemExtract

# Weights stored in 1/128-disc units (design doc B1; Edax midgame.c:36-44),
# identical scale to LGW1 / PatternEval / GlemEval.
DISC_UNIT = 128
SMOOTH = train_eval.SMOOTH  # adjacent-5 stage smoothing half-window (B2/B3)


@dataclass
class GlemFitReport:
    """Bookkeeping the CLI prints (the §4.3.6 / Objective-5 counts)."""

    n_base: int  # order-1 base literals present in the corpus
    n_generated: int  # candidate conjunctions enumerated (orders 2..K)
    n_after_support: int  # features kept after the τ_support filter
    n_after_weight_prune: int  # features kept after the |w| >= τ_weight prune


def base_literals(ex: GlemExtract) -> np.ndarray:
    """The order-1 base literals that actually occur in the corpus.

    GLEM's ``base_features`` are the boolean literals; we restrict to the
    ones with non-zero support so the order-2.. enumeration is over a
    meaningful universe (an all-absent literal can never gain support and
    only inflates ``C(n, k)``). Returns a sorted id array.
    """
    if ex.n_records == 0:
        return np.zeros(0, dtype=np.int64)
    seen = np.unique(np.concatenate(ex.literals)) if ex.literals else np.zeros(
        0, dtype=np.int64
    )
    return seen.astype(np.int64)


def occurrence_matrix(ex: GlemExtract) -> sp.csr_matrix:
    """Sparse 0/1 position × base-literal occurrence matrix ``P``.

    ``P[r, l] == 1`` iff base literal ``l`` is active in position ``r``.
    Every GLEM quantity reduces to a vectorised op on ``P``:

    * order-1 support of literal ``l``  = column sum ``P[:, l]``;
    * order-``k`` conjunction column     = elementwise AND of its member
      columns (``min`` of the 0/1 columns), so support = that column's sum;

    making support filtering / design construction matrix ops instead of
    O(n_pos · n_feat) Python set tests (essential for the cell64 demo,
    ~19k order-2 features × thousands of rows).
    """
    n = ex.n_records
    n_lit = ex.n_literals
    if n == 0:
        return sp.csr_matrix((0, n_lit), dtype=np.int8)
    rows: list[int] = []
    cols: list[int] = []
    for r, lit in enumerate(ex.literals):
        rows.extend([r] * len(lit))
        cols.extend(int(x) for x in lit)
    data = np.ones(len(rows), dtype=np.int8)
    return sp.csr_matrix(
        (data, (rows, cols)), shape=(n, n_lit), dtype=np.int8
    )


def _feature_column(p: sp.csr_matrix, members: tuple[int, ...]) -> np.ndarray:
    """Dense 0/1 indicator over rows: 1 iff the position has *all* member
    literals (the conjunction is satisfied). Vectorised: count how many
    member columns are active per row and compare to the order."""
    if not members:
        return np.ones(p.shape[0], dtype=np.int8)
    sub = p[:, list(members)]
    have = np.asarray(sub.sum(axis=1)).ravel()
    return (have == len(members)).astype(np.int8)


def enumerate_conjunctions(
    base: np.ndarray, max_order: int
) -> list[tuple[int, ...]]:
    """All conjunctions of orders ``1..max_order`` over ``base``.

    Order 1 = the base literals themselves (so ``max_order == 1`` makes GLEM
    a plain per-literal linear model — design doc §4.3.6). Orders ``>= 2``
    are ``itertools.combinations`` (so the order-``k`` count is exactly
    ``C(len(base), k)`` *before* the support filter).
    """
    if max_order < 1:
        raise ValueError("max_order must be >= 1")
    feats: list[tuple[int, ...]] = [(int(b),) for b in base]
    for order in range(2, max_order + 1):
        for combo in itertools.combinations((int(b) for b in base), order):
            feats.append(tuple(sorted(combo)))
    return feats


def feature_matrix(
    p: sp.csr_matrix, feats: list[tuple[int, ...]]
) -> sp.csr_matrix:
    """Sparse 0/1 design ``F`` (n_pos × n_feat): ``F[r, j] == 1`` iff
    position ``r`` satisfies conjunction ``feats[j]`` (all members active).
    Vectorised over positions via :func:`_feature_column`."""
    n = p.shape[0]
    m = len(feats)
    if m == 0 or n == 0:
        return sp.csr_matrix((n, m), dtype=np.float64)
    cols = [_feature_column(p, f) for f in feats]
    return sp.csr_matrix(
        np.asarray(cols, dtype=np.float64).T
    )


def support_counts(
    p: sp.csr_matrix, feats: list[tuple[int, ...]]
) -> np.ndarray:
    """#positions that satisfy each conjunction. Frequency = count /
    n_records. Vectorised: column sums of the feature design matrix."""
    f = feature_matrix(p, feats)
    if f.shape[1] == 0:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(f.sum(axis=0)).ravel().astype(np.int64)


def fit_per_stage(
    feats: list[tuple[int, ...]],
    p: sp.csr_matrix,
    ex: GlemExtract,
    method: str,
) -> np.ndarray:
    """Per-stage B3 least-squares fit over ``feats`` with adjacent-5
    smoothing (reuses the exact ``train_eval`` GD/muting code path).

    Returns an ``(N_STAGES, n_feat)`` weight array in **disc units**.
    """
    m = len(feats)
    w = np.zeros((N_STAGES, m), dtype=np.float64)
    if m == 0 or ex.n_records == 0:
        return w
    stage = ex.stage
    label = ex.label.astype(np.float64)
    f_all = feature_matrix(p, feats).tocsr()

    def design_for_stage(t: int):
        lo, hi = max(0, t - SMOOTH), min(N_STAGES - 1, t + SMOOTH)
        mask = (stage >= lo) & (stage <= hi)
        rows = np.nonzero(mask)[0]
        if rows.size == 0:
            return None
        return f_all[rows], label[rows]

    if method == "logistic":
        # Objective-5 only: Logistello-1 logistic path (design doc §4.5 B3).
        for t in range(N_STAGES):
            d = design_for_stage(t)
            if d is None:
                continue
            w[t] = _fit_stage_logistic(*d)
        return w

    fitted = train_eval.fit_all_stages(design_for_stage, method=method)
    for t, fw in enumerate(fitted):
        if fw is not None:
            w[t] = fw
    # Degenerate toy stages (no learning-rate curvature, constant target)
    # can leave non-finite weights; the GLM1 contract is finite i32, so
    # sanitise to 0 (no signal) — keeps Rust/Python byte-identical.
    return np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)


def _fit_stage_logistic(x: sp.csr_matrix, y: np.ndarray) -> np.ndarray:
    """Optional Logistello-1 path (design doc §4.5 B3): logistic regression
    of win(1)/draw(0.5)/loss(0) — here we map the disc-diff label to a
    win/draw/loss target and fit ``LogisticRegression``. Not the canonical
    Logistello-2 fit; provided for Objective-5 comparison only."""
    if x.shape[0] == 0:
        return np.zeros(x.shape[1], dtype=np.float64)
    from sklearn.linear_model import LogisticRegression

    cls = np.where(y > 0, 1, 0)
    if len(np.unique(cls)) < 2:
        return np.zeros(x.shape[1], dtype=np.float64)
    model = LogisticRegression(
        fit_intercept=False, max_iter=200, C=1.0, solver="liblinear"
    )
    model.fit(x, cls)
    return np.asarray(model.coef_, dtype=np.float64).ravel()


def generate_and_fit(
    ex: GlemExtract,
    max_order: int,
    support_threshold: float,
    weight_threshold: float,
    method: str = "gd",
) -> tuple[GlemModel, GlemFitReport]:
    """The §4.3.6 ``GenerateFeatures`` procedure end to end (B3 fit).

    1. ``features <- base_features`` (order-1) ∪ order-2.. conjunctions;
    2. keep ``f`` iff ``support(f) / n_records >= support_threshold``;
    3. ``w <- LinearLeastSquares`` (B3 per-stage GD/muting/smoothing);
    4. prune to ``{ f : max_stage |w_f| >= weight_threshold }``;
    5. **refit** weights on the pruned feature set;
    6. return the ``GlemModel`` (+ a :class:`GlemFitReport`).
    """
    family_names = ex.family_names
    p = occurrence_matrix(ex)
    base = base_literals(ex)

    all_feats = enumerate_conjunctions(base, max_order)
    n_base = int((base.size))
    n_generated = len(all_feats) - n_base  # orders 2..K only

    n = max(ex.n_records, 1)
    sup = support_counts(p, all_feats)
    keep = (sup.astype(np.float64) / n) >= support_threshold
    # Order-1 base literals are always kept (they ARE base_features in
    # §4.3.6; only the *generated* order>=2 conjunctions are support-gated).
    for i in range(n_base):
        keep[i] = True
    feats = [f for f, k in zip(all_feats, keep) if k]
    n_after_support = len(feats)

    # B3 fit on the support-filtered set.
    w_disc = fit_per_stage(feats, p, ex, method)
    # Prune: keep features whose max |weight| across stages >= τ_weight.
    if feats:
        maxabs = np.max(np.abs(w_disc), axis=0)
        pruned_mask = maxabs >= weight_threshold
        pruned_feats = [f for f, m in zip(feats, pruned_mask) if m]
    else:
        pruned_feats = []
    n_after_weight_prune = len(pruned_feats)

    # Refit on the pruned feature set (§4.3.6 "return features_pruned" with
    # weights consistent with the final feature set).
    w_ref = fit_per_stage(pruned_feats, p, ex, method)
    # The GLM1 contract is signed-i32 1/128-disc weights (design doc §4.4
    # B1). Round then **clip to the i32 range** so the in-memory model is
    # byte-identical to what ``to_bytes``/Rust read back (a fit that
    # diverges on a collinear design must still produce a contract-valid,
    # round-trip-stable model — never an int64 overflow sentinel).
    i32min, i32max = np.iinfo(np.int32).min, np.iinfo(np.int32).max
    w_scaled = np.rint(
        np.clip(w_ref * DISC_UNIT, float(i32min), float(i32max))
    )
    w_units = w_scaled.astype(np.int64)

    model = GlemModel(
        family_names=family_names,
        features=[list(f) for f in pruned_feats],
        w=w_units.reshape(N_STAGES, len(pruned_feats)),
    )
    report = GlemFitReport(
        n_base=n_base,
        n_generated=n_generated,
        n_after_support=n_after_support,
        n_after_weight_prune=n_after_weight_prune,
    )
    return model, report


def predict(model: GlemModel, ex: GlemExtract) -> np.ndarray:
    """Per-position ``round(Σ satisfied w / 128)`` with Edax bias rounding +
    ±63 clamp — identical convention to the Rust ``GlemEval``.

    Vectorised: raw stage sum = (satisfied-feature design ``F``) · (per-row
    stage weight). The bias/clamp arithmetic is the *exact* integer formula
    (truncate toward zero, then ±63 clamp) the Rust ``GlemEval`` uses, just
    applied with numpy int64."""
    n = ex.n_records
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    p = occurrence_matrix(ex)
    f = feature_matrix(p, [tuple(c) for c in model.features]).tocsr()
    raw = np.zeros(n, dtype=np.int64)
    if f.shape[1] > 0:
        for s in range(N_STAGES):
            m = ex.stage == s
            if np.any(m):
                ws = np.asarray(model.w[s], dtype=np.int64)
                raw[m] = (f[m] @ ws).astype(np.int64)
    biased = np.where(raw > 0, raw + 64, raw - 64)
    # Integer truncation toward zero — exactly Rust's `biased / 128` on i64
    # (NOT floor; floor differs for negatives). sign * (|biased| // 128).
    sign = np.sign(biased).astype(np.int64)
    score = sign * (np.abs(biased) // 128)
    return np.clip(score, -63, 63).astype(np.float64)
