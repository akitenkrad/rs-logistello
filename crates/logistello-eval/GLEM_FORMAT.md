# GLEM file formats (`GLX1` extract, `GLM1` model)

Phase 7 (Buro 1998 CG'98; design doc §4.3.6, §4.1 Phase 7, §5
Objective-5). Two explicit **little-endian** byte formats, the same
cross-language discipline as `LGW1` / `PEX1`:

- **`GLX1`** — the GLEM base-literal *extract* the Rust CLI
  `logistello glem-extract` *writes* and the Python trainer
  (`logistello_tools.glx1`) *reads*.
- **`GLM1`** — the trained GLEM *model* the Python trainer
  (`logistello_tools.glm1`) *writes* and the Rust inference side
  (`logistello_eval::GlemModel` / `GlemEval`) *reads*.

## Literal-extraction split (which side owns it, and why)

**Rust owns the base-literal extraction.** This mirrors the Phase-4b
decision that *Rust owns the B4 canonicalisation* (`EXTRACT_FORMAT.md`):
the position → active-base-literal-ids map is defined **once** in
`logistello_eval::glem::BaseFeatureSpec::active_literals` and is the
single source of truth shared by the `glem-extract` corpus writer *and*
the `GlemEval` inference evaluator. Python never re-implements literal
extraction — it only consumes the active-literal sets from `GLX1` and does
the conjunction enumeration / support filter / B3 linear fit / weight
prune. The GOLD interop test (`tests/glem_interop_test.rs`) proves the
Python-trained `GLM1`, loaded by the Rust `GlemEval`, returns **exactly**
Python's own model prediction for many seeded positions (and that `GLM1`
serialization is byte-identical both ways).

## Base-feature families (declarative / extensible)

A *base feature* is a boolean/categorical literal over the board
(side-to-move POV). Families contribute a contiguous block of global
literal ids; the global id of a family-local id is
`offset(family) + local`, where `offset` is the sum of the widths of the
**preceding** families in spec order. The ordered family list **is** the
"base-feature spec id" recorded in both files.

| family    | wire id | literals | encoding                                            |
|-----------|---------|----------|-----------------------------------------------------|
| `cell64`  | 0       | 192      | 64 cells × 3 states; literal `cell*3 + state`, `state ∈ {0=mine,1=opp,2=empty}` |
| `mobility`| 1       | 5        | bucketed legal-move count: `0 \| 1-2 \| 3-5 \| 6-9 \| 10+` |
| `corner`  | 2       | 12       | 4 corners (A1,H1,A8,H8) × 3 states; literal `corner_idx*3 + state` |

A position "has" a conjunction iff **all** its member literals are active.
`max_order = 1` ⇒ exactly the base features (GLEM reduces to a plain
per-literal linear model — design doc §4.3.6).

## `GLX1` layout (all integers little-endian)

| field        | type        | meaning                                          |
|--------------|-------------|--------------------------------------------------|
| magic        | `u32`       | `0x31584C47` (ASCII `"GLX1"`)                    |
| version      | `u32`       | `1`                                              |
| n_stages     | `u32`       | `13` (design doc §4.4 B2)                         |
| n_families   | `u32`       | spec family count                                |
| family_ids   | `[u8; F]`   | spec family wire ids, in spec order              |
| n_records    | `u64`       | number of position records                       |

Then `n_records` **variable-width** rows:

| field   | type          | meaning                                              |
|---------|---------------|------------------------------------------------------|
| label   | `i16`         | final disc differential, side-to-move POV, ∈ [-64,64]|
| stage   | `u8`          | `clamp(floor((discs-13)/4),0,12)` (design doc B2)    |
| pad     | `u8`          | `0`                                                  |
| n_lit   | `u32`         | active base-literal count for this row               |
| lit     | `[u32; n_lit]`| sorted, deduped active base-literal ids              |

Label semantics are exactly the Phase-4b `PEX1` ones (design doc §4.4 B3):
the finished game's Black-relative margin sign-flipped to the position's
side to move. Corpora (`--source selfplay|wthor`, `--max-empties-skip`)
are identical to `extract` and use the **same** per-game seed derivation,
so a `(games, seed)` `GLX1` corpus and `PEX1` corpus visit the same
snapshots in the same order (used by the Objective-5 apparatus).

## `GLM1` layout (all integers little-endian)

| field        | type        | meaning                                          |
|--------------|-------------|--------------------------------------------------|
| magic        | `u32`       | `0x314D4C47` (ASCII `"GLM1"`)                    |
| version      | `u32`       | `1`                                              |
| n_stages     | `u32`       | `13`                                             |
| n_families   | `u32`       | spec family count                                |
| family_ids   | `[u8; F]`   | spec family wire ids, in spec order              |
| n_features   | `u64`       | number of selected conjunctions                  |
| per feature  | …           | `order: u32`, then `order × u32` member literal ids (sorted, deduped) |
| weights      | `i32[]`     | `n_stages × n_features`, row-major stage→feature  |

Weights are signed `i32` in **1/128-disc units** (design doc §4.4 B1;
Edax `midgame.c:36-44`). The leaf score is `round(sum/128)` with Edax bias
rounding, clamped to `±63` — **identical** to `PatternEval` so a heuristic
GLEM leaf can never rival an exact terminal/mate value (design doc §4.5
B6). `from_bytes` rejects a wrong magic/version, wrong stage count,
unknown family id, an order-0 conjunction, a member id ≥ the spec's
literal count, non-sorted/deduped members, truncation, or trailing bytes.

## §4.3.6 `GenerateFeatures` (exact, with B3 substitution)

```
features  <- base_features                               # order-1 literals
for order = 2..max_order:
    for combo in Combinations(base_features, order):
        f_new   <- Conjunction(combo)
        support <- FrequencyInTrainingData(f_new)
        if support >= τ_support: keep f_new
w               <- LinearLeastSquares(features, training_data)   # B3
features_pruned <- { f : |w_f| >= τ_weight }
return features_pruned (+ refit weights on the pruned set)
```

`LinearLeastSquares` is the **reused Phase-4b** per-stage GD-300 +
rare-config muting (`min{1,Ni/50}/Ni`) + adjacent-5 stage smoothing
(`train_eval.fit_all_stages` — a single shared code path, design doc §4.4
B3 / §4.5 B3). `ridge`/`sgd` are the §7 scikit-learn alternatives;
`logistic` is the optional Logistello-1 path used **only** for
Objective-5. Order-1 base literals are always kept (they *are*
`base_features`); only generated order≥2 conjunctions are support-gated.
After the weight prune the model is **refit** on the pruned set so the
emitted weights are consistent with the final feature list.
