# Training-position extract format (`PEX1`)

`PEX1` is the columnar binary the Rust CLI `logistello extract` *writes*
(Phase 4b, design doc §4.4 B3 / §4.5 B5) and the Python trainer
`logistello_tools.pex1` *reads*. It is a fixed **little-endian**,
self-describing layout designed for fast `numpy.fromfile` ingestion.

Rust **owns the B4 canonicalisation**: each feature's canonical index is
computed with the *exact same* `logistello-eval` pack tables / key
extraction the `PatternEval` evaluator uses, so the Python trainer never
re-implements canonicalisation — it only consumes indices. (The Rust GOLD
interop test proves the resulting weights round-trip byte-identically.)

## Header (all integers little-endian)

| offset | type       | meaning                                                     |
|--------|------------|-------------------------------------------------------------|
| 0      | `u32`      | magic = `0x31584550` (ASCII `"PEX1"`)                       |
| 4      | `u32`      | version = `1`                                              |
| 8      | `u32`      | n_feat = `47` (design doc §4.4 B1)                          |
| 12     | `u32`      | n_types = `9`                                              |
| 16     | `[u32;9]`  | canon_sizes — per-type canonical-class count, in            |
|        |            | `WEIGHT_TYPE_ORDER` `C9,C10,S10,S8,S7,S6,S5,S4,Const`       |
|        |            | = `10206,29889,29646,3321,1134,378,135,45,1` (same as LGW1) |
| 52     | `[u8;47]`  | feat_type — `type_slot` (0..9) per feature (self-describing)|
| 99     | `u64`      | n_records                                                   |

Header size = `4+4+4+4 + 9*4 + 47 + 8` = **107 bytes**.

> Note: `canon_sizes` here is the **9-entry, type-order** table — the same
> one LGW1 uses — *not* the 13-entry accumulate-order Edax
> `EVAL_PACKED_SIZE` (where S10/S8 appear several times). The Python trainer
> rejects any `PEX1` whose `canon_sizes` disagree with the LGW1 contract.

## Records

`n_records` fixed-width rows of **192 bytes** each:

| offset | type       | meaning                                                          |
|--------|------------|------------------------------------------------------------------|
| 0      | `i16`      | label — final disc differential, **side-to-move POV**, ∈ [-64,64]|
| 2      | `u8`       | stage — `clamp(floor((discs-13)/4),0,12)` (design doc §4.4 B2)    |
| 3      | `u8`       | pad — `0` (reserved / 4-byte alignment of the canon block)        |
| 4      | `[u32;47]` | canon — player-pack canonical index per feature, accumulate order|

Row size = `2 + 1 + 1 + 47*4` = **192 bytes**; file size =
`107 + 192 * n_records`.

## Label semantics (design doc §4.4 B3)

The label is the game's **terminal disc differential propagated to the
position** (the faithful "game result propagated" target). Concretely it is
the finished game's Black-relative margin `count(Black) - count(White)`
sign-flipped to the position's side to move:

```
label = (side_to_move == Black) ?  (black - white)_terminal
                                : -(black - white)_terminal
```

so a Black-to-move position in a Black-winning game gets a **positive**
label and the same board with White to move gets the **negated** label.
Endgame negamax-refinement of the label is an optional documented extra
(not required for Phase 4); the disc-count terminal margin is the faithful
default.

## Corpora

- `--source selfplay --games N --seed S` — `RandomPlayer` vs `RandomPlayer`
  games via the reused `othello-engine`; deterministic for `(N, S)`; needs
  **no external data** (the test vehicle).
- `--source wthor --wthor-dir DIR` — real expert games parsed from `.wtb`
  via `othello_io::WthorReader`. WTHOR omits passes; the reader auto-inserts
  `Move::Pass` and the replay additionally guards forced passes
  (design doc §4.5 B6). Every non-terminal position of every game is
  emitted. `.jou`/`.trn` (player/tournament names) are not needed for
  labels and are ignored.

`--max-empties-skip K` drops positions with more than `K` empty squares
(skip the very opening) — omit to keep every non-terminal position.

## WTHOR data (best-effort, design doc §4.5 B5)

The FFO WTHOR archives are at
<https://www.ffothello.org/informatique/la-base-wthor/>. If a download is
unreachable/slow the **selfplay** path needs nothing and is the test
vehicle. To use real games, download `WTH_*.wtb` files into a directory and
run:

```bash
cargo run --release -p logistello-cli -- extract \
    --source wthor --wthor-dir data/wthor/ --output data/positions.pex1
uv run logistello-tools train-eval \
    --positions data/positions.pex1 --output results/eval.lgw1 --method gd
```
