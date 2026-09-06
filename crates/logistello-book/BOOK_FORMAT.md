# `OpeningBook` serialization format (`OPB1`, explicit little-endian)

This is the on-disk format of the Phase-8 learned opening book
(`logistello-book`, design doc §4.3.7 / Buro 1999, "Toward Opening Book
Learning"). It is **our own** explicit, versioned, little-endian byte layout
(no `serde` / `bincode` internals), mirroring the project's other formats
(`LGW1`, `PEX1`, `GLX1`). The only consumer/producer is Rust
(`OpeningBook::{to_bytes,from_bytes,save,load}`), but it is kept fully
explicit and self-describing so the bytes are reproducible across builds and
the round-trip is provably bit-exact (asserted by the GOLD tests).

## Determinism

`to_bytes` emits positions in **ascending `zobrist_key` order** and, within a
position, recorded moves in **first-seen order**. Two books that are equal as
maps therefore serialize to byte-identical blobs, and `learn_book` with a
fixed `BookConfig` (the exploration RNG is seeded from `cfg.seed`) produces a
byte-identical file on every run.

## Units & semantics

- A *position* is identified by `logistello_core::zobrist_key` (the
  side-to-move-aware 64-bit Zobrist hash, design doc §4.3.2). Transpositions
  collapse to one entry, so the book is a DAG.
- The full `(side_to_move, 64 cells)` board is stored alongside the key so a
  loaded book can re-derive legal moves (for `OpeningBook::probe`'s legality
  re-check and for `negamax_backpropagate`).
- `value` is the **Negamax-back-propagated** value from the position's
  side-to-move POV (disc scale, design doc §4.5 B6 sign convention); before
  back-prop it is the accumulated self-play estimate.
- `best_move` is the negamax-best move, possibly shifted by drawishness
  (design doc §4.3.7 / Buro 1999). `Pass` only at a must-pass position.
- Per recorded move: `eval_sum` (sum of parent-POV `final_eval` over the
  self-play games that played it here) and `visits` — the
  `UpdateStatistics` accumulator of design doc §4.3.7.

## Move encoding (`u16` LE)

| value      | meaning                                  |
|------------|------------------------------------------|
| `0xFFFF`   | `Move::Pass`                             |
| `0..=63`   | `Move::Place`, bit index `row*8 + col`   |

Any other value is rejected by `from_bytes`.

## Cell encoding (`u8`)

`0 = empty`, `1 = black`, `2 = white`. Any other value is rejected.

## Layout (exact byte offsets, all integers little-endian)

### Header (20 bytes)

| offset | type     | value / meaning                                   |
|--------|----------|---------------------------------------------------|
| 0      | `u32` LE | `magic` = `0x3142504F` = ASCII `"OPB1"`            |
| 4      | `u32` LE | `version` = `1`                                   |
| 8      | `u64` LE | `n_entries` = number of booked positions          |

### Then `n_entries` records, in **ascending `zobrist_key`** order

Each record:

| field          | type        | meaning                                      |
|----------------|-------------|----------------------------------------------|
| `key`          | `u64` LE    | `zobrist_key` of the position (strictly ↑)   |
| `side_to_move` | `u8`        | `0 = Black`, `1 = White`                     |
| `cells[64]`    | `u8 × 64`   | row-major (`row 0..8`, `col 0..8`); cell code |
| `best_move`    | `u16` LE    | move encoding (see above)                    |
| `value`        | `i32` LE    | negamax-backed value (side-to-move POV)      |
| `visit_count`  | `u32` LE    | total self-play visits to this position      |
| `n_moves`      | `u32` LE    | number of distinct recorded moves            |
| then `n_moves` move-stat triples, in **first-seen** order:           |
| `move`         | `u16` LE    | move encoding                                |
| `eval_sum`     | `i64` LE    | Σ parent-POV `final_eval` for this move      |
| `visits`       | `u32` LE    | games that played this move here             |

Per-record byte size = `8 + 1 + 64 + 2 + 4 + 4 + 4 + n_moves*(2+8+4)`
= `87 + 14 * n_moves`.

## Rejection rules (`from_bytes`)

`from_bytes` returns an `io::Error` (`InvalidData`) for any of:

- wrong `magic`;
- unsupported `version` (`!= 1`);
- a truncated header or body (fewer bytes than the layout requires);
- trailing bytes after the last record;
- keys not strictly ascending (a malformed / tampered file);
- an out-of-range encoded move (`64..=0xFFFE`);
- an out-of-range cell code (`> 2`) or side byte (`> 1`).

The empty book (`n_entries = 0`) is valid and round-trips (20-byte file).

## Relationship to the design doc

- §4.3.7 record fields `{best move, value, visit count}` →
  `best_move`, `value`, `visit_count`.
- §4.3.7 `UpdateStatistics(move, final_eval)` accumulator → the per-move
  `(eval_sum, visits)` triples.
- §4.3.7 `NegamaxBackpropagate` operates on the stored board (recomputes
  successors); §4.3.7 / Buro 1999 `ApplyDrawishness` only changes
  `best_move`, never the stored stats, so re-serializing a drawishness-
  shifted book differs from the negamax-only book **only** in `best_move`
  bytes (drawishness at `λ = 0` is an exact no-op).
