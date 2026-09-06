[English](opening-book.md) | [日本語](../ja/opening-book.md)

# Opening-book learning

`learn-book` reproduces Buro's opening-book learning (Buro 1999, design doc
§4.3.7): it learns the book by **self-play + Negamax back-propagation +
drawishness**. It writes the explicit little-endian `OPB1` book (see
[`BOOK_FORMAT.md`](../../crates/logistello-book/BOOK_FORMAT.md)) and is
deterministic for a fixed `--seed`.

## `learn-book` — learn the book

```bash
cargo run --release -p logistello-cli -- learn-book \
    --num-games 1000 --depth 24 --drawishness 0.3 --seed 1 \
    --output results/book.opb1
```

| Flag | Meaning | Default |
|---|---|---|
| `--num-games` | Number of self-play games (design doc §5.1: 10000) | `1000` |
| `--depth` | Per-move self-play search depth (§5.1: 24; §6 `--book-depth-values {12,18,24,30}`) | `24` |
| `--drawishness` | Drawishness blend `λ ∈ [0,0.5]` (§5.1: 0.3; §6 range `0.0..0.5`). Clamped into range | `0.3` |
| `--seed` | RNG seed for the (deterministic) exploration policy | `1` |
| `--max-book-plies` | Cap on the parent ply still booked (opening-only book bound) | `20` |
| `--book-endgame-empties` | Empties floor below which positions are not booked | `30` |
| `--eval-weights` | Optional `LGW1`: self-play with `PatternEval` instead of `BasicEval` | — |
| `--output` | Output `OPB1` book path (required) | — |

## Play using the learned book

```bash
cargo run --release -p logistello-cli -- play \
    --black engine --white random --depth 8 --seed 42 \
    --book results/book.opb1
```

With `--book`, the `engine` player consults the book first: if the current
position is booked (and the booked move is legal) it plays the booked move
instead of searching; once play leaves the booked opening it falls back to
the normal search (composes with every other flag).

> **Honest note:** the §4.3.7 mechanism is faithfully self-play +
> Negamax back-propagation of leaf values + a drawishness `λ` blend, exactly
> per Buro 1999. One simplification is noted: the self-play used to discover
> book positions runs at the single search depth `--depth D` for the whole
> game (rather than an iterative / variable-depth schedule), so the book is
> only as discriminating as depth `D` self-play.
