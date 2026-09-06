# Edax v4.6 setup (external engine)

Phase 9a wires **Edax v4.6** in as the fixed-strength external opponent and
non-playing oracle (design doc `Logistello.md` §4.5 **B5**, "Edax integration
— canonical"). Phase 9b (`elo-vs-edax`) consumes it.

Edax is a **platform binary plus multi-MB evaluation weights**. Neither is
committed: `.edax/` is in `.gitignore`. Regenerate it locally with the
reproducible steps below (or just run the script).

## Quick start

```bash
bash scripts/setup_edax.sh
```

Idempotent: it re-clones, re-patches, rebuilds, re-fetches the weights, and
overwrites `.edax/`. Nothing tracked by git is touched.

## What gets produced

```
.edax/
├── edax            # Edax v4.6 binary (renamed from mEdax-<arch> / lEdax-<arch>)
└── data/
    └── eval.dat    # evaluation weights (v4.4 release; 4.6 format == 4.4)
```

`book.dat` / `game.ggf` are **not** installed — Edax only requires
`data/eval.dat`; the book is disabled at runtime with `-book-usage off`.

## Manual reproducible steps

### 1. Clone the pinned source

```bash
git clone --depth 1 -b v4.6 https://github.com/abulmo/edax-reversi.git /tmp/edax-build
# (if the shallow tag clone fails: full clone, then `git checkout v4.6`)
```

- Tag: **v4.6**
- Resolved commit: **`713a434f13b3d15fb69c61b9ba3641aed82c496c`**

### 2. macOS portability patch (required on macOS)

Edax v4.6 `src/hash.c` line 59 calls:

```c
hash_table->hash = aligned_alloc(32, (size + HASH_N_WAY + 1) * sizeof (Hash));
```

`sizeof(Hash) == 24`, so the byte size is **not** a multiple of the 32-byte
alignment. The C11 `aligned_alloc` on macOS strictly requires
`size % alignment == 0`; otherwise it returns `NULL` with `errno = EINVAL (22)`
and Edax aborts at startup:

```
FATAL ERROR: ./hash.c : hash_init : 61 :  error #22 : Invalid argument
hash_init: cannot allocate the hash table
```

This is an upstream bug: the **other two** `aligned_alloc` call sites in Edax
(`src/eval.c:759`, `src/perft.c:348`) already wrap the size with the existing
`adjust_size()` helper (which rounds up to a multiple of the alignment);
`hash.c` simply forgot. The fix applies the **same existing helper** — minimal,
consistent, no behaviour change:

```c
hash_table->hash = aligned_alloc(32, adjust_size(32, (size + HASH_N_WAY + 1) * sizeof (Hash)));
```

`scripts/setup_edax.sh` applies this patch automatically (idempotently). On
Linux/glibc the unpatched code also works (glibc `aligned_alloc` is lenient),
but the patch is harmless there too.

### 3. Build (macOS arm64 / x86_64)

```bash
make -C /tmp/edax-build/src build ARCH=native COMP=clang OS=osx CC=clang
```

- `OS=osx` (use `OS=linux` on Linux), `ARCH=native` targets the host CPU
  (works for both Apple Silicon `arm64` and Intel `x86_64`), `CC=clang`.
- Output: `/tmp/edax-build/bin/mEdax-native` (osx) — a
  `Mach-O 64-bit executable arm64`, ≈ 504 KB.
- Version banner: `Edax version 4.6 ... copyright 1998 - 2024 Richard
  Delorme, Toshihiko Okuhara`.

### 4. Fetch + extract eval weights

```bash
curl -OL https://github.com/abulmo/edax-reversi/releases/download/v4.4/eval.7z
# extract eval.dat from the 7z archive; if no 7z/7za/p7zip:
uvx --from py7zr py7zr x eval.7z ./out/
```

`eval.dat` is ≈ 13.95 MB. The 4.6 eval format is identical to 4.4 (per the
design doc), so the v4.4 release weights are used.

### 5. Install

```bash
mkdir -p .edax/data
cp /tmp/edax-build/bin/mEdax-native .edax/edax && chmod +x .edax/edax
cp .../eval.dat .edax/data/eval.dat
```

## Driving Edax (protocol)

Edax is driven via **GTP** (`-gtp`), its native Go-Text-Protocol mode
(`src/gtp.c`). This is the protocol `rs-othello-sim`'s
`othello-player::ExternalEnginePlayer` speaks as `Protocol::Gtp`:

| rs-othello-sim sends | Edax `-gtp` replies      |
| -------------------- | ------------------------ |
| `boardsize 8`        | `= \n\n`                 |
| `clear_board`        | `= \n\n`                 |
| `play white d3`      | `= \n\n`                 |
| `genmove black`      | `= f5\n\n`               |
| `quit`               | `=\n`                    |

The `Protocol::Ntest` dialect in `rs-othello-sim` uses `set game <board>` +
`go`, which Edax does **not** understand (Edax's own text UI uses `setboard`,
not `set game`). GTP is therefore the correct and only viable choice, and it
matches `rs-othello-sim`'s GTP response parser (`= body\n\n`) exactly.

Canonical invocation (fixed-strength, deterministic):

```bash
.edax/edax -gtp -eval-file <ABS>/.edax/data/eval.dat \
  -level <N> -book-usage off -n 1 -q
```

- `-level N` — search level (fixed strength).
- `-book-usage off` — disable the opening book (none installed anyway).
- `-n 1` — single thread → **deterministic** (verified: same opening
  produces the same move across runs).
- `-q` — quiet (no board redisplay; clean GTP stream).
- `-eval-file` is passed as an **absolute path** so the working directory
  does not matter.

The shared Rust helper that builds this configuration lives in
`crates/logistello-cli/src/edax.rs` (`EdaxConfig`); the default binary path
is `.edax/edax` (`EdaxConfig::DEFAULT_EDAX_PATH`, overridable with
`--edax-path`). The integration test
`tests/edax_smoke_test.rs` exercises it (and **skips, passing**, when
`.edax/edax` is absent, so CI / other machines stay green).

## References

- Edax: <https://github.com/abulmo/edax-reversi> (tag `v4.6`)
- Weights: <https://github.com/abulmo/edax-reversi/releases/download/v4.4/eval.7z>
- Design doc `Logistello.md` §4.5 **B5** (Edax integration — canonical)
