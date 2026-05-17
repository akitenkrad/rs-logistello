#!/usr/bin/env bash
#
# setup_edax.sh -- reproducibly build Edax v4.6 + fetch its eval weights and
# install them into the gitignored .edax/ directory at the repo root.
#
# Phase 9a of replication-logistello (design doc Logistello.md §4.5 B5,
# "Edax integration -- canonical"). Edax is the fixed-strength external
# opponent / non-playing oracle used by Phase 9b (elo-vs-edax).
#
# This script is idempotent: re-running it rebuilds from a clean clone and
# overwrites .edax/. It never touches anything tracked by git -- .edax/ is
# listed in .gitignore (platform binary + multi-MB weights are release-only
# and must never be committed).
#
# Requirements (macOS / Linux):
#   - git, curl, make, a C17 compiler (clang or gcc)
#   - a 7z extractor: one of `7z` / `7za` / `p7zip`, OR `uv`/`uvx`
#     (falls back to `uvx --from py7zr py7zr`).
#
# Usage:
#   bash scripts/setup_edax.sh            # build + fetch + install
#   EDAX_BUILD_DIR=/tmp/edax-build \
#     bash scripts/setup_edax.sh          # override scratch build dir
#
# Result layout (all gitignored, under repo-root .edax/):
#   .edax/edax            -- the Edax v4.6 binary (renamed from mEdax-<arch>)
#   .edax/data/eval.dat   -- evaluation weights (v4.4 release; 4.6 format == 4.4)
#
# The project drives this binary in GTP mode:
#   .edax/edax -gtp -eval-file <abs>/.edax/data/eval.dat \
#     -level <N> -book-usage off -n 1 -q
# (deterministic; see EDAX_SETUP.md and src/edax.rs).
#
set -euo pipefail

# --- Pinned upstream -------------------------------------------------------
EDAX_REPO="https://github.com/abulmo/edax-reversi.git"
EDAX_TAG="v4.6"
# Resolved commit for v4.6 (verify after clone; informational):
EDAX_COMMIT="713a434f13b3d15fb69c61b9ba3641aed82c496c"
# Eval weights: 4.4 release; the 4.6 eval format is identical to 4.4.
EVAL_URL="https://github.com/abulmo/edax-reversi/releases/download/v4.4/eval.7z"

# --- Paths -----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EDAX_DIR="$REPO_ROOT/.edax"
BUILD_DIR="${EDAX_BUILD_DIR:-/tmp/edax-build}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "==> Edax setup (tag=$EDAX_TAG, repo=$EDAX_REPO)"
echo "    repo root : $REPO_ROOT"
echo "    build dir : $BUILD_DIR"
echo "    install   : $EDAX_DIR"

# --- 1. Clone v4.6 ---------------------------------------------------------
rm -rf "$BUILD_DIR"
if ! git clone --depth 1 -b "$EDAX_TAG" "$EDAX_REPO" "$BUILD_DIR" 2>/dev/null; then
  echo "    (shallow tag clone failed; full clone + checkout)"
  git clone "$EDAX_REPO" "$BUILD_DIR"
  git -C "$BUILD_DIR" checkout "$EDAX_TAG"
fi
RESOLVED="$(git -C "$BUILD_DIR" rev-parse HEAD)"
echo "==> Cloned Edax $EDAX_TAG @ $RESOLVED"
if [ "$RESOLVED" != "$EDAX_COMMIT" ]; then
  echo "    NOTE: resolved commit differs from pinned $EDAX_COMMIT" >&2
fi

# --- 2. macOS portability patch -------------------------------------------
# Edax v4.6 src/hash.c calls aligned_alloc(32, (size+HASH_N_WAY+1)*sizeof(Hash))
# but sizeof(Hash)==24, so the byte size is not a multiple of 32. C11
# aligned_alloc on macOS strictly requires size % alignment == 0 and otherwise
# returns NULL with errno=EINVAL (Edax then aborts: "hash_init: cannot
# allocate the hash table"). The other two aligned_alloc call sites in Edax
# (eval.c, perft.c) already wrap the size with adjust_size(); hash.c upstream
# did not. Apply the same existing helper -- minimal, consistent, no behavior
# change. (Idempotent: skipped if already patched.)
HASH_C="$BUILD_DIR/src/hash.c"
if grep -q 'aligned_alloc(32, (size + HASH_N_WAY + 1) \* sizeof (Hash))' "$HASH_C"; then
  perl -0pi -e 's/aligned_alloc\(32, \(size \+ HASH_N_WAY \+ 1\) \* sizeof \(Hash\)\)/aligned_alloc(32, adjust_size(32, (size + HASH_N_WAY + 1) * sizeof (Hash)))/' "$HASH_C"
  echo "==> Applied macOS aligned_alloc portability patch to src/hash.c"
else
  echo "==> src/hash.c already patched (or upstream changed); skipping patch"
fi

# --- 3. Build --------------------------------------------------------------
UNAME_S="$(uname -s)"
case "$UNAME_S" in
  Darwin) OS=osx ;;
  Linux)  OS=linux ;;
  *)      OS=linux ;;
esac
# ARCH=native lets the compiler target the host CPU (works for arm64 + x86_64).
CC="${CC:-clang}"
echo "==> Building (OS=$OS ARCH=native CC=$CC)"
mkdir -p "$BUILD_DIR/bin"
make -C "$BUILD_DIR/src" build ARCH=native COMP="$CC" OS="$OS" CC="$CC"

# The Makefile names the binary mEdax-<arch> (osx) / lEdax-<arch> (linux).
BUILT_BIN="$(ls -1 "$BUILD_DIR"/bin/[ml]Edax-* 2>/dev/null | head -n1 || true)"
if [ -z "$BUILT_BIN" ] || [ ! -x "$BUILT_BIN" ]; then
  echo "ERROR: build did not produce an Edax binary in $BUILD_DIR/bin" >&2
  exit 1
fi
echo "==> Built $BUILT_BIN"

# --- 4. Fetch + extract eval weights --------------------------------------
echo "==> Downloading eval weights ($EVAL_URL)"
curl -sL -o "$WORK/eval.7z" "$EVAL_URL"
[ -s "$WORK/eval.7z" ] || { echo "ERROR: eval.7z download failed/empty" >&2; exit 1; }

mkdir -p "$WORK/eval"
extract_ok=0
for z in 7z 7za p7zip; do
  if command -v "$z" >/dev/null 2>&1; then
    ( cd "$WORK/eval" && "$z" x -y "$WORK/eval.7z" >/dev/null ) && extract_ok=1 && break
  fi
done
if [ "$extract_ok" -ne 1 ]; then
  if command -v uvx >/dev/null 2>&1; then
    uvx --from py7zr py7zr x "$WORK/eval.7z" "$WORK/eval/" && extract_ok=1
  elif command -v uv >/dev/null 2>&1; then
    uv run --with py7zr python -c \
      "import py7zr,sys; py7zr.SevenZipFile(sys.argv[1]).extractall(sys.argv[2])" \
      "$WORK/eval.7z" "$WORK/eval" && extract_ok=1
  fi
fi
[ "$extract_ok" -eq 1 ] || { echo "ERROR: no 7z extractor available (install 7z or uv)" >&2; exit 1; }

EVAL_DAT="$(find "$WORK/eval" -name eval.dat -type f | head -n1)"
[ -n "$EVAL_DAT" ] || { echo "ERROR: eval.dat not found in archive" >&2; exit 1; }
echo "==> Extracted $(basename "$EVAL_DAT") ($(wc -c < "$EVAL_DAT") bytes)"

# --- 5. Install into gitignored .edax/ ------------------------------------
rm -rf "$EDAX_DIR"
mkdir -p "$EDAX_DIR/data"
cp "$BUILT_BIN" "$EDAX_DIR/edax"
chmod +x "$EDAX_DIR/edax"
cp "$EVAL_DAT" "$EDAX_DIR/data/eval.dat"

echo "==> Installed:"
ls -la "$EDAX_DIR" "$EDAX_DIR/data"

# --- 6. Smoke check --------------------------------------------------------
echo "==> Smoke (GTP genmove from the standard opening, level 2):"
SMOKE="$(
  { printf 'boardsize 8\nclear_board\n'; sleep 0.3; printf 'genmove black\n'; sleep 2; printf 'quit\n'; sleep 0.3; } \
  | "$EDAX_DIR/edax" -gtp -eval-file "$EDAX_DIR/data/eval.dat" -level 2 -book-usage off -n 1 -q 2>/dev/null \
  | grep -a '^= [a-h]' | head -n1 || true
)"
if [ -n "$SMOKE" ]; then
  echo "    OK: genmove black -> '$SMOKE'"
else
  echo "    WARNING: smoke produced no move line (shell pacing race; the Rust" >&2
  echo "    integration test uses blocking IO and is reliable)." >&2
fi

echo "==> Done. .edax/ ready (gitignored)."
