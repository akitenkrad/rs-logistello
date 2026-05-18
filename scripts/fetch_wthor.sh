#!/usr/bin/env bash
#
# fetch_wthor.sh -- download the FFO WThor 1997 game base + the shared
# player (.jou) / tournament (.trn) name databases into the gitignored
# data/wthor/ directory at the repo root.
#
# Phase 9b of replication-logistello (design doc Logistello.md §4.5 B5,
# "Data acquisition / format"). The 1997 base contains (among ~7681 games)
# the six Takeshi Murakami vs. Logistello games -- the reproduction's gold
# validation set. `logistello murakami-extract` then filters & emits the
# tiny committed tests/data/murakami_1997.json from this raw DB.
#
# This script is idempotent: it skips any artifact already present with a
# plausible size (pass FORCE=1 to re-download). It only ever writes under
# data/, which is listed in .gitignore -- the raw third-party DB must never
# be committed (only the derived 6-game JSON is).
#
# Requirements: curl, unzip.
#
# Usage:
#   bash scripts/fetch_wthor.sh           # download (skip what exists)
#   FORCE=1 bash scripts/fetch_wthor.sh   # force re-download
#   WTHOR_YEAR=1998 bash scripts/fetch_wthor.sh   # a different year base
#
# Result layout (all gitignored, under repo-root data/wthor/):
#   data/wthor/WTH_1997.wtb   -- the 1997 game base (binary .wtb)
#   data/wthor/WTHOR.JOU      -- player-name DB (20-byte ASCII records)
#   data/wthor/WTHOR.TRN      -- tournament-name DB (26-byte ASCII records)
#
set -euo pipefail

YEAR="${WTHOR_YEAR:-1997}"
BASE="https://www.ffothello.org/wthor"
ZIP_URL="$BASE/base_zip/WTH_${YEAR}.ZIP"
JOU_URL="$BASE/base/WTHOR.JOU"
TRN_URL="$BASE/base/WTHOR.TRN"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$REPO_ROOT/data/wthor"

mkdir -p "$OUT_DIR"
echo "==> WThor fetch (year=$YEAR)"
echo "    out dir : $OUT_DIR  (gitignored)"

# present_ok <file> <min_bytes> -> 0 if the file exists and is big enough.
present_ok() {
  local f="$1" min="$2"
  [ -z "${FORCE:-}" ] || return 1
  [ -f "$f" ] || return 1
  local sz
  sz="$(wc -c < "$f" | tr -d ' ')"
  [ "$sz" -ge "$min" ]
}

# --- 1. The year base (.wtb) comes zipped ---------------------------------
WTB="$OUT_DIR/WTH_${YEAR}.wtb"
if present_ok "$WTB" 1000; then
  echo "==> WTH_${YEAR}.wtb already present ($(wc -c < "$WTB" | tr -d ' ') bytes); skipping"
else
  TMP_ZIP="$(mktemp -t wthor_zip.XXXXXX)"
  trap 'rm -f "$TMP_ZIP"' EXIT
  echo "==> Downloading $ZIP_URL"
  curl -sSL --fail --max-time 120 -o "$TMP_ZIP" "$ZIP_URL"
  echo "==> Unzipping into $OUT_DIR"
  unzip -o "$TMP_ZIP" -d "$OUT_DIR" >/dev/null
  rm -f "$TMP_ZIP"
  trap - EXIT
  [ -f "$WTB" ] || { echo "ERROR: $WTB not produced by unzip" >&2; exit 1; }
  echo "    OK: $WTB ($(wc -c < "$WTB" | tr -d ' ') bytes)"
fi

# --- 2. Player / tournament name DBs (shared, not per-year) ---------------
fetch_plain() {
  local url="$1" dest="$2" min="$3" label="$4"
  if present_ok "$dest" "$min"; then
    echo "==> $label already present ($(wc -c < "$dest" | tr -d ' ') bytes); skipping"
    return
  fi
  echo "==> Downloading $url"
  curl -sSL --fail --max-time 120 -o "$dest" "$url"
  local sz
  sz="$(wc -c < "$dest" | tr -d ' ')"
  if [ "$sz" -lt "$min" ]; then
    echo "ERROR: $label download too small ($sz bytes) -- got an error page?" >&2
    rm -f "$dest"
    exit 1
  fi
  echo "    OK: $dest ($sz bytes)"
}
fetch_plain "$JOU_URL" "$OUT_DIR/WTHOR.JOU" 1000 "WTHOR.JOU (players)"
fetch_plain "$TRN_URL" "$OUT_DIR/WTHOR.TRN" 100 "WTHOR.TRN (tournaments)"

echo "==> Done. data/wthor/ ready (gitignored):"
ls -la "$OUT_DIR"
echo
echo "Next: cargo run --release -p logistello-cli -- murakami-extract \\"
echo "        --wthor-dir data/wthor --output tests/data/murakami_1997.json"
