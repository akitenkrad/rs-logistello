"""GLX1 extracted-GLEM-position reader (Phase 7).

Mirrors the Rust writer ``logistello_cli::glem_extract::write_glx1``
documented in ``crates/logistello-eval/GLEM_FORMAT.md``. Rust **owns the
base-literal extraction** (the single source of truth shared with
``GlemEval``); Python only consumes the active-literal sets and does the
conjunction enumeration / support filter / linear fit / weight prune.

Layout (all integers little-endian)::

    u32  magic       = 0x31584C47  ("GLX1")
    u32  version     = 1
    u32  n_stages    = 13
    u32  n_families
    u8   family_ids[n_families]   base-feature spec family wire ids
    u64  n_records
    --- then n_records variable-width rows: ---
    i16  label       final disc diff, side-to-move POV, [-64,64]
    u8   stage       0..12
    u8   pad         0
    u32  n_lit       active base-literal count for this row
    u32  lit[n_lit]  sorted, deduped active base-literal ids
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

GLX1_MAGIC = 0x31584C47  # ASCII "GLX1"
GLX1_VERSION = 1
N_STAGES = 13

# Base-family wire ids (mirror logistello_eval::glem::BaseFamily::wire_id).
FAMILY_NAME_BY_WIRE = {0: "cell64", 1: "mobility", 2: "corner"}
FAMILY_WIRE_BY_NAME = {v: k for k, v in FAMILY_NAME_BY_WIRE.items()}
FAMILY_WIDTH = {"cell64": 64 * 3, "mobility": 5, "corner": 4 * 3}


@dataclass
class GlemExtract:
    """A loaded GLX1 corpus.

    ``literals`` is a list of ``np.ndarray`` (one sorted id array per
    position) because rows are variable-width. ``label`` / ``stage`` are
    dense int arrays parallel to it.
    """

    family_ids: list[int]  # base-feature spec family wire ids, in order
    label: np.ndarray  # int64[N]
    stage: np.ndarray  # int64[N]
    literals: list[np.ndarray]  # N entries; each int64[n_lit_i]

    @property
    def n_records(self) -> int:
        return int(self.label.shape[0])

    @property
    def family_names(self) -> list[str]:
        return [FAMILY_NAME_BY_WIRE[w] for w in self.family_ids]

    @property
    def n_literals(self) -> int:
        return sum(FAMILY_WIDTH[n] for n in self.family_names)


def load_glx1(path: str) -> GlemExtract:
    """Reads a GLX1 file into a :class:`GlemExtract`."""
    with open(path, "rb") as fh:
        blob = fh.read()
    if len(blob) < 16:
        raise ValueError("GLX1 truncated header")
    magic, version, n_stages, n_fam = struct.unpack_from("<IIII", blob, 0)
    if magic != GLX1_MAGIC:
        raise ValueError("bad magic (expected GLX1)")
    if version != GLX1_VERSION:
        raise ValueError(f"unsupported GLX1 version {version}")
    if n_stages != N_STAGES:
        raise ValueError("GLX1 stage count mismatch")
    off = 16
    family_ids = list(blob[off : off + n_fam])
    for w in family_ids:
        if w not in FAMILY_NAME_BY_WIRE:
            raise ValueError(f"unknown family wire id {w}")
    off += n_fam
    (n_records,) = struct.unpack_from("<Q", blob, off)
    off += 8

    labels = np.empty(n_records, dtype=np.int64)
    stages = np.empty(n_records, dtype=np.int64)
    literals: list[np.ndarray] = []
    for r in range(n_records):
        label, stage, _pad, n_lit = struct.unpack_from("<hBBI", blob, off)
        off += 8
        lit = np.frombuffer(blob, dtype="<u4", count=n_lit, offset=off).astype(
            np.int64
        )
        off += 4 * n_lit
        labels[r] = label
        stages[r] = stage
        literals.append(lit)
    if off != len(blob):
        raise ValueError("GLX1 trailing bytes / size mismatch")
    return GlemExtract(
        family_ids=family_ids,
        label=labels,
        stage=stages,
        literals=literals,
    )
