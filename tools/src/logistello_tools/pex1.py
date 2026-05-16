"""PEX1 extracted-position reader (Phase 4b).

Mirrors the Rust writer ``logistello_cli::extract::write_pex1`` documented in
``EXTRACT_FORMAT.md``. Layout (all integers little-endian)::

    u32  magic     = 0x31584550  ("PEX1")
    u32  version   = 1
    u32  n_feat    = 47
    u32  n_types   = 9
    u32  canon_sizes[9]    (EVAL_PACKED_SIZE in WEIGHT_TYPE_ORDER)
    u8   feat_type[47]     type_slot per feature (self-describing)
    u64  n_records
    --- then n_records fixed-width rows: ---
    i16  label             final disc diff, side-to-move POV, [-64,64]
    u8   stage             0..12
    u8   pad               0
    u32  canon[47]         player-pack canonical index per feature
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

PEX1_MAGIC = 0x31584550  # ASCII "PEX1"
N_FEATURES = 47
N_TYPES = 9
_HEADER_FMT = "<IIII"  # magic, version, n_feat, n_types
_HEADER_BYTES = 4 + 4 + 4 + 4 + 9 * 4 + N_FEATURES + 8
RECORD_BYTES = 2 + 1 + 1 + N_FEATURES * 4  # 192


@dataclass
class Extract:
    """A loaded PEX1 corpus."""

    feat_type: np.ndarray  # uint8[47] : type_slot per feature
    canon_sizes: list[int]  # [9]
    label: np.ndarray  # int16[N]
    stage: np.ndarray  # uint8[N]
    canon: np.ndarray  # int64[N, 47] : canonical index per feature

    @property
    def n_records(self) -> int:
        return int(self.label.shape[0])


def load_pex1(path: str) -> Extract:
    """Reads a PEX1 file into vectorised numpy arrays."""
    with open(path, "rb") as fh:
        blob = fh.read()
    if len(blob) < _HEADER_BYTES:
        raise ValueError("PEX1 truncated header")
    magic, version, n_feat, n_types = struct.unpack_from(_HEADER_FMT, blob, 0)
    if magic != PEX1_MAGIC:
        raise ValueError("bad magic (expected PEX1)")
    if version != 1:
        raise ValueError(f"unsupported PEX1 version {version}")
    if n_feat != N_FEATURES or n_types != N_TYPES:
        raise ValueError("PEX1 feature/type count mismatch")
    canon_sizes = list(struct.unpack_from("<9I", blob, 16))
    feat_type = np.frombuffer(blob, dtype=np.uint8, count=N_FEATURES, offset=52)
    (n_records,) = struct.unpack_from("<Q", blob, 52 + N_FEATURES)
    want = _HEADER_BYTES + n_records * RECORD_BYTES
    if len(blob) != want:
        raise ValueError(
            f"PEX1 size mismatch: got {len(blob)} expected {want}"
        )
    # Structured dtype over the fixed-width rows -> vectorised columns.
    row_dt = np.dtype(
        [
            ("label", "<i2"),
            ("stage", "u1"),
            ("pad", "u1"),
            ("canon", "<u4", (N_FEATURES,)),
        ]
    )
    rows = np.frombuffer(
        blob, dtype=row_dt, count=n_records, offset=_HEADER_BYTES
    )
    return Extract(
        feat_type=feat_type.copy(),
        canon_sizes=canon_sizes,
        label=rows["label"].astype(np.int64),
        stage=rows["stage"].astype(np.int64),
        canon=rows["canon"].astype(np.int64),
    )
