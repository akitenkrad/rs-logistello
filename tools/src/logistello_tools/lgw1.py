"""LGW1 evaluation-weight container (explicit little-endian).

This is the Python side of the cross-language ``LGW1`` contract documented in
``crates/logistello-eval/WEIGHTS_FORMAT.md``. It must produce / consume
**byte-identical** files with the Rust ``logistello_eval::EvalWeights``
(``to_bytes`` / ``from_bytes``). The Edax pack/unpack symmetry tables
(design doc §4.4 B4) are ported *verbatim* from
``crates/logistello-eval/src/weights.rs`` so the canonical-class mapping
agrees bit-for-bit; the interop GOLD test proves it.

Layout (all integers little-endian)::

    u32  magic       = 0x3157474C  ("LGW1")
    u32  version     = 1
    u32  n_stages    = 13
    u32  n_types     = 9
    u32  canon_sizes[9]   (C9,C10,S10,S8,S7,S6,S5,S4,Const)
    u64  data_len    = 13 * sum(canon_sizes) = 971815
    i32  data[data_len]   row-major stage -> type -> canonical_index
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

WEIGHTS_MAGIC = 0x3157474C  # ASCII "LGW1", little-endian
WEIGHTS_VERSION = 1
N_STAGES = 13
N_TYPES = 9

# 3**k for k = 0..10 (mirrors weights.rs POW3).
POW3 = [1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049]

# Edax symmetry generators (eval.c:581-583), vendored verbatim from
# weights.rs SYM_S10 / SYM_C10 / SYM_C9.
SYM_S10 = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
SYM_C10 = [9, 8, 7, 6, 4, 5, 3, 2, 1, 0]
SYM_C9 = [0, 2, 1, 4, 3, 5, 7, 6, 8]

# Pattern types in WEIGHT_TYPE_ORDER (= LGW1 type order). Index == type_slot.
# (name, k). Const has k = 0.
TYPE_ORDER = [
    ("C9", 9),
    ("C10", 10),
    ("S10", 10),
    ("S8", 8),
    ("S7", 7),
    ("S6", 6),
    ("S5", 5),
    ("S4", 4),
    ("Const", 0),
]

# Edax EVAL_PACKED_SIZE reordered to TYPE_ORDER (weights.rs).
EVAL_PACKED_SIZE = [10206, 29889, 29646, 3321, 1134, 378, 135, 45, 1]


def _sym_for(name: str) -> list[int]:
    """Symmetry generator slice for a pattern type (weights.rs sym_for)."""
    if name == "C9":
        return SYM_C9
    if name == "C10":
        return SYM_C10
    if name == "S10":
        return SYM_S10
    if name == "S8":
        return SYM_S10[2:]  # sym_S10 + 2
    if name == "S7":
        return SYM_S10[3:]
    if name == "S6":
        return SYM_S10[4:]
    if name == "S5":
        return SYM_S10[5:]
    if name == "S4":
        return SYM_S10[6:]
    return []  # Const


def _player_feature(sym: list[int], n: int, l: int) -> int:
    """Edax player_feature (eval.c:517-527): digit permutation of base-3 key."""
    f = 0
    for i in range(n):
        s = sym[i]
        f += ((l // POW3[s]) % 3) * POW3[i]
    return f


def _opponent_feature(l: int, d: int) -> int:
    """Edax opponent_feature (eval.c:498-506): colour-swap each base-3 digit."""
    o = (1, 0, 2)
    f = o[l % 3]
    if d > 1:
        f += _opponent_feature(l // 3, d - 1) * 3
    return f


@dataclass
class PackTable:
    """Per-pattern-type Edax pack table (raw key -> canonical class)."""

    player: np.ndarray  # int64[3**k] : raw key (player POV) -> canonical
    n_canonical: int


def build_pack_table(name: str, k: int) -> PackTable:
    """Builds the pack table for a type exactly as Rust ``PackTable::build``."""
    if name == "Const":
        return PackTable(player=np.zeros(1, dtype=np.int64), n_canonical=1)
    size = POW3[k]
    sym = _sym_for(name)
    player = np.zeros(size, dtype=np.int64)
    n = 0
    for i in range(size):
        j = _player_feature(sym, k, i)
        if j < i:
            player[i] = player[j]
        else:
            player[i] = n
            n += 1
    return PackTable(player=player, n_canonical=n)


def build_pack_tables() -> dict[str, PackTable]:
    """All pack tables keyed by type name; validates EVAL_PACKED_SIZE."""
    tables: dict[str, PackTable] = {}
    for idx, (name, k) in enumerate(TYPE_ORDER):
        t = build_pack_table(name, k)
        if t.n_canonical != EVAL_PACKED_SIZE[idx]:
            raise ValueError(
                f"pack table {name}: n_canonical {t.n_canonical} != "
                f"EVAL_PACKED_SIZE {EVAL_PACKED_SIZE[idx]} — pack tables "
                f"disagree with the vendored Edax arrays"
            )
    for idx, (name, k) in enumerate(TYPE_ORDER):
        tables[name] = build_pack_table(name, k)
    return tables


class EvalWeights:
    """Per-stage, per-type canonical weight vectors (1/128-disc i32 units).

    ``w[stage][type_slot]`` is an int array of length ``canon_sizes[slot]``.
    """

    def __init__(self) -> None:
        self.canon_sizes = list(EVAL_PACKED_SIZE)
        self.w: list[list[np.ndarray]] = [
            [np.zeros(sz, dtype=np.int64) for sz in self.canon_sizes]
            for _ in range(N_STAGES)
        ]

    @property
    def total_weights(self) -> int:
        return N_STAGES * sum(self.canon_sizes)

    def to_bytes(self) -> bytes:
        """Serializes to the explicit little-endian ``LGW1`` byte stream."""
        head = struct.pack(
            "<IIII",
            WEIGHTS_MAGIC,
            WEIGHTS_VERSION,
            N_STAGES,
            N_TYPES,
        )
        head += struct.pack("<9I", *self.canon_sizes)
        head += struct.pack("<Q", self.total_weights)
        # Row-major: stage -> type -> canonical index. clip to i32 range.
        flat = np.concatenate(
            [self.w[s][t] for s in range(N_STAGES) for t in range(N_TYPES)]
        )
        flat = np.clip(flat, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
        return head + flat.astype("<i4").tobytes()

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            fh.write(self.to_bytes())

    @classmethod
    def from_bytes(cls, blob: bytes) -> EvalWeights:
        magic, version, n_stages, n_types = struct.unpack_from("<IIII", blob, 0)
        if magic != WEIGHTS_MAGIC:
            raise ValueError("bad magic (expected LGW1)")
        if version != WEIGHTS_VERSION:
            raise ValueError("unsupported weights version")
        if n_stages != N_STAGES or n_types != N_TYPES:
            raise ValueError("stage/type count mismatch")
        canon_sizes = list(struct.unpack_from("<9I", blob, 16))
        if canon_sizes != EVAL_PACKED_SIZE:
            raise ValueError("canonical size mismatch vs vendored pack tables")
        (data_len,) = struct.unpack_from("<Q", blob, 52)
        expected = N_STAGES * sum(canon_sizes)
        if data_len != expected:
            raise ValueError("weight data length mismatch")
        body = np.frombuffer(blob, dtype="<i4", count=data_len, offset=60)
        if len(blob) != 60 + 4 * data_len:
            raise ValueError("blob size does not match declared data length")
        out = cls()
        off = 0
        for s in range(N_STAGES):
            for t in range(N_TYPES):
                n = canon_sizes[t]
                out.w[s][t] = body[off : off + n].astype(np.int64)
                off += n
        return out

    @classmethod
    def load(cls, path: str) -> EvalWeights:
        with open(path, "rb") as fh:
            return cls.from_bytes(fh.read())
