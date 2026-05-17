"""GLM1 GLEM-model container (explicit little-endian).

Python side of the cross-language ``GLM1`` contract documented in
``crates/logistello-eval/GLEM_FORMAT.md``. It must produce / consume
**byte-identical** files with the Rust ``logistello_eval::GlemModel``
(``to_bytes`` / ``from_bytes``). The Rust GOLD interop test proves it
bit-for-bit.

Layout (all integers little-endian)::

    u32  magic       = 0x314D4C47  ("GLM1")
    u32  version     = 1
    u32  n_stages    = 13
    u32  n_families
    u8   family_ids[n_families]
    u64  n_features
    --- per feature: ---
      u32 order
      u32 member[order]              sorted, deduped base-literal ids
    --- weights: ---
    i32  data[n_stages * n_features] row-major stage -> feature, 1/128 disc
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

GLM1_MAGIC = 0x314D4C47  # ASCII "GLM1"
GLM1_VERSION = 1
N_STAGES = 13

# Mirror logistello_eval::glem::BaseFamily.
FAMILY_NAME_BY_WIRE = {0: "cell64", 1: "mobility", 2: "corner"}
FAMILY_WIRE_BY_NAME = {v: k for k, v in FAMILY_NAME_BY_WIRE.items()}
FAMILY_WIDTH = {"cell64": 64 * 3, "mobility": 5, "corner": 4 * 3}


def spec_n_literals(family_names: list[str]) -> int:
    """Total base-literal id count for an ordered family-name list."""
    return sum(FAMILY_WIDTH[n] for n in family_names)


def spec_offsets(family_names: list[str]) -> dict[str, int]:
    """Global base-literal-id offset of each family in spec order."""
    out: dict[str, int] = {}
    off = 0
    for name in family_names:
        out[name] = off
        off += FAMILY_WIDTH[name]
    return out


@dataclass
class GlemModel:
    """A GLEM model: base-feature spec, selected conjunctions, per-stage
    weights (1/128-disc i32 units).

    ``features`` is a list of sorted, deduped member-id lists. ``w`` is an
    ``(N_STAGES, n_features)`` int array (row-major stage -> feature).
    """

    family_names: list[str]
    features: list[list[int]] = field(default_factory=list)
    w: np.ndarray | None = None  # (N_STAGES, n_features) int64

    def __post_init__(self) -> None:
        if self.w is None:
            self.w = np.zeros(
                (N_STAGES, len(self.features)), dtype=np.int64
            )
        # Normalise every feature (sorted + deduped) so serialization is
        # canonical and matches the Rust reader's defensive check.
        self.features = [sorted(set(f)) for f in self.features]
        for f in self.features:
            if not f:
                raise ValueError("conjunction of order 0")
        assert self.w.shape == (N_STAGES, len(self.features))

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def n_literals(self) -> int:
        return spec_n_literals(self.family_names)

    def to_bytes(self) -> bytes:
        fam_ids = bytes(FAMILY_WIRE_BY_NAME[n] for n in self.family_names)
        out = bytearray()
        out += struct.pack(
            "<IIII",
            GLM1_MAGIC,
            GLM1_VERSION,
            N_STAGES,
            len(self.family_names),
        )
        out += fam_ids
        out += struct.pack("<Q", self.n_features)
        for f in self.features:
            out += struct.pack("<I", len(f))
            out += np.asarray(f, dtype="<u4").tobytes()
        flat = np.asarray(self.w, dtype=np.int64).reshape(-1)
        flat = np.clip(flat, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
        out += flat.astype("<i4").tobytes()
        return bytes(out)

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            fh.write(self.to_bytes())

    @classmethod
    def from_bytes(cls, blob: bytes) -> GlemModel:
        magic, version, n_stages, n_fam = struct.unpack_from("<IIII", blob, 0)
        if magic != GLM1_MAGIC:
            raise ValueError("bad magic (expected GLM1)")
        if version != GLM1_VERSION:
            raise ValueError("unsupported GLM1 version")
        if n_stages != N_STAGES:
            raise ValueError("stage count mismatch")
        off = 16
        fam_ids = list(blob[off : off + n_fam])
        off += n_fam
        family_names = [FAMILY_NAME_BY_WIRE[w] for w in fam_ids]
        (n_feat,) = struct.unpack_from("<Q", blob, off)
        off += 8
        n_lit = spec_n_literals(family_names)
        features: list[list[int]] = []
        for _ in range(n_feat):
            (order,) = struct.unpack_from("<I", blob, off)
            off += 4
            if order == 0:
                raise ValueError("conjunction of order 0")
            members = np.frombuffer(
                blob, dtype="<u4", count=order, offset=off
            ).astype(np.int64)
            off += 4 * order
            ml = members.tolist()
            if any(m >= n_lit for m in ml):
                raise ValueError("member literal id out of range")
            if ml != sorted(set(ml)):
                raise ValueError("members not sorted/deduped")
            features.append(ml)
        w = np.frombuffer(
            blob, dtype="<i4", count=N_STAGES * n_feat, offset=off
        ).astype(np.int64)
        off += 4 * N_STAGES * n_feat
        if off != len(blob):
            raise ValueError("trailing bytes after GLM1 body")
        return cls(
            family_names=family_names,
            features=features,
            w=w.reshape(N_STAGES, n_feat).copy(),
        )

    @classmethod
    def load(cls, path: str) -> GlemModel:
        with open(path, "rb") as fh:
            return cls.from_bytes(fh.read())
