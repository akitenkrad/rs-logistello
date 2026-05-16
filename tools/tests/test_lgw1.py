"""LGW1 Python (de)serialization + pack-table self-consistency tests.

The cross-language byte-identity with Rust is proven by the Rust GOLD
interop test (``tests/eval_interop_test.rs``); here we only assert the
Python side is internally consistent and reproduces the documented layout.
"""
from __future__ import annotations

import struct

import numpy as np

from logistello_tools.lgw1 import (
    EVAL_PACKED_SIZE,
    N_STAGES,
    N_TYPES,
    WEIGHTS_MAGIC,
    EvalWeights,
    build_pack_tables,
)


def test_pack_tables_match_eval_packed_size():
    tables = build_pack_tables()
    from logistello_tools.lgw1 import TYPE_ORDER

    for idx, (name, _k) in enumerate(TYPE_ORDER):
        assert tables[name].n_canonical == EVAL_PACKED_SIZE[idx], name


def test_lgw1_header_layout_and_roundtrip():
    w = EvalWeights()
    # Deterministic non-trivial pokes mirroring the Rust round-trip test.
    for s in range(N_STAGES):
        for t in range(N_TYPES):
            v = w.w[s][t]
            idx = np.arange(v.shape[0])
            w.w[s][t] = ((s * 31 + t * 7 + idx) % 257 - 128).astype(np.int64)
    blob = w.to_bytes()
    # Explicit header is hand-checkable little-endian.
    magic, version, n_stages, n_types = struct.unpack_from("<IIII", blob, 0)
    assert magic == WEIGHTS_MAGIC
    assert version == 1
    assert n_stages == N_STAGES
    assert n_types == N_TYPES
    canon_sizes = list(struct.unpack_from("<9I", blob, 16))
    assert canon_sizes == EVAL_PACKED_SIZE
    (data_len,) = struct.unpack_from("<Q", blob, 52)
    assert data_len == w.total_weights == 13 * sum(EVAL_PACKED_SIZE)
    assert len(blob) == 60 + 4 * data_len

    back = EvalWeights.from_bytes(blob)
    for s in range(N_STAGES):
        for t in range(N_TYPES):
            np.testing.assert_array_equal(back.w[s][t], w.w[s][t])


def test_from_bytes_rejects_bad_magic(tmp_path):
    blob = bytearray(EvalWeights().to_bytes())
    blob[0] ^= 0xFF
    try:
        EvalWeights.from_bytes(bytes(blob))
        raise AssertionError("expected bad-magic rejection")
    except ValueError:
        pass
