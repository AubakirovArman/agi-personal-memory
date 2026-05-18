"""Compatibility facade for WAL v1 binary format helpers."""
from __future__ import annotations

from .format_deserialize import deserialize_wal_v1
from .format_helpers import (
    FLAG_HAS_RESIDUALS,
    FLAG_PACKED_COEFFS,
    MAGIC,
    OP_ADD,
    OP_CONST,
    OP_MUL,
    OP_NEG,
    VERSION,
    _OP_DECODE,
    _OP_ENCODE,
    _deserialize_hierarchical,
    _pack_bitmap,
    _pack_uint4,
    _serialize_hierarchical,
    _unpack_bitmap,
    _unpack_uint4,
)
from .format_serialize import serialize_wal_v1

__all__ = [
    "FLAG_HAS_RESIDUALS",
    "FLAG_PACKED_COEFFS",
    "MAGIC",
    "OP_ADD",
    "OP_CONST",
    "OP_MUL",
    "OP_NEG",
    "VERSION",
    "_OP_DECODE",
    "_OP_ENCODE",
    "_deserialize_hierarchical",
    "_pack_bitmap",
    "_pack_uint4",
    "_serialize_hierarchical",
    "_unpack_bitmap",
    "_unpack_uint4",
    "deserialize_wal_v1",
    "serialize_wal_v1",
]
