#!/usr/bin/env python3
"""Compatibility facade for WAL v1 PyTorch integration."""
from __future__ import annotations

from .nn_encode import encode_linear_weight
from .nn_layers import WALCachedLinear, WALLinear, WALParameter
from .nn_replace import replace_linear_with_wal, replace_wal_with_linear, wal_load_state_dict, wal_state_dict

__all__ = [
    "WALCachedLinear",
    "WALLinear",
    "WALParameter",
    "encode_linear_weight",
    "replace_linear_with_wal",
    "replace_wal_with_linear",
    "wal_load_state_dict",
    "wal_state_dict",
]
