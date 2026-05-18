#!/usr/bin/env python3
"""Compatibility facade for WAL QAT helpers."""
from __future__ import annotations

from .qat_convert import count_qat_parameters, linear_to_qat, model_to_qat
from .qat_layers import WALQATLinear

__all__ = [
    "WALQATLinear",
    "count_qat_parameters",
    "linear_to_qat",
    "model_to_qat",
]
