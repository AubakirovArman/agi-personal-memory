#!/usr/bin/env python3
"""Compatibility facade for WAL v1 meta-learning helpers."""
from __future__ import annotations

from .meta_adapters import WALAtomAdapter, WALCoeffAdapter, WALProgramAdapter
from .meta_evolve import evolve_programs
from .meta_gradient import compute_program_gradient
from .meta_soup import program_soup

__all__ = [
    "WALAtomAdapter",
    "WALCoeffAdapter",
    "WALProgramAdapter",
    "compute_program_gradient",
    "evolve_programs",
    "program_soup",
]
