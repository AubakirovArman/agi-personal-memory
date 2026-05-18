"""Compatibility facade for WAL v2 grammar helpers."""
from __future__ import annotations

from .grammar_core import (
    WAL_BNF,
    ParsedProgram,
    ParsedStream,
    WALParseError,
    _tokenize_line,
    format_program_stream,
    format_unique_programs,
    parse_program_stream,
)
from .grammar_text_io import format_wal_text, parse_wal_text

__all__ = [
    "WAL_BNF",
    "ParsedProgram",
    "ParsedStream",
    "WALParseError",
    "_tokenize_line",
    "format_program_stream",
    "format_unique_programs",
    "format_wal_text",
    "parse_program_stream",
    "parse_wal_text",
]
