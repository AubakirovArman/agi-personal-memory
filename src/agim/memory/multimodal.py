"""v4.0: Multimodal Memory — unified representation for text, image, audio, video."""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class Modality:
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class MultimodalAtomicUnit:
    """MAU — unified representation for any data type."""
    summary: str
    embedding: list[float] | None = None
    pointer: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    modality: str = Modality.TEXT
    links: list[str] = field(default_factory=list)
    mau_id: str = field(default_factory=lambda: uuid4().hex[:12])
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.summary.encode()).hexdigest()[:16]


class MultimodalMemory:
    """Unified multimodal memory store."""

    def __init__(self, path: str | Path | None = None):
        self.units: dict[str, MultimodalAtomicUnit] = {}
        self._path = Path(path) if path else None
        self._load()

    def _load(self):
        if self._path and self._path.exists():
            d = json.loads(self._path.read_text())
            self.units = {k: MultimodalAtomicUnit(**v) for k, v in d.items()}

    def _save(self):
        if self._path:
            self._path.write_text(json.dumps(
                {k: {"summary": u.summary, "embedding": u.embedding,
                     "pointer": u.pointer, "timestamp": u.timestamp,
                     "modality": u.modality, "links": u.links,
                     "mau_id": u.mau_id, "metadata": u.metadata}
                 for k, u in self.units.items()},
                indent=2, ensure_ascii=False, default=str))

    def store(self, summary: str, modality: str = Modality.TEXT,
              pointer: str = "", embedding: list[float] | None = None,
              links: list[str] | None = None) -> MultimodalAtomicUnit:
        mau = MultimodalAtomicUnit(summary=summary, modality=modality,
                                   pointer=pointer, embedding=embedding,
                                   links=links or [])
        self.units[mau.mau_id] = mau
        self._save()
        return mau

    def search(self, query: str, modality: str | None = None,
               max_results: int = 10) -> list[MultimodalAtomicUnit]:
        results = []
        ql = query.lower()
        for mau in self.units.values():
            if modality and mau.modality != modality:
                continue
            if ql in mau.summary.lower():
                results.append(mau)
        results.sort(key=lambda u: len(u.summary))
        return results[:max_results]

    def get_links(self, mau_id: str) -> list[MultimodalAtomicUnit]:
        mau = self.units.get(mau_id)
        if not mau:
            return []
        return [self.units[lid] for lid in mau.links if lid in self.units]

    @property
    def count_by_modality(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for mau in self.units.values():
            counts[mau.modality] = counts.get(mau.modality, 0) + 1
        return counts
