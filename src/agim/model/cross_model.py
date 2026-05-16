"""v9.0: Cross-Model Memory Transfer — model-agnostic knowledge portability."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class ModelAgnosticFact:
    """Fact in model-independent representation for cross-model transfer."""
    subject: str
    predicate: str
    obj: str
    confidence: float = 1.0
    source_model: str = ""
    fact_id: str = field(default_factory=lambda: uuid4().hex[:12])

    def to_layer_specific(self, model_family: str, layer_map: dict[str, int]) -> dict:
        """Convert to model-specific layer edit instructions."""
        layer = layer_map.get(self.predicate, layer_map.get("default", 5))
        return {"target_layer": layer, "subject": self.subject,
                "target": self.obj, "relation": self.predicate,
                "source_model": self.source_model}


class CrossModelTransfer:
    """v9.0: Transfers knowledge between different base models."""

    def __init__(self):
        self._abstraction_layer: dict[str, ModelAgnosticFact] = {}
        self._layer_maps: dict[str, dict[str, int]] = {
            "llama-3-70b": {"capital": 5, "located_in": 7, "born_in": 6, "default": 5},
            "qwen-3-72b": {"capital": 6, "located_in": 8, "born_in": 7, "default": 6},
            "gemma-4-31b": {"capital": 4, "located_in": 6, "born_in": 5, "default": 4},
        }

    def abstract(self, fact: dict, source_model: str) -> ModelAgnosticFact:
        mf = ModelAgnosticFact(
            subject=fact.get("subject", fact.get("question", "")),
            predicate=fact.get("predicate", fact.get("relation", "is")),
            obj=fact.get("obj", fact.get("answer", "")),
            confidence=fact.get("confidence", 1.0),
            source_model=source_model,
        )
        self._abstraction_layer[mf.fact_id] = mf
        return mf

    def transfer(self, fact_id: str, target_model: str) -> dict | None:
        mf = self._abstraction_layer.get(fact_id)
        if not mf:
            return None
        layer_map = self._layer_maps.get(target_model, self._layer_maps.get("gemma-4-31b", {}))
        return mf.to_layer_specific(target_model, layer_map)

    def transfer_all(self, target_model: str) -> list[dict]:
        return [self.transfer(fid, target_model) for fid in self._abstraction_layer
                if self._abstraction_layer[fid].source_model != target_model]

    def export_abstractions(self, path: Path):
        path.write_text(json.dumps([
            {"fact_id": mf.fact_id, "subject": mf.subject,
             "predicate": mf.predicate, "object": mf.obj,
             "source_model": mf.source_model, "confidence": mf.confidence}
            for mf in self._abstraction_layer.values()
        ], indent=2, ensure_ascii=False))
