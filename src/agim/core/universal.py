"""v10.0: Universal AGI Memory Substrate — open standard for AI memory."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4


AGIM_MEM_VERSION = "1.0.0"


@dataclass
class AGIMSpec:
    """AGIM-MEM: Universal memory protocol specification (v10.0)."""
    version: str = AGIM_MEM_VERSION
    spec_url: str = "https://github.com/AubakirovArman/agi-personal-memory"
    operations: list[str] = field(default_factory=lambda: [
        "memory.propose", "memory.compile", "memory.commit",
        "memory.rollback", "memory.ask", "memory.search",
        "memory.teach", "memory.verify", "memory.history", "memory.stats",
    ])
    tiers: list[str] = field(default_factory=lambda: [
        "wal_recipe", "retrieval", "lora_adapter", "refusal", "reject",
    ])
    safety_requirements: list[str] = field(default_factory=lambda: [
        "verification_gates", "provenance_chain", "constitutional_principles",
        "risk_ledger", "regression_suite",
    ])


@dataclass
class MemoryFormat:
    """Standard AGIM-MEM format for inter-system memory exchange."""
    format_version: str = AGIM_MEM_VERSION
    facts: list[dict[str, Any]] = field(default_factory=list)
    provenance: list[dict[str, str]] = field(default_factory=list)
    verification: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    export_id: str = field(default_factory=lambda: uuid4().hex[:12])

    def to_json(self) -> str:
        return json.dumps({
            "format": "AGIM-MEM",
            "version": self.format_version,
            "facts": self.facts,
            "provenance": self.provenance,
            "verification": self.verification,
            "metadata": self.metadata,
            "export_id": self.export_id,
        }, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> "MemoryFormat":
        d = json.loads(data)
        if d.get("format") != "AGIM-MEM":
            raise ValueError(f"Not an AGIM-MEM format: {d.get('format')}")
        return cls(
            format_version=d["version"],
            facts=d.get("facts", []),
            provenance=d.get("provenance", []),
            verification=d.get("verification", []),
            metadata=d.get("metadata", {}),
            export_id=d.get("export_id", ""),
        )


class UniversalMemorySubstrate:
    """v10.0: Universal substrate — any AI system can plug into AGIM memory."""

    def __init__(self, spec: AGIMSpec | None = None):
        self.spec = spec or AGIMSpec()
        self._clients: dict[str, Any] = {}
        self._orchestrator: dict[str, str] = {}

    def register_model(self, model_id: str, model_type: str,
                       tiers: list[str] | None = None):
        self._clients[model_id] = {
            "type": model_type,
            "tiers": tiers or self.spec.tiers,
            "registered": True,
        }

    def route_memory(self, model_id: str, tier: str) -> bool:
        if model_id not in self._clients:
            return False
        return tier in self._clients[model_id]["tiers"]

    def orchestrate(self, edge_model: str, cloud_model: str):
        """Set up hierarchical memory sync between edge and cloud models."""
        self._orchestrator[edge_model] = cloud_model

    def export_spec(self, path: str | Path):
        Path(path).write_text(json.dumps({
            "protocol": "AGIM-MEM",
            "version": AGIM_MEM_VERSION,
            "spec": {
                "operations": self.spec.operations,
                "tiers": self.spec.tiers,
                "safety": self.spec.safety_requirements,
            },
            "clients": len(self._clients),
        }, indent=2))
