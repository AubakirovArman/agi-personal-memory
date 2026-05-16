"""Commit decision report — auditable summary of each memory change."""
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


@dataclass
class CommitDecisionReport:
    report_id: str = field(default_factory=lambda: uuid4().hex[:12])
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    memory_id: str = ""
    tier: str = ""
    decision: str = ""
    risk_score: float = 0.0
    gates_passed: int = 0
    gates_total: int = 0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "created_at": self.created_at,
            "memory_id": self.memory_id,
            "tier": self.tier,
            "decision": self.decision,
            "risk_score": self.risk_score,
            "gates_passed": f"{self.gates_passed}/{self.gates_total}",
            "reason": self.reason,
        }

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
