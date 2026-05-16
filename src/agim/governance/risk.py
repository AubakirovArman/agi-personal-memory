"""Risk ledger — track and assess risk of each memory change."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RiskEntry:
    memory_id: str
    risk_score: float
    reason: str
    auto_rollback: bool = False


@dataclass
class RiskLedger:
    entries: list[RiskEntry] = field(default_factory=list)
    dangerous_threshold: float = 8.0
    risky_threshold: float = 5.0

    def assess(self, memory_id: str, metadata: dict[str, Any]) -> RiskEntry:
        score = 0.0
        reasons = []
        if metadata.get("kind") == "fact_correct":
            score += 2.0
            reasons.append("correction_overrides_previous +2")
        if metadata.get("confidence", 1.0) < 0.7:
            score += 3.0
            reasons.append("low_confidence +3")
        if any(kw in str(metadata).lower() for kw in ["password", "token", "injection"]):
            score += 5.0
            reasons.append("sensitive_content +5")
        auto = score >= self.dangerous_threshold
        entry = RiskEntry(memory_id=memory_id, risk_score=score,
                         reason="; ".join(reasons) or "low_risk", auto_rollback=auto)
        self.entries.append(entry)
        return entry

    @property
    def total_risky(self) -> int:
        return sum(1 for e in self.entries if e.risk_score >= self.risky_threshold)

    @property
    def total_dangerous(self) -> int:
        return sum(1 for e in self.entries if e.auto_rollback)

    def save(self, path: Path):
        path.write_text(json.dumps([
            {"memory_id": e.memory_id, "risk_score": e.risk_score,
             "reason": e.reason, "auto_rollback": e.auto_rollback}
            for e in self.entries
        ], indent=2))

    @classmethod
    def load(cls, path: Path) -> "RiskLedger":
        if path.exists():
            entries = [RiskEntry(**e) for e in json.loads(path.read_text())]
            return cls(entries=entries)
        return cls()
