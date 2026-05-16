"""Core state types for AGI Personal Memory."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4


class Intent(StrEnum):
    FACT_TEACH = "fact_teach"
    FACT_CORRECT = "fact_correct"
    FACT_QUESTION = "fact_question"
    PREFERENCE = "preference"
    FEEDBACK = "feedback"
    FORGET = "forget"
    HISTORY = "history"
    STATS = "stats"
    UNKNOWN = "unknown"


class MemoryTier(StrEnum):
    WAL_RECIPE = "wal_recipe"
    RETRIEVAL = "retrieval"
    LORA_ADAPTER = "lora_adapter"
    REFUSAL = "refusal"
    REJECT = "reject"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class MemoryCandidate:
    question: str
    answer: str
    kind: str = "fact_teach"
    source: str = "user"
    confidence: float = 1.0
    previous_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    candidate_id: str = field(default_factory=lambda: uuid4().hex[:12])
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    reason: str = ""


@dataclass(frozen=True)
class CompileReport:
    candidate: MemoryCandidate
    tier: MemoryTier
    passed: bool
    gates: tuple[GateResult, ...]
    artifact_id: str | None = None
    reason: str = ""
    created_at: str = field(default_factory=utc_now)

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


@dataclass(frozen=True)
class AIGIResponse:
    question: str
    answer: str
    source: str
    memory_id: str | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class CommitRecord:
    artifact_id: str
    tier: MemoryTier
    question: str
    answer: str
    previous_entry: dict[str, str] | None = None
    rolled_back: bool = False
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class MemoryStats:
    total_facts: int
    facts_by_tier: dict[str, int]
    facts_by_kind: dict[str, int]
    rollback_count: int
    total_commits: int
    avg_confidence: float
