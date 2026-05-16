"""Structured Memory Extractor — Pydantic-based fact extraction with relations."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from ..core.state import Intent, MemoryCandidate


@dataclass
class StructuredFact:
    """Rich fact representation with relations, sources, and temporal validity."""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = "user"
    source_url: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    language: str = "en"
    tags: list[str] = field(default_factory=list)
    fact_id: str = field(default_factory=lambda: uuid4().hex[:12])
    extracted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_question_answer(self) -> tuple[str, str]:
        if self.predicate in ("is", "are", "was", "were"):
            question = f"What is {self.object}?"
            answer = self.subject
        else:
            question = f"What is the {self.predicate} of {self.subject}?"
            answer = self.object
        return question, answer

    def to_candidate(self, kind: str = "fact_teach") -> MemoryCandidate:
        q, a = self.to_question_answer()
        return MemoryCandidate(question=q, answer=a, kind=kind,
                              source=self.source, confidence=self.confidence,
                              metadata={"fact_id": self.fact_id, "predicate": self.predicate,
                                       "subject": self.subject, "object": self.object,
                                       "valid_from": self.valid_from,
                                       "valid_until": self.valid_until})


class StructuredExtractor:
    """Extract structured facts with relations, not just Q&A pairs.

    Handles patterns:
    - "X is Y" → subject=X, predicate=is, object=Y
    - "X has Y" → subject=X, predicate=has, object=Y
    - "X was born in Y" → subject=X, predicate=born_in, object=Y
    - "The capital of X is Y" → subject=X, predicate=capital, object=Y
    """

    RELATION_PATTERNS = [
        (" is the capital of ", "capital"),
        (" is a ", "is_a"),
        (" was born in ", "born_in"),
        (" was born on ", "born_on"),
        (" died in ", "died_in"),
        (" died on ", "died_on"),
        (" has a population of ", "population"),
        (" is located in ", "located_in"),
        (" was founded in ", "founded_in"),
        (" speaks ", "language"),
        (" invented ", "invented"),
        (" discovered ", "discovered"),
        (" wrote ", "wrote"),
        (" is married to ", "married_to"),
    ]

    CORRECTION_PATTERNS = [
        ("no, ", ""), ("actually, ", ""), ("wrong, ", ""),
        ("that's not right, ", ""), ("it's not ", ""),
        ("not correct, ", ""),
    ]

    def extract(self, text: str, intent: Intent) -> StructuredFact | None:
        t = text.strip()
        if intent == Intent.FACT_CORRECT:
            t = self._clean_correction(t)
        for pattern, predicate in self.RELATION_PATTERNS:
            if pattern in t.lower():
                parts = t.lower().split(pattern, 1)
                return StructuredFact(
                    subject=parts[0].strip(),
                    predicate=predicate,
                    object=parts[1].strip().rstrip("."),
                )
        # Fallback: "X is Y"
        for separator in [" is ", " are ", " was ", " were "]:
            if separator in t.lower():
                parts = t.lower().split(separator, 1)
                return StructuredFact(
                    subject=parts[0].strip(),
                    predicate=separator.strip(),
                    object=parts[1].strip().rstrip("."),
                )
        # Unknown pattern — store as raw
        return StructuredFact(subject="unknown", predicate="stated", object=t)

    def _clean_correction(self, text: str) -> str:
        for pattern, replacement in self.CORRECTION_PATTERNS:
            if text.lower().startswith(pattern):
                text = text[len(pattern):]
        return text

    def extract_candidate(self, text: str, intent: Intent) -> MemoryCandidate:
        fact = self.extract(text, intent)
        if fact is None:
            return MemoryCandidate(question=text, answer="[recorded]", kind=intent.value)
        return fact.to_candidate(kind=intent.value)
