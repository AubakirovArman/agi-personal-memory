"""Phase 1: LLM-based atomic fact extraction from dialog turns.

Replaces raw turn indexing with structured atomic facts.
Implements Steps 1.1-1.10 from the 50-step plan.
"""
import json, re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class AtomicFact:
    text: str
    subject: str = ""
    relation: str = ""
    obj: str = ""
    fact_type: str = "statement"  # statement, temporal, persona, preference, relation, event
    confidence: float = 1.0
    source_turn: str = ""
    source_speaker: str = ""
    session_id: str = ""
    timestamp: str = ""
    entities: list[str] = field(default_factory=list)
    fact_id: str = field(default_factory=lambda: uuid4().hex[:12])

    def to_search_text(self) -> str:
        if self.subject and self.relation and self.obj:
            return f"{self.subject} {self.relation} {self.obj}"
        return self.text

    def to_context_text(self) -> str:
        parts = [self.text]
        if self.timestamp:
            parts.append(f"[{self.timestamp}]")
        if self.source_speaker:
            parts.append(f"({self.source_speaker})")
        return " ".join(parts)


class LLMFactExtractor:
    """GPT-4o-mini based atomic fact extraction from conversation turns.

    Implements ADD-only strategy (Step 1.2): facts are only added, never updated.
    Extracts: statements, temporal facts, persona, preferences, relations, events.
    """

    EXTRACTION_PROMPT = """Extract ALL atomic facts from this conversation turn as a JSON array.
Each fact must be a JSON object with these fields:
- "text": the fact as a concise statement (required)
- "subject": who/what the fact is about
- "relation": the relationship (graduated_from, works_at, likes, moved_to, visited, etc.)
- "object": the object of the relation
- "fact_type": one of "statement", "temporal", "persona", "preference", "relation", "event"

Rules:
- Extract EVERY fact mentioned, including implied ones
- For temporal facts: include dates, times, sequences
- For persona facts: characteristics, background, identity
- For preferences: likes, dislikes, habits
- For relations: connections between people/places/things
- For events: things that happened with temporal anchors
- Output ONLY the JSON array, nothing else.
- If no facts can be extracted, output [].

Turn: {turn_text}

Facts (JSON array):"""

    def __init__(self, client, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    def extract(self, turn_text: str, speaker: str = "",
                turn_id: str = "", session_id: str = "") -> list[AtomicFact]:
        if len(turn_text.strip()) < 30:
            return []

        prompt = self.EXTRACTION_PROMPT.format(turn_text=turn_text[:500])
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300, temperature=0.0)
            resp = r.choices[0].message.content.strip()
            # Extract JSON array from response
            match = re.search(r'\[.*\]', resp, re.DOTALL)
            if match:
                facts_data = json.loads(match.group())
            else:
                return []
        except Exception:
            return []

        facts = []
        for fd in facts_data:
            if not isinstance(fd, dict) or not fd.get("text"):
                continue
            text = fd["text"].strip()
            if len(text) < 5:
                continue
            facts.append(AtomicFact(
                text=text,
                subject=fd.get("subject", ""),
                relation=fd.get("relation", ""),
                obj=fd.get("object", ""),
                fact_type=fd.get("fact_type", "statement"),
                confidence=0.9,
                source_turn=turn_id,
                source_speaker=speaker,
                session_id=session_id,
                entities=self._extract_entities(fd),
            ))
        return facts

    def _extract_entities(self, fd: dict) -> list[str]:
        entities = []
        for key in ("subject", "object"):
            val = fd.get(key, "")
            if val and len(val) > 1:
                entities.append(val.lower())
        return entities


class FactIndex:
    """Multi-type index for atomic facts with type-aware retrieval."""

    def __init__(self):
        self.facts: list[AtomicFact] = []
        self._type_index: dict[str, list[int]] = {}

    def add(self, fact: AtomicFact):
        idx = len(self.facts)
        self.facts.append(fact)
        ft = fact.fact_type
        if ft not in self._type_index:
            self._type_index[ft] = []
        self._type_index[ft].append(idx)

    def add_all(self, facts: list[AtomicFact]):
        for f in facts:
            self.add(f)

    def get_by_type(self, fact_type: str) -> list[AtomicFact]:
        return [self.facts[i] for i in self._type_index.get(fact_type, [])]

    def get_all_texts(self) -> list[str]:
        return [f.to_search_text() for f in self.facts]

    def get_all_contexts(self) -> list[str]:
        return [f.to_context_text() for f in self.facts]

    def __len__(self) -> int:
        return len(self.facts)

    @property
    def type_counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._type_index.items()}
