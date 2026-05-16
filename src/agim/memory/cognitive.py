"""v8.0: Cognitive Memory — causal reasoning, hypotheses, counterfactuals."""
from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from .knowledge_graph import KnowledgeGraph, Relation


@dataclass
class CausalEdge:
    cause_id: str
    effect_id: str
    strength: float = 1.0
    confidence: float = 0.5
    edge_id: str = field(default_factory=lambda: uuid4().hex[:12])


@dataclass
class Hypothesis:
    statement: str
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    confidence: float = 0.3
    status: str = "unverified"
    hypothesis_id: str = field(default_factory=lambda: uuid4().hex[:12])


@dataclass
class Counterfactual:
    scenario: str
    base_fact: str
    alternative: str
    consequences: list[str] = field(default_factory=list)
    is_factual: bool = False


class CausalMemory:
    """v8.0: Causal reasoning layer on top of knowledge graph."""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.causal_edges: list[CausalEdge] = []
        self.hypotheses: list[Hypothesis] = []

    def infer_causal(self, cause_entity: str, effect_entity: str,
                     strength: float = 1.0) -> CausalEdge:
        cause_eids = [eid for eid, e in self.kg.entities.items()
                     if cause_entity.lower() in e.name.lower()]
        effect_eids = [eid for eid, e in self.kg.entities.items()
                      if effect_entity.lower() in e.name.lower()]
        if cause_eids and effect_eids:
            edge = CausalEdge(cause_id=cause_eids[0], effect_id=effect_eids[0],
                            strength=strength)
            self.causal_edges.append(edge)
            return edge
        return None

    def transitive_closure(self, entity: str, max_depth: int = 3) -> list[CausalEdge]:
        """Find all causal chains starting from entity."""
        eids = [eid for eid, e in self.kg.entities.items()
               if entity.lower() in e.name.lower()]
        if not eids:
            return []
        visited = set()
        chain = []
        queue = [(eids[0], 0)]
        while queue and len(chain) < max_depth * 10:
            current, depth = queue.pop(0)
            if current in visited or depth > max_depth:
                continue
            visited.add(current)
            for edge in self.causal_edges:
                if edge.cause_id == current:
                    chain.append(edge)
                    queue.append((edge.effect_id, depth + 1))
        return chain

    def generate_hypothesis(self, subject: str, predicate: str,
                            obj: str) -> Hypothesis:
        """Generate a hypothesis from observed patterns."""
        h = Hypothesis(
            statement=f"{subject} {predicate} {obj}",
            confidence=0.3,
            status="unverified",
        )
        self.hypotheses.append(h)
        return h

    def create_counterfactual(self, base_fact: tuple[str, str, str],
                              alternative: str) -> Counterfactual:
        return Counterfactual(
            scenario=f"What if {alternative}?",
            base_fact=f"{base_fact[0]} {base_fact[1]} {base_fact[2]}",
            alternative=alternative,
        )

    def detect_concept_drift(self, entity_name: str,
                             current_value: str) -> list[Relation]:
        """Find facts about entity that conflict with current value."""
        conflicts = []
        for rel in self.kg.query(subject=entity_name):
            obj_entity = self.kg.entities.get(rel.object_id)
            if obj_entity and obj_entity.name != current_value:
                conflicts.append(rel)
        return conflicts
