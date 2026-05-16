"""v2.5: Curriculum Generator + Knowledge Prioritization (PageRank on KG)."""
from __future__ import annotations

from collections import defaultdict

from ..memory.knowledge_graph import KnowledgeGraph, Relation


class PageRankPrioritizer:
    """PageRank-inspired knowledge prioritization on knowledge graph."""

    def __init__(self, kg: KnowledgeGraph, damping: float = 0.85, iters: int = 20):
        self.kg = kg
        self.damping = damping
        self.iters = iters

    def rank(self) -> dict[str, float]:
        """Compute PageRank scores for all entities."""
        if not self.kg.entities:
            return {}
        eids = list(self.kg.entities.keys())
        n = len(eids)
        scores = {eid: 1.0 / n for eid in eids}
        adj: dict[str, list[str]] = defaultdict(list)
        for rel in self.kg.relations:
            adj[rel.subject_id].append(rel.object_id)

        for _ in range(self.iters):
            new_scores = {eid: (1 - self.damping) / n for eid in eids}
            for eid in eids:
                if eid in adj:
                    share = scores[eid] / max(len(adj[eid]), 1)
                    for target in adj[eid]:
                        new_scores[target] = new_scores.get(target, 0) + self.damping * share
            scores = new_scores
        return scores

    def top_entities(self, k: int = 20) -> list[tuple[str, float]]:
        scores = self.rank()
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(self.kg.entities[eid].name, score) for eid, score in ranked[:k]]


class CurriculumGenerator:
    """v2.5: Generates optimal learning order — simple to complex."""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.prioritizer = PageRankPrioritizer(kg)

    def generate_curriculum(self, topic: str, max_items: int = 10) -> list[str]:
        """Generate learning order: foundational concepts first."""
        top = self.prioritizer.top_entities(100)
        topic_eids = [eid for eid, e in self.kg.entities.items()
                     if topic.lower() in e.name.lower()]
        if not topic_eids:
            return [e[0] for e in top[:max_items]]

        depths: dict[str, int] = {}
        queue = [(topic_eids[0], 0)]
        visited = set()
        while queue:
            current, depth = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            depths[current] = depth
            for rel in self.kg.relations:
                if rel.subject_id == current and rel.object_id not in visited:
                    queue.append((rel.object_id, depth + 1))
                if rel.object_id == current and rel.subject_id not in visited:
                    queue.append((rel.subject_id, depth + 1))

        sorted_items = sorted(depths.items(), key=lambda x: x[1])
        return [self.kg.entities[eid].name for eid, _ in sorted_items[:max_items]]

    def suggest_next(self, learned: set[str], max_suggestions: int = 5) -> list[str]:
        """Suggest what to learn next based on what's already known."""
        top = self.prioritizer.top_entities(50)
        candidates = []
        for name, score in top:
            if name.lower() not in learned:
                prerequisites = self._count_prerequisites(name, learned)
                candidates.append((name, score * (1 + prerequisites)))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:max_suggestions]]

    def _count_prerequisites(self, entity_name: str, learned: set[str]) -> int:
        eids = [eid for eid, e in self.kg.entities.items()
               if entity_name.lower() in e.name.lower()]
        if not eids:
            return 0
        count = 0
        for rel in self.kg.relations:
            if rel.object_id == eids[0]:
                prereq_name = self.kg.entities.get(rel.subject_id)
                if prereq_name and prereq_name.name.lower() in learned:
                    count += 1
        return count
