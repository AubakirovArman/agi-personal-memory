"""v2.0: Temporal Knowledge Graph — structured fact storage with relations."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class Entity:
    name: str
    entity_type: str = "unknown"
    properties: dict[str, Any] = field(default_factory=dict)
    entity_id: str = field(default_factory=lambda: uuid4().hex[:12])


@dataclass
class Relation:
    subject_id: str
    predicate: str
    object_id: str
    confidence: float = 1.0
    valid_from: str | None = None
    valid_until: str | None = None
    source: str = "user"
    relation_id: str = field(default_factory=lambda: uuid4().hex[:12])
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class KnowledgeGraph:
    """Temporal knowledge graph for storing structured, relation-aware facts."""

    def __init__(self, path: str | Path | None = None):
        self.entities: dict[str, Entity] = {}
        self.relations: list[Relation] = []
        self._path = Path(path) if path else None
        self._load()

    def _load(self):
        if self._path and self._path.exists():
            d = json.loads(self._path.read_text())
            self.entities = {k: Entity(**v) for k, v in d.get("entities", {}).items()}
            self.relations = [Relation(**r) for r in d.get("relations", [])]

    def _save(self):
        if self._path:
            self._path.write_text(json.dumps({
                "entities": {k: {"name": e.name, "entity_type": e.entity_type,
                                "properties": e.properties, "entity_id": e.entity_id}
                            for k, e in self.entities.items()},
                "relations": [{"subject_id": r.subject_id, "predicate": r.predicate,
                              "object_id": r.object_id, "confidence": r.confidence,
                              "valid_from": r.valid_from, "valid_until": r.valid_until,
                              "source": r.source, "relation_id": r.relation_id}
                             for r in self.relations],
            }, indent=2, ensure_ascii=False, default=str))

    def add_fact(self, subject: str, predicate: str, obj: str,
                 confidence: float = 1.0, source: str = "user") -> Relation:
        subj_id = self._get_or_create_entity(subject)
        obj_id = self._get_or_create_entity(obj)
        rel = Relation(subject_id=subj_id, predicate=predicate,
                      object_id=obj_id, confidence=confidence, source=source)
        self.relations.append(rel)
        self._save()
        return rel

    def _get_or_create_entity(self, name: str) -> str:
        for eid, entity in self.entities.items():
            if entity.name.lower() == name.lower():
                return eid
        entity = Entity(name=name)
        self.entities[entity.entity_id] = entity
        return entity.entity_id

    def query(self, subject: str | None = None, predicate: str | None = None,
              obj: str | None = None) -> list[Relation]:
        results = self.relations
        if subject:
            eids = [eid for eid, e in self.entities.items()
                   if subject.lower() in e.name.lower()]
            results = [r for r in results if r.subject_id in eids]
        if predicate:
            results = [r for r in results if predicate.lower() in r.predicate.lower()]
        if obj:
            eids = [eid for eid, e in self.entities.items()
                   if obj.lower() in e.name.lower()]
            results = [r for r in results if r.object_id in eids]
        return results

    def traverse(self, start_entity: str, max_depth: int = 3) -> list[list[Relation]]:
        paths = []
        eids = [eid for eid, e in self.entities.items()
               if start_entity.lower() in e.name.lower()]
        for eid in eids:
            for rel in self.relations:
                if rel.subject_id == eid:
                    paths.append([rel])
        return paths

    @property
    def num_entities(self) -> int:
        return len(self.entities)

    @property
    def num_relations(self) -> int:
        return len(self.relations)
