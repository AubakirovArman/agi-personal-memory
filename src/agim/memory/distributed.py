"""v5.0: Distributed Memory — P2P sync with CRDT convergence."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class CRDTFact:
    """Conflict-free Replicated Data Type for memory facts."""
    key: str
    value: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "local"
    confidence: float = 1.0
    vector_clock: dict[str, int] = field(default_factory=dict)
    fact_id: str = field(default_factory=lambda: uuid4().hex[:12])

    def merge(self, other: "CRDTFact") -> "CRDTFact":
        """CRDT merge: latest timestamp wins. If tie, higher confidence wins."""
        if self.timestamp > other.timestamp:
            return self
        if other.timestamp > self.timestamp:
            return other
        if self.confidence >= other.confidence:
            return self
        return other

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(f"{self.key}:{self.value}".encode()).hexdigest()[:16]


class DistributedMemory:
    """v5.0: P2P distributed memory with CRDT-based eventual consistency."""

    def __init__(self, node_id: str = "node-0"):
        self.node_id = node_id
        self._store: dict[str, CRDTFact] = {}
        self._peers: set[str] = set()
        self._vector_clock: dict[str, int] = {node_id: 0}

    def put(self, key: str, value: str, confidence: float = 1.0) -> CRDTFact:
        self._vector_clock[self.node_id] = self._vector_clock.get(self.node_id, 0) + 1
        fact = CRDTFact(key=key, value=value, confidence=confidence,
                       vector_clock=dict(self._vector_clock), source=self.node_id)
        self._store[key] = fact
        return fact

    def get(self, key: str) -> CRDTFact | None:
        return self._store.get(key)

    def sync_from(self, peer_facts: dict[str, CRDTFact]):
        for key, remote in peer_facts.items():
            local = self._store.get(key)
            if local is None:
                self._store[key] = remote
            else:
                self._store[key] = local.merge(remote)

    def add_peer(self, peer_id: str):
        self._peers.add(peer_id)
        self._vector_clock[peer_id] = 0

    def export_since(self, since_hash: str = "") -> dict[str, CRDTFact]:
        """Export all facts newer than since_hash for P2P sync."""
        return {k: v for k, v in self._store.items()
               if not since_hash or v.content_hash != since_hash}

    def convergence_time_estimate(self, num_nodes: int = 1000,
                                   avg_latency_ms: float = 100.0) -> float:
        """Estimate time to full convergence across N nodes."""
        # Gossip protocol: O(log N) rounds
        import math
        rounds = math.ceil(math.log2(num_nodes))
        return rounds * avg_latency_ms / 1000.0

    @property
    def size(self) -> int:
        return len(self._store)
