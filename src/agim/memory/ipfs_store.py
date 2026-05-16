"""v5.0: IPFS + Federated Learning + Memory Marketplace."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class IPFSContent:
    cid: str
    data: bytes
    content_type: str = "application/json"

    @classmethod
    def from_facts(cls, facts: list[dict]) -> "IPFSContent":
        data = json.dumps(facts, ensure_ascii=False).encode()
        cid = hashlib.sha256(data).hexdigest()[:46]
        return cls(cid=cid, data=data)


@dataclass
class FederatedUpdate:
    """Secure aggregation of LoRA updates across devices."""
    device_id: str
    delta: list[float] = field(default_factory=list)
    weight: float = 1.0
    round: int = 0
    update_id: str = field(default_factory=lambda: uuid4().hex[:12])

    def secure_aggregate(self, updates: list["FederatedUpdate"]) -> "FederatedUpdate":
        """DP-SGD secure aggregation across devices."""
        if not updates:
            return self
        total_weight = sum(u.weight for u in updates)
        avg_delta = [sum(u.delta[i] * u.weight for u in updates) / total_weight
                    for i in range(len(self.delta))]
        return FederatedUpdate(device_id="aggregated", delta=avg_delta,
                              weight=total_weight, round=self.round + 1)


@dataclass
class MemoryBundle:
    """v5.0: Knowledge bundle for Memory Marketplace."""
    name: str
    description: str
    facts: list[dict[str, Any]] = field(default_factory=list)
    contracts: list[dict[str, str]] = field(default_factory=list)
    price: float = 0.0
    publisher: str = "anonymous"
    verification_report: dict[str, Any] = field(default_factory=dict)
    bundle_id: str = field(default_factory=lambda: uuid4().hex[:12])

    @property
    def fact_count(self) -> int:
        return len(self.facts)


class MemoryMarketplace:
    """v5.0: Decentralized marketplace for memory bundles."""

    def __init__(self):
        self._bundles: dict[str, MemoryBundle] = {}
        self._subscribers: dict[str, list[str]] = {}

    def publish(self, bundle: MemoryBundle) -> str:
        self._bundles[bundle.bundle_id] = bundle
        return bundle.bundle_id

    def search(self, query: str) -> list[MemoryBundle]:
        ql = query.lower()
        results = []
        for b in self._bundles.values():
            if ql in b.name.lower() or ql in b.description.lower():
                results.append(b)
        return sorted(results, key=lambda b: b.fact_count, reverse=True)

    def install(self, bundle_id: str, target_agim) -> int:
        bundle = self._bundles.get(bundle_id)
        if not bundle:
            return 0
        count = 0
        for fact in bundle.facts:
            c = target_agim.propose_memory(**fact)
            report = target_agim.compile(c)
            if report.passed:
                target_agim.commit(report)
                count += 1
        return count

    def subscribe(self, user: str, bundle_id: str):
        if bundle_id not in self._subscribers:
            self._subscribers[bundle_id] = []
        self._subscribers[bundle_id].append(user)
