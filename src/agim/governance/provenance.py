"""Provenance chain — cryptographic signing of memory commits."""
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ProvenanceRecord:
    commit_id: str
    previous_hash: str
    data_hash: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    signature: str = ""

    def compute_data_hash(self, question: str, answer: str, tier: str,
                          metadata: dict | None = None) -> str:
        payload = json.dumps({
            "question": question, "answer": answer, "tier": tier,
            "metadata": metadata or {},
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def compute_chain_hash(self) -> str:
        return hashlib.sha256(
            f"{self.previous_hash}:{self.data_hash}:{self.timestamp}".encode()
        ).hexdigest()[:16]

    def sign(self, private_key: str):
        self.signature = hashlib.sha256(
            f"{self.compute_chain_hash()}:{private_key}".encode()
        ).hexdigest()[:32]

    def verify(self, private_key: str) -> bool:
        expected = hashlib.sha256(
            f"{self.compute_chain_hash()}:{private_key}".encode()
        ).hexdigest()[:32]
        return self.signature == expected


class ProvenanceChain:
    def __init__(self, path: str | Path, signing_key: str = "agim-default-key"):
        self.path = Path(path)
        self.signing_key = signing_key
        self.chain: list[ProvenanceRecord] = []
        self._load()

    def _load(self):
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.chain = [ProvenanceRecord(**r) for r in data]

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([
            {"commit_id": r.commit_id, "previous_hash": r.previous_hash,
             "data_hash": r.data_hash, "timestamp": r.timestamp,
             "signature": r.signature}
            for r in self.chain
        ], indent=2))

    def add(self, commit_id: str, question: str, answer: str,
            tier: str, metadata: dict | None = None) -> ProvenanceRecord:
        prev = self.chain[-1].compute_chain_hash() if self.chain else "0" * 16
        rec = ProvenanceRecord(commit_id=commit_id, previous_hash=prev, data_hash="")
        rec.data_hash = rec.compute_data_hash(question, answer, tier, metadata)
        rec.sign(self.signing_key)
        self.chain.append(rec)
        self._save()
        return rec

    def verify_chain(self) -> bool:
        for i in range(1, len(self.chain)):
            if self.chain[i].previous_hash != self.chain[i - 1].compute_chain_hash():
                return False
            if not self.chain[i].verify(self.signing_key):
                return False
        return True

    @property
    def length(self) -> int:
        return len(self.chain)
