"""Governance helpers for Path B patch artifacts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .patch_artifact import PatchArtifact


@dataclass
class PatchAuditEvent:
    event_id: str
    patch_id: str
    actor: str
    action: str
    timestamp: str
    previous_hash: str
    event_hash: str
    metadata: dict[str, Any]


class PatchGovernance:
    """Signatures, ACL checks, and audit chain for patch lifecycle events."""

    def __init__(self, signing_key: str = "agim-patch-governance"):
        self.signing_key = signing_key
        self._acl: dict[str, dict[str, set[str]]] = {}
        self._events: list[PatchAuditEvent] = []

    def sign_artifact(self, artifact: PatchArtifact, signer: str) -> dict[str, str]:
        digest = artifact_digest(artifact)
        signature = _sha({"digest": digest, "signer": signer, "key": self.signing_key})
        payload = {"digest": digest, "signer": signer, "signature": signature}
        artifact.metadata["signature"] = payload
        self.audit("signed", artifact.patch_id, signer, {"digest": digest})
        return payload

    def verify_signature(self, artifact: PatchArtifact) -> bool:
        payload = artifact.metadata.get("signature", {})
        signer = payload.get("signer")
        signature = payload.get("signature")
        digest = payload.get("digest")
        if not signer or not signature or digest != artifact_digest(artifact):
            return False
        expected = _sha({"digest": digest, "signer": signer, "key": self.signing_key})
        return signature == expected

    def grant(self, patch_id: str, actor: str, action: str) -> None:
        actions = self._acl.setdefault(patch_id, {}).setdefault(actor, set())
        actions.add(action)

    def check_access(self, patch_id: str, actor: str, action: str) -> bool:
        patch_acl = self._acl.get(patch_id, {})
        return action in patch_acl.get(actor, set()) or action in patch_acl.get("*", set())

    def audit(self, action: str, patch_id: str, actor: str,
              metadata: dict[str, Any] | None = None) -> PatchAuditEvent:
        previous = self._events[-1].event_hash if self._events else "0" * 64
        timestamp = datetime.now(timezone.utc).isoformat()
        base = {
            "patch_id": patch_id,
            "actor": actor,
            "action": action,
            "timestamp": timestamp,
            "previous_hash": previous,
            "metadata": metadata or {},
        }
        event_hash = _sha(base)
        event = PatchAuditEvent(
            event_id=f"event-{len(self._events) + 1}",
            event_hash=event_hash,
            **base,
        )
        self._events.append(event)
        return event

    def verify_audit_chain(self) -> bool:
        previous = "0" * 64
        for event in self._events:
            if event.previous_hash != previous:
                return False
            payload = {
                "patch_id": event.patch_id,
                "actor": event.actor,
                "action": event.action,
                "timestamp": event.timestamp,
                "previous_hash": event.previous_hash,
                "metadata": event.metadata,
            }
            if event.event_hash != _sha(payload):
                return False
            previous = event.event_hash
        return True

    def audit_trail(self) -> list[dict[str, Any]]:
        return [event.__dict__.copy() for event in self._events]


def artifact_digest(artifact: PatchArtifact) -> str:
    payload = artifact.to_dict()
    metadata = dict(payload.get("metadata", {}))
    metadata.pop("signature", None)
    payload["metadata"] = metadata
    return _sha(payload)


def _sha(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
