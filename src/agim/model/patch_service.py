"""In-process PatchService API for Path B patch lifecycle."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .patch_artifact import PatchArtifact, conflict_summary
from .wal_memit_batch_editor import WALMemitBatchEditor


@dataclass
class PatchRecord:
    artifact: PatchArtifact
    status: str = "proposed"
    canaries: dict[str, bool] = field(default_factory=dict)
    approvals: list[str] = field(default_factory=list)
    backup: dict[str, Any] | None = None


class PatchService:
    """Small service contract for audited patch proposal and rollout."""

    def __init__(self):
        self._records: dict[str, PatchRecord] = {}
        self._applier = WALMemitBatchEditor()

    def propose_patch(self, artifact: PatchArtifact) -> dict[str, Any]:
        if artifact.patch_id in self._records:
            raise ValueError(f"Patch already exists: {artifact.patch_id}")
        self._records[artifact.patch_id] = PatchRecord(artifact=artifact)
        return self.inspect_patch(artifact.patch_id)

    def simulate_patch(self, patch_id: str) -> dict[str, Any]:
        record = self._record(patch_id)
        conflicts = [
            conflict_summary(record.artifact, other.artifact)
            for other in self._records.values()
            if other.artifact.patch_id != patch_id and other.status == "applied"
        ]
        return {
            "patch_id": patch_id,
            "status": record.status,
            "row_counts": record.artifact.row_counts(),
            "norms": record.artifact.norm_summary(),
            "conflicts_with_applied": conflicts,
        }

    def run_canaries(self, patch_id: str, checks: dict[str, bool]) -> dict[str, Any]:
        record = self._record(patch_id)
        record.canaries.update({str(name): bool(value) for name, value in checks.items()})
        return {
            "patch_id": patch_id,
            "canaries": dict(record.canaries),
            "passed": self._canaries_passed(record),
        }

    def approve_patch(self, patch_id: str, approver: str) -> dict[str, Any]:
        record = self._record(patch_id)
        if not self._canaries_passed(record):
            raise ValueError(f"Patch canaries have not passed: {patch_id}")
        if approver not in record.approvals:
            record.approvals.append(approver)
        record.status = "approved"
        return self.inspect_patch(patch_id)

    def apply_patch(self, patch_id: str, model) -> dict[str, Any]:
        record = self._record(patch_id)
        if record.status != "approved":
            raise ValueError(f"Patch must be approved before apply: {patch_id}")
        record.backup = self._applier.apply(model, record.artifact)
        record.status = "applied"
        return self.inspect_patch(patch_id)

    def rollback_patch(self, patch_id: str, model) -> dict[str, Any]:
        record = self._record(patch_id)
        if record.status != "applied" or record.backup is None:
            raise ValueError(f"Patch is not applied: {patch_id}")
        self._applier.rollback(model, record.backup)
        record.backup = None
        record.status = "rolled_back"
        return self.inspect_patch(patch_id)

    def inspect_patch(self, patch_id: str) -> dict[str, Any]:
        record = self._record(patch_id)
        return {
            "patch_id": patch_id,
            "status": record.status,
            "method_profile_id": record.artifact.method_profile_id,
            "subject": record.artifact.subject,
            "relation_id": record.artifact.relation_id,
            "row_counts": record.artifact.row_counts(),
            "norms": record.artifact.norm_summary(),
            "canaries": dict(record.canaries),
            "approvals": list(record.approvals),
        }

    def diff_patch(self, left_id: str, right_id: str) -> dict[str, Any]:
        return conflict_summary(
            self._record(left_id).artifact,
            self._record(right_id).artifact,
        )

    def list_patches(self) -> list[dict[str, Any]]:
        return [self.inspect_patch(patch_id) for patch_id in sorted(self._records)]

    def _record(self, patch_id: str) -> PatchRecord:
        if patch_id not in self._records:
            raise KeyError(f"Unknown patch: {patch_id}")
        return self._records[patch_id]

    @staticmethod
    def _canaries_passed(record: PatchRecord) -> bool:
        return bool(record.canaries) and all(record.canaries.values())
