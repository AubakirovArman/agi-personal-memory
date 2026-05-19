"""In-process PatchService API for Path B patch lifecycle."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .patch_artifact import NormBudgetPolicy, PatchArtifact, RowPatch, conflict_summary
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

    def __init__(
        self,
        *,
        enforce_conflicts: bool = False,
        enforce_budget: bool = False,
        budget_policy: NormBudgetPolicy | None = None,
    ):
        self._records: dict[str, PatchRecord] = {}
        self._applier = WALMemitBatchEditor()
        self._enforce_conflicts = enforce_conflicts
        self._enforce_budget = enforce_budget
        self._budget_policy = budget_policy
        self._shared_row_deltas: dict[tuple[str, int], float] = {}

    def set_policy(self, policy: NormBudgetPolicy | None) -> None:
        self._budget_policy = policy

    def propose_patch(self, artifact: PatchArtifact) -> dict[str, Any]:
        if artifact.patch_id in self._records:
            raise ValueError(f"Patch already exists: {artifact.patch_id}")
        if self._enforce_conflicts and self._has_conflict_with_applied(artifact):
            raise ValueError(f"Patch conflicts with applied patches: {artifact.patch_id}")
        if self._enforce_budget:
            budget = self._effective_budget_decision(
                PatchRecord(artifact=artifact),
            )
            if budget["no_commit"]:
                raise ValueError(
                    f"Patch violates budget policy: {artifact.patch_id} "
                    f"reasons={budget['reasons']}"
                )
        self._records[artifact.patch_id] = PatchRecord(artifact=artifact)
        return self.inspect_patch(artifact.patch_id)

    def simulate_patch(self, patch_id: str) -> dict[str, Any]:
        record = self._record(patch_id)
        conflicts = [
            summary
            for summary in [
                conflict_summary(record.artifact, other.artifact)
                for other in self._records.values()
                if other.artifact.patch_id != patch_id and other.status == "applied"
            ]
            if summary.get("has_conflict")
        ]
        return {
            "patch_id": patch_id,
            "status": record.status,
            "row_counts": record.artifact.row_counts(),
            "norms": record.artifact.norm_summary(),
            "conflicts_with_applied": conflicts,
        }

    def materialize_patch(
        self,
        patch_id: str,
        editor,
        edit_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Turn a draft proposal into row-level patch deltas.

        The editor is applied only long enough to capture changed rows. The
        live model is rolled back before this method returns, so approval and
        canary gates still happen before deployment.
        """
        record = self._record(patch_id)
        if record.status not in {"proposed", "materialized"}:
            raise ValueError(f"Patch cannot be materialized from status: {record.status}")
        rewrite = record.artifact.metadata.get("requested_rewrite")
        if not isinstance(rewrite, dict):
            raise ValueError(f"Patch has no requested_rewrite metadata: {patch_id}")
        backup = editor.apply_edit(**_editor_apply_kwargs(rewrite, edit_kwargs or {}))
        rows = _rows_from_editor_backup(editor, backup)
        if hasattr(editor, "rollback"):
            editor.rollback(backup)
        metadata = dict(record.artifact.metadata)
        metadata.update({
            "requires_backend_materialization": False,
            "materialized_row_counts": _row_counts(rows),
            "materialization_metadata": backup.get("metadata", {}),
        })
        record.artifact = PatchArtifact(
            patch_id=record.artifact.patch_id,
            base_model_digest=record.artifact.base_model_digest,
            method_profile_id=record.artifact.method_profile_id,
            subject=record.artifact.subject,
            relation_id=record.artifact.relation_id,
            target_new=record.artifact.target_new,
            target_true=record.artifact.target_true,
            rows=rows,
            metadata=metadata,
        )
        record.status = "materialized"
        return self.inspect_patch(patch_id)

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
        if self._enforce_budget:
            budget = self._effective_budget_decision(record)
            if budget["no_commit"]:
                raise ValueError(f"Patch violates budget policy: {patch_id}")
        if approver not in record.approvals:
            record.approvals.append(approver)
        record.status = "approved"
        return self.inspect_patch(patch_id)

    def apply_patch(self, patch_id: str, model) -> dict[str, Any]:
        record = self._record(patch_id)
        if record.status != "approved":
            raise ValueError(f"Patch must be approved before apply: {patch_id}")
        if self._enforce_conflicts and self._has_conflict_with_applied(record.artifact):
            raise ValueError(f"Patch has active conflicts with applied patches: {patch_id}")
        if self._enforce_budget:
            budget = self._effective_budget_decision(record)
            if budget["no_commit"]:
                raise ValueError(f"Patch violates budget policy: {patch_id}")
        record.backup = self._applier.apply(model, record.artifact)
        self._apply_shared_row_deltas(record.artifact)
        record.status = "applied"
        return self.inspect_patch(patch_id)

    def rollback_patch(self, patch_id: str, model) -> dict[str, Any]:
        record = self._record(patch_id)
        if record.status != "applied" or record.backup is None:
            raise ValueError(f"Patch is not applied: {patch_id}")
        self._applier.rollback(model, record.backup)
        record.backup = None
        self._release_shared_row_deltas(record.artifact)
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
            "conflicts_with_applied": self._has_conflict_with_applied(record.artifact),
            "budget_decision": self._effective_budget_decision(record),
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

    def _effective_budget_decision(self, record: PatchRecord) -> dict[str, Any]:
        decision = record.artifact.metadata.get("budget_decision")
        if isinstance(decision, dict):
            decision = dict(decision)
        else:
            decision = None
        if self._budget_policy is None:
            return {
                "allow_commit": True,
                "no_commit": False,
                "reasons": [],
                "policy": None,
                "norms": record.artifact.norm_summary(),
                "row_count": len(record.artifact.rows),
            }
        if decision is None:
            decision = record.artifact.budget_decision(self._budget_policy)
        return self._add_shared_row_budget_reasons(decision, record.artifact)

    def _has_conflict_with_applied(self, artifact: PatchArtifact) -> list[dict[str, Any]]:
        conflicts: list[dict[str, Any]] = []
        for record in self._records.values():
            if record.status != "applied" or record.artifact.patch_id == artifact.patch_id:
                continue
            summary = conflict_summary(artifact, record.artifact)
            if summary.get("has_conflict"):
                conflicts.append(summary)
        return conflicts

    def _add_shared_row_budget_reasons(
        self,
        base_decision: dict[str, Any],
        artifact: PatchArtifact,
    ) -> dict[str, Any]:
        limit = self._budget_policy.max_shared_row_delta_norm if self._budget_policy else None
        reasons = list(base_decision.get("reasons", []))
        if limit is None:
            base_decision["reasons"] = reasons
            return base_decision
        projected_usage = {}
        for key, value in self._row_delta_profile(artifact).items():
            current = self._shared_row_deltas.get(key, 0.0)
            proposed = current + value
            projected_usage[f"{key[0]}:{key[1]}"] = {
                "existing": round(current, 6),
                "patch": round(value, 6),
                "projected": round(proposed, 6),
            }
            if proposed > limit:
                reasons.append({
                    "metric": "shared_row_delta_norm",
                    "value": round(proposed, 6),
                    "limit": limit,
                    "row": f"{key[0]}:{key[1]}",
                    "existing": round(current, 6),
                    "patch": round(value, 6),
                })
        return {
            "allow_commit": not reasons,
            "no_commit": bool(reasons),
            "reasons": reasons,
            "policy": base_decision.get("policy", self._budget_policy.to_dict() if self._budget_policy else None),
            "norms": base_decision.get("norms", artifact.norm_summary()),
            "row_count": base_decision.get("row_count", len(artifact.rows)),
            "shared_row_usage": {
                "limit": limit,
                "projected": projected_usage,
            },
        }

    @staticmethod
    def _canaries_passed(record: PatchRecord) -> bool:
        return bool(record.canaries) and all(record.canaries.values())

    def _row_delta_profile(self, artifact: PatchArtifact) -> dict[tuple[str, int], float]:
        deltas: dict[tuple[str, int], float] = {}
        for row in artifact.rows:
            key = (row.layer, int(row.row_id))
            deltas[key] = deltas.get(key, 0.0) + float(row.delta_norm)
        return deltas

    def _apply_shared_row_deltas(self, artifact: PatchArtifact) -> None:
        for key, value in self._row_delta_profile(artifact).items():
            self._shared_row_deltas[key] = self._shared_row_deltas.get(key, 0.0) + float(value)

    def _release_shared_row_deltas(self, artifact: PatchArtifact) -> None:
        for key, value in self._row_delta_profile(artifact).items():
            self._shared_row_deltas[key] = self._shared_row_deltas.get(key, 0.0) - float(value)
            if self._shared_row_deltas[key] <= 0:
                del self._shared_row_deltas[key]


def _editor_apply_kwargs(
    rewrite: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    subject = str(rewrite.get("subject", ""))
    target_new = rewrite.get("target_new", {})
    target_true = rewrite.get("target_true", {})
    prompt = str(rewrite.get("prompt", ""))
    kwargs = {
        "subject": subject,
        "target": _target_text(target_new),
        "relation": str(rewrite.get("relation_id", "")),
        "prompt": prompt.format(subject) if "{}" in prompt else prompt,
        "old_target": _target_text(target_true),
    }
    kwargs.update(overrides)
    return kwargs


def _rows_from_editor_backup(editor, backup: dict[str, Any]) -> list[RowPatch]:
    rows: list[RowPatch] = []
    model = editor.model
    for row_id, before in backup.get("lm_backup", {}).items():
        after = model.lm_head.weight.data[int(row_id), :]
        rows.append(RowPatch.from_tensors("lm_head", int(row_id), before, after))
    for row_id, before in backup.get("emb_backup", {}).items():
        after = model.model.embed_tokens.weight.data[int(row_id), :]
        rows.append(RowPatch.from_tensors("embed_tokens", int(row_id), before, after))
    for (layer_idx, row_id), before in backup.get("ffn_backup", {}).items():
        layer = f"model.layers.{int(layer_idx)}.mlp.down_proj"
        after = editor.ffn_weight(int(layer_idx))[int(row_id), :]
        rows.append(RowPatch.from_tensors(layer, int(row_id), before, after))
    return rows


def _target_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("str", ""))
    return str(value)


def _row_counts(rows: list[RowPatch]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.layer] = counts.get(row.layer, 0) + 1
    return dict(sorted(counts.items()))
