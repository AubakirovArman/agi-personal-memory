"""Norm-budget no-commit helpers for EasyEdit runs."""
from __future__ import annotations

from typing import Any

from agim.model.patch_artifact import NormBudgetPolicy, PatchArtifact, RowPatch

from .easyedit_run_metadata import method_profile_id


def evaluate_edit_budget(editor, args, fact: dict[str, Any],
                         backup: dict[str, Any]) -> dict[str, Any] | None:
    policy = norm_budget_policy(args)
    if policy is None:
        return None
    artifact = patch_artifact_from_backup(editor, args, fact, backup)
    return artifact.budget_decision(policy)


def norm_budget_policy(args) -> NormBudgetPolicy | None:
    policy = NormBudgetPolicy(
        max_patch_delta_norm=_positive_or_none(
            getattr(args, "max_patch_delta_norm", 0.0)),
        max_row_delta_norm=_positive_or_none(
            getattr(args, "max_row_delta_norm", 0.0)),
        max_mean_delta_norm=_positive_or_none(
            getattr(args, "max_mean_delta_norm", 0.0)),
        max_rows=_positive_int_or_none(getattr(args, "max_edited_rows", 0)),
    )
    if not any(value is not None for value in policy.to_dict().values()):
        return None
    return policy


def patch_artifact_from_backup(editor, args, fact: dict[str, Any],
                               backup: dict[str, Any]) -> PatchArtifact:
    rewrite = fact.get("requested_rewrite", {})
    rows = []
    for row_id, before in backup.get("lm_backup", {}).items():
        after = editor.model.lm_head.weight.data[row_id, :]
        rows.append(RowPatch.from_tensors("lm_head", row_id, before, after))
    for row_id, before in backup.get("emb_backup", {}).items():
        after = editor.model.model.embed_tokens.weight.data[row_id, :]
        rows.append(RowPatch.from_tensors("embed_tokens", row_id, before, after))
    return PatchArtifact(
        patch_id=f"case-{fact.get('case_id', 'unknown')}",
        base_model_digest="runtime",
        method_profile_id=method_profile_id(args),
        subject=str(rewrite.get("subject", "")),
        relation_id=str(rewrite.get("relation_id", "")),
        target_new=str(rewrite.get("target_new", {}).get("str", "")),
        target_true=str(rewrite.get("target_true", {}).get("str", "")),
        rows=rows,
    )


def _positive_or_none(value: float) -> float | None:
    return float(value) if value and value > 0 else None


def _positive_int_or_none(value: int) -> int | None:
    return int(value) if value and value > 0 else None
