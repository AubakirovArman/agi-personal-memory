"""Norm-budget no-commit helpers for EasyEdit runs."""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
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
        max_shared_row_delta_norm=_positive_or_none(
            getattr(args, "max_shared_row_delta_norm", 0.0)),
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
    relation_id = str(rewrite.get("relation_id", ""))
    relation_slot_buckets = int(getattr(args, "relation_slot_buckets", 0) or 0)
    relation_slot_id = _relation_slot_id(
        relation_id,
        relation_slot_buckets,
    )
    target_new_text = str(rewrite.get("target_new", {}).get("str", ""))
    subject_text = str(rewrite.get("subject", ""))
    target_true_text = str(rewrite.get("target_true", {}).get("str", ""))
    tokenizer = getattr(editor, "tokenizer", None)
    relation_shard = (relation_id.strip() or "global")
    metadata = dict(backup.get("metadata", {}))
    metadata.setdefault("relation_shard", relation_shard)
    metadata["relation_slot_id"] = relation_slot_id
    metadata["relation_slot_buckets"] = relation_slot_buckets
    metadata["subject_token_ids"] = _unique_ids(_token_ids(tokenizer, subject_text))
    metadata["target_token_ids"] = _unique_ids(_token_ids(tokenizer, target_new_text))
    control_row_ids = _control_rows(tokenizer)
    if control_row_ids:
        metadata["control_row_ids"] = control_row_ids
    protected_basis_ids = _unique_values(
        backup.get("relation_protected_ids", []),
    )
    if protected_basis_ids:
        metadata["protected_basis_ids"] = protected_basis_ids
    rows = []
    for row_id, before in backup.get("lm_backup", {}).items():
        after = editor.model.lm_head.weight.data[row_id, :]
        rows.append(RowPatch.from_tensors("lm_head", row_id, before, after))
    for row_id, before in backup.get("emb_backup", {}).items():
        after = editor.model.model.embed_tokens.weight.data[row_id, :]
        rows.append(RowPatch.from_tensors("embed_tokens", row_id, before, after))
    for (layer_idx, row_id), before in backup.get("ffn_backup", {}).items():
        layer = f"model.layers.{int(layer_idx)}.mlp.down_proj"
        after = editor.ffn_weight(int(layer_idx))[row_id, :]
        rows.append(RowPatch.from_tensors(layer, row_id, before, after))
    return PatchArtifact(
        patch_id=f"case-{fact.get('case_id', 'unknown')}",
        base_model_digest="runtime",
        method_profile_id=method_profile_id(args),
        subject=subject_text,
        relation_id=relation_id,
        target_new=target_new_text,
        target_true=target_true_text,
        rows=rows,
        metadata=metadata,
    )


def _relation_slot_id(relation_id: str, buckets: int) -> str:
    clean_relation = (relation_id or "").strip()
    if buckets <= 0:
        return clean_relation
    if not clean_relation:
        return "global"
    digest = hashlib.sha256(clean_relation.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % buckets
    return f"{clean_relation}:slot-{bucket:04d}"


def _token_ids(tokenizer, text: str) -> list[int]:
    if tokenizer is None:
        return []
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return []


def _control_rows(tokenizer) -> list[int]:
    if tokenizer is None:
        return []
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        return []
    return [int(eos)]


def _unique_ids(values: Iterable[Any]) -> list[int]:
    unique: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number in seen:
            continue
        seen.add(number)
        unique.append(number)
    return unique


def _unique_values(values: Iterable[Any]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _positive_or_none(value: float) -> float | None:
    return float(value) if value and value > 0 else None


def _positive_int_or_none(value: int) -> int | None:
    return int(value) if value and value > 0 else None
