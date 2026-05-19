"""Side-slot EasyEdit evaluation loop."""
from __future__ import annotations

import copy
from typing import Any, Callable

from agim.model.side_slot_memory import SideSlotMemory

from .easyedit_budget import evaluate_edit_budget, patch_artifact_from_backup
from .easyedit_summary import summarize_official
from .easyedit_utils import jsonable, parse_retention_steps


def run_sequential_side_slot(
    *,
    args,
    model,
    tok,
    hparams,
    editor,
    facts: list[dict[str, Any]],
    records: list[dict[str, Any]],
    compute_edit_quality,
    test_prediction_acc,
    device_id: int,
    apply_one: Callable,
    compute_pre: Callable,
    post_bundle: Callable,
    base_row: Callable,
    budget_status: Callable,
    args_by_idx: list[Any] | None = None,
    relation_profiles: list[dict[str, Any] | None] | None = None,
) -> tuple[list[dict[str, Any]], list[float], dict[str, Any]]:
    if args_by_idx is None:
        args_by_idx = [copy.deepcopy(args) for _ in records]
    if relation_profiles is None:
        relation_profiles = [None for _ in records]
    pres = [
        compute_pre(fact_args, model, tok, hparams, rec, compute_edit_quality, device_id)
        for fact_args, rec in zip(args_by_idx, records)
    ]
    side_memory = SideSlotMemory(
        relation_slot_buckets=int(getattr(args, "relation_slot_buckets", 0) or 0),
    )
    edit_times: list[float] = []
    edit_statuses: list[dict[str, Any]] = []
    retention: dict[str, Any] = {}
    retention_steps = parse_retention_steps(args.retention_steps, len(records))
    for edit_idx, (fact, record, fact_args, relation_profile) in enumerate(
        zip(facts, records, args_by_idx, relation_profiles),
        start=1,
    ):
        backup, edit_time = apply_one(editor, fact_args, fact, record)
        budget = evaluate_edit_budget(editor, fact_args, fact, backup)
        relation_slot_id = side_memory.relation_slot_for(
            str(fact.get("requested_rewrite", {}).get("relation_id", "")),
        )
        status = _side_slot_status(
            budget_status(budget),
            fact,
            relation_slot_id,
        )
        status["relation_profile"] = relation_profile
        if not (budget and budget["no_commit"]):
            artifact = patch_artifact_from_backup(editor, fact_args, fact, backup)
            side_memory.add_patch(artifact, relation_slot_id=relation_slot_id)
            status["side_slot_id"] = artifact.patch_id
            status["relation_slot_id"] = relation_slot_id
        editor.rollback(backup)
        edit_times.append(edit_time)
        edit_statuses.append(status)
        if edit_idx in retention_steps:
            retention[f"after_{edit_idx}"] = _checkpoint(
                args_by_idx, relation_profiles, model, tok, hparams, facts, records,
                pres, edit_times,
                edit_statuses, side_memory, compute_edit_quality,
                test_prediction_acc, device_id, edit_idx, post_bundle, base_row,
                test_fluency=getattr(args, "test_fluency", False),
            )
    metrics = _evaluate(
        args_by_idx, relation_profiles, model, tok, hparams, facts, records, pres, edit_times,
        edit_statuses, side_memory, compute_edit_quality, test_prediction_acc,
        device_id, post_bundle, base_row,
        test_fluency=getattr(args, "test_fluency", False),
    )
    retention["side_slot_summary"] = side_memory.summary()
    retention["relation_slot_allocator"] = side_memory.allocator_summary()
    retention["relation_slot_summary"] = side_memory.relation_slot_summary()
    return metrics, edit_times, retention


def _checkpoint(args_by_idx, relation_profiles, model, tok, hparams, facts, records, pres,
                edit_times,
                edit_statuses, side_memory, compute_edit_quality,
                test_prediction_acc, device_id, edit_idx, post_bundle, base_row,
                test_fluency: bool = False):
    rows = _evaluate(
        args_by_idx[:edit_idx],
        relation_profiles[:edit_idx],
        model, tok, hparams, facts[:edit_idx], records[:edit_idx],
        pres[:edit_idx], edit_times, edit_statuses, side_memory,
        compute_edit_quality, test_prediction_acc, device_id, post_bundle, base_row,
        test_fluency=test_fluency)
    return {
        "n_edits_applied": edit_idx,
        "n_evaluated": len(rows),
        "summary": summarize_official(rows),
        "case_ids": [fact.get("case_id") for fact in facts[:edit_idx]],
        "side_slot_summary": side_memory.summary(),
        "relation_slot_allocator": side_memory.allocator_summary(),
        "relation_slot_summary": side_memory.relation_slot_summary(),
    }


def _evaluate(args_by_idx, relation_profiles, model, tok, hparams, facts, records, pres, edit_times,
              edit_statuses, side_memory, compute_edit_quality,
              test_prediction_acc, device_id, post_bundle, base_row,
              test_fluency: bool = False):
    rows = []
    for idx, (fact, record, pre, fact_args, relation_profile) in enumerate(
        zip(facts, records, pres, args_by_idx, relation_profiles)
    ):
        row = base_row(fact, record, idx, edit_times[idx], edit_statuses[idx])
        rewrite = fact.get("requested_rewrite", {})
        with side_memory.overlay_for(
            model,
            subject=str(rewrite.get("subject", "")),
            relation_slot_id=side_memory.relation_slot_for(
                str(rewrite.get("relation_id", "")),
            ),
        ):
            row.update(post_bundle(
                fact_args, model, tok, hparams, compute_edit_quality,
                test_prediction_acc, record, copy.deepcopy(pre), device_id,
                fluency=test_fluency))
        row["relation_profile"] = relation_profile
        rows.append(jsonable(row))
    return rows


def _side_slot_status(status: dict[str, Any], fact: dict[str, Any],
                      relation_slot_id: str) -> dict[str, Any]:
    status = dict(status)
    status.setdefault("edit_status", "side_slot")
    status["edit_backend"] = "side_slot"
    status["relation_slot_id"] = relation_slot_id or str(
        fact.get("requested_rewrite", {}).get("relation_id", ""))
    return status
