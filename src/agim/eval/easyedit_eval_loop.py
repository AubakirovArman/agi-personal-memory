"""Evaluation loops for EasyEdit-compatible single and sequential runs."""
from __future__ import annotations

import copy
import time
from typing import Any

from .easyedit_budget import evaluate_edit_budget
from .easyedit_bundle import post_edit_bundle
from .easyedit_cli import print_progress
from .easyedit_metrics import edit_nt_metrics
from .easyedit_summary import summarize_official
from .easyedit_utils import jsonable, parse_retention_steps


def apply_one(editor, args, fact: dict[str, Any],
              record: dict[str, Any]) -> tuple[dict, float]:
    rw = fact["requested_rewrite"]
    neg_prompts = None
    if args.use_neg_prompts:
        neg_prompts = record.get("locality", {}).get("neighborhood", {}).get("prompt")
    positive_prompts = None
    if args.use_positive_prompts:
        positive_prompts = record.get("rephrase_prompts", [])
    start = time.time()
    backup = editor.apply_edit(
        rw["subject"],
        rw["target_new"]["str"],
        rw["relation_id"],
        prompt=record["prompt"],
        clamp_lm=args.clamp_lm,
        clamp_embed=args.clamp_embed,
        clamp_eos=args.clamp_eos,
        clamp_anti=args.clamp_anti,
        old_target=rw["target_true"]["str"],
        clamp_old=args.clamp_old,
        target_token_mode=args.target_token_mode,
        positive_prompts=positive_prompts,
        max_positive_prompts=args.positive_prompt_limit,
        positive_key_weight=args.positive_key_weight,
        positive_constraint_mode=getattr(args, "positive_constraint_mode", "none"),
        positive_constraint_k_pos=getattr(args, "positive_constraint_k_pos", 4),
        positive_constraint_k_neg=getattr(args, "positive_constraint_k_neg", 4),
        neg_prompts=neg_prompts,
        max_neg_prompts=args.neg_prompt_limit,
        clamp_anti_scope=getattr(args, "clamp_anti_scope", "both"),
        neg_projection_strength=args.neg_projection_strength,
        history_projection_strength=args.history_projection_strength,
        embed_history_projection_strength=args.embed_history_projection_strength,
        projection_mode=args.projection_mode,
        history_slot_mode=args.history_slot_mode,
        max_history_keys=args.max_history_keys,
        state_namespace=args.state_namespace,
        relation_protected_mode=args.relation_protected_mode,
        max_relation_protected_keys=args.max_relation_protected_keys,
        wal_encode_updates=args.wal_encode_updates,
    )
    return backup, time.time() - start


def run_evaluation_loop(
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
) -> tuple[list[dict[str, Any]], list[float], dict[str, Any]]:
    if args.sequential_edit and args.edit_backend == "side_slot":
        from .easyedit_side_slot_loop import run_sequential_side_slot
        return run_sequential_side_slot(
            args=args, model=model, tok=tok, hparams=hparams, editor=editor,
            facts=facts, records=records, compute_edit_quality=compute_edit_quality,
            test_prediction_acc=test_prediction_acc, device_id=device_id,
            apply_one=apply_one, compute_pre=_compute_pre,
            post_bundle=_post_bundle, base_row=_base_row,
            budget_status=_budget_status,
        )
    if args.sequential_edit:
        return run_sequential(
            args=args, model=model, tok=tok, hparams=hparams, editor=editor,
            facts=facts, records=records, compute_edit_quality=compute_edit_quality,
            test_prediction_acc=test_prediction_acc, device_id=device_id,
        )
    return run_single(
        args=args, model=model, tok=tok, hparams=hparams, editor=editor,
        facts=facts, records=records, compute_edit_quality=compute_edit_quality,
        test_prediction_acc=test_prediction_acc, device_id=device_id,
    )


def run_sequential(
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
) -> tuple[list[dict[str, Any]], list[float], dict[str, Any]]:
    pres = [_compute_pre(args, model, tok, hparams, rec, compute_edit_quality, device_id)
            for rec in records]
    backups: list[dict[str, Any]] = []
    edit_times: list[float] = []
    edit_statuses: list[dict[str, Any]] = []
    retention: dict[str, Any] = {}
    retention_steps = parse_retention_steps(args.retention_steps, len(records))
    for edit_idx, (fact, record) in enumerate(zip(facts, records), start=1):
        backup, edit_time = apply_one(editor, args, fact, record)
        budget = evaluate_edit_budget(editor, args, fact, backup)
        if budget and budget["no_commit"]:
            editor.rollback(backup)
            backup = {"lm_backup": {}, "emb_backup": {}, "budget_decision": budget}
        elif budget:
            backup["budget_decision"] = budget
        backups.append(backup)
        edit_statuses.append(_budget_status(budget))
        edit_times.append(edit_time)
        if edit_idx in retention_steps:
            retention[f"after_{edit_idx}"] = _retention_checkpoint(
                args, model, tok, hparams, facts, records, pres, edit_times,
                edit_statuses, compute_edit_quality, test_prediction_acc,
                device_id, edit_idx,
            )
    metrics = _evaluate_accumulated(
        args, model, tok, hparams, facts, records, pres, edit_times, edit_statuses,
        compute_edit_quality, test_prediction_acc, device_id,
    )
    for backup in reversed(backups):
        editor.rollback(backup)
    return metrics, edit_times, retention


def run_single(
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
) -> tuple[list[dict[str, Any]], list[float], dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    edit_times: list[float] = []
    for idx, (fact, record) in enumerate(zip(facts, records)):
        pre = _compute_pre(args, model, tok, hparams, record, compute_edit_quality, device_id)
        backup, edit_time = apply_one(editor, args, fact, record)
        budget = evaluate_edit_budget(editor, args, fact, backup)
        proposed_nt = None
        if budget and budget["no_commit"]:
            proposed_nt = edit_nt_metrics(editor, backup, tok.eos_token_id)
            editor.rollback(backup)
        row = _base_row(fact, record, idx, edit_time, _budget_status(budget))
        row.update(_post_bundle(args, model, tok, hparams, compute_edit_quality,
                                test_prediction_acc, record, pre, device_id,
                                fluency=args.test_fluency))
        row["NT"] = proposed_nt or edit_nt_metrics(editor, backup, tok.eos_token_id)
        if not (budget and budget["no_commit"]):
            editor.rollback(backup)
        edit_times.append(edit_time)
        metrics.append(jsonable(row))
        if (idx + 1) % 10 == 0 or idx + 1 == len(records):
            print_progress(summarize_official(metrics), idx + 1, len(records))
    return metrics, edit_times, {}


def _retention_checkpoint(
    args, model, tok, hparams, facts, records, pres, edit_times, edit_statuses,
    compute_edit_quality, test_prediction_acc, device_id, edit_idx: int,
) -> dict[str, Any]:
    retention_rows = []
    for idx, (fact, record, pre) in enumerate(zip(facts[:edit_idx],
                                                  records[:edit_idx],
                                                  pres[:edit_idx])):
        row = _base_row(fact, record, idx, edit_times[idx], edit_statuses[idx])
        row.update(_post_bundle(args, model, tok, hparams, compute_edit_quality,
                                test_prediction_acc, record, copy.deepcopy(pre),
                                device_id, fluency=False))
        retention_rows.append(jsonable(row))
    return {
        "n_edits_applied": edit_idx,
        "n_evaluated": len(retention_rows),
        "summary": summarize_official(retention_rows),
        "case_ids": [fact.get("case_id") for fact in facts[:edit_idx]],
    }


def _evaluate_accumulated(
    args, model, tok, hparams, facts, records, pres, edit_times, edit_statuses,
    compute_edit_quality, test_prediction_acc, device_id,
) -> list[dict[str, Any]]:
    metrics = []
    for idx, (fact, record, pre) in enumerate(zip(facts, records, pres)):
        row = _base_row(fact, record, idx, edit_times[idx], edit_statuses[idx])
        row.update(_post_bundle(args, model, tok, hparams, compute_edit_quality,
                                test_prediction_acc, record, copy.deepcopy(pre),
                                device_id, fluency=args.test_fluency))
        metrics.append(jsonable(row))
    return metrics


def _post_bundle(
    args, model, tok, hparams, compute_edit_quality, test_prediction_acc,
    record: dict[str, Any], pre: dict[str, Any], device_id: int, fluency: bool,
) -> dict[str, Any]:
    return post_edit_bundle(
        model=model,
        tok=tok,
        hparams=hparams,
        compute_edit_quality=compute_edit_quality,
        test_prediction_acc=test_prediction_acc,
        record=record,
        pre=pre,
        model_name=args.model,
        device_id=device_id,
        device=args.device,
        probability=args.probability_metrics,
        fluency=fluency,
    )


def _compute_pre(args, model, tok, hparams, record, compute_edit_quality, device_id):
    return compute_edit_quality(
        model, args.model, hparams, tok, record, device_id,
        eval_metric="token_em"
    )


def _base_row(fact: dict[str, Any], record: dict[str, Any],
              idx: int, edit_time: float,
              edit_status: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "case_id": fact.get("case_id", idx),
        "relation_id": fact.get("requested_rewrite", {}).get("relation_id"),
        "requested_rewrite": record,
        "edit_time_s": round(edit_time, 4),
    }
    if edit_status:
        row.update(edit_status)
    return row


def _budget_status(decision: dict[str, Any] | None) -> dict[str, Any]:
    if not decision:
        return {}
    return {
        "edit_status": "no_commit" if decision["no_commit"] else "committed",
        "budget_decision": jsonable(decision),
    }
