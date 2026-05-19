"""Evaluation loops for EasyEdit-compatible single and sequential runs."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time
from typing import Any

from .easyedit_budget import evaluate_edit_budget
from .easyedit_bundle import post_edit_bundle
from .easyedit_cli import print_progress
from .easyedit_metrics import edit_nt_metrics
from .easyedit_summary import summarize_official
from .easyedit_utils import jsonable, parse_retention_steps
from .easyedit_presets import ANTI_PROFILES, POSITIVE_PROFILES


_DEFAULT_CANDIDATE_PROFILES: dict[str, dict[str, str | float | int | bool]] = {
    "safe": {},
    "conservative": {},
    "positive_w025": {"positive_profile": "w025"},
    "positive_w035": {"positive_profile": "w035"},
    "positive_w025_ridge": {"positive_profile": "w025_ridge"},
    "anti_target_low": {"anti_profile": "target_low"},
    "anti_subject_low": {"anti_profile": "subject_low"},
    "anti_both_low": {"anti_profile": "both_low"},
}


def _parse_relation_profile_map(profile_path: str | None) -> dict[str, dict[str, Any]]:
    if not profile_path:
        return {}
    path = Path(profile_path)
    if not path.exists():
        raise ValueError(f"relation-profile-map path not found: {profile_path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("relation-profile-map must be a JSON object")
    profiles: dict[str, dict[str, Any]] = {}
    for relation_id, raw_profile in raw.items():
        if raw_profile is None:
            continue
        if not isinstance(raw_profile, dict):
            raise ValueError(
                f"relation profile for relation_id={relation_id!r} must be an object"
            )
        profiles[str(relation_id)] = {
            str(key): value for key, value in raw_profile.items()
        }
    return profiles


def _apply_relation_profile(
    args,
    relation_profile_map: dict[str, dict[str, Any]],
    relation_id: str | None,
) -> tuple[Any, dict[str, Any] | None]:
    if not relation_profile_map or not relation_id:
        return args, None
    profile = relation_profile_map.get(str(relation_id))
    if not profile:
        return args, None

    anti_profile = profile.get("anti_profile", getattr(args, "anti_profile", "off"))
    if anti_profile not in ANTI_PROFILES:
        raise ValueError(f"Unknown anti profile in relation map: {anti_profile}")
    positive_profile = profile.get(
        "positive_profile", getattr(args, "positive_profile", "off")
    )
    if positive_profile not in POSITIVE_PROFILES:
        raise ValueError(f"Unknown positive profile in relation map: {positive_profile}")

    args.anti_profile = anti_profile
    args.positive_profile = positive_profile
    for key, value in ANTI_PROFILES[anti_profile].items():
        setattr(args, key, value)
    for key, value in POSITIVE_PROFILES[positive_profile].items():
        setattr(args, key, value)
    for key, value in profile.items():
        if key in {"anti_profile", "positive_profile"}:
            continue
        setattr(args, key, value)

    return args, profile


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
    relation_profile_map = _parse_relation_profile_map(
        getattr(args, "relation_profile_map", "")
    )
    relation_profiles: list[dict[str, Any] | None] = []
    relation_args_by_idx: list[Any] = []
    pres = [_compute_pre(args, model, tok, hparams, rec, compute_edit_quality, device_id)
            for rec in records]
    backups: list[dict[str, Any]] = []
    edit_times: list[float] = []
    edit_statuses: list[dict[str, Any]] = []
    retention: dict[str, Any] = {}
    retention_steps = parse_retention_steps(args.retention_steps, len(records))
    for edit_idx, (fact, record) in enumerate(zip(facts, records), start=1):
        fact_args, relation_profile = _apply_relation_profile(
            copy.deepcopy(args),
            relation_profile_map,
            fact.get("requested_rewrite", {}).get("relation_id"),
        )
        relation_profiles.append(relation_profile)
        relation_args_by_idx.append(fact_args)
        backup, edit_time = apply_one(editor, fact_args, fact, record)
        budget = evaluate_edit_budget(editor, fact_args, fact, backup)
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
                relation_args_by_idx, model, tok, hparams, facts, records, pres,
                relation_profiles,
                edit_times,
                edit_statuses, compute_edit_quality, test_prediction_acc,
                device_id, edit_idx,
            )
    metrics = _evaluate_accumulated(
        relation_args_by_idx, model, tok, hparams, facts, records, pres,
        edit_times, edit_statuses, relation_profiles,
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
    candidate_grid = _parse_candidate_grid(getattr(args, "candidate_grid", ""))
    relation_profile_map = _parse_relation_profile_map(
        getattr(args, "relation_profile_map", "")
    )
    for idx, (fact, record) in enumerate(zip(facts, records)):
        fact_args, relation_profile = _apply_relation_profile(
            copy.deepcopy(args),
            relation_profile_map,
            fact.get("requested_rewrite", {}).get("relation_id"),
        )
        pre = _compute_pre(fact_args, model, tok, hparams, record, compute_edit_quality, device_id)
        if candidate_grid:
            best = _select_candidate(
                candidates=candidate_grid,
                idx=idx,
                args=fact_args,
                fact=fact,
                record=record,
                model=model,
                tok=tok,
                hparams=hparams,
                compute_edit_quality=compute_edit_quality,
                test_prediction_acc=test_prediction_acc,
                device_id=device_id,
                editor=editor,
                pre=pre,
            )
            backup = best["backup"]
            edit_time = best["edit_time"]
            budget = best["budget"]
            edit_status = best["edit_status"]
            proposed_nt = best.get("nt")
            row = _base_row(fact, record, idx, edit_time, edit_status)
            row["candidate_grid"] = candidate_grid
            row["candidate_grid_selected"] = best["name"]
            row["candidate_grid_evaluated"] = len(best["candidates"])
            row["candidate_evaluations"] = best["candidates"]
            row["candidate_selected"] = best.get("selected_profile")
            row["relation_profile"] = relation_profile
            post_bundle_args = best["args"]
        else:
            backup, edit_time = apply_one(editor, fact_args, fact, record)
            budget = evaluate_edit_budget(editor, fact_args, fact, backup)
            proposed_nt = None
            if budget and budget["no_commit"]:
                proposed_nt = edit_nt_metrics(editor, backup, tok.eos_token_id)
                editor.rollback(backup)
            row = _base_row(fact, record, idx, edit_time, _budget_status(budget))
            row["relation_profile"] = relation_profile
            post_bundle_args = fact_args
            row["candidate_selected"] = None
            row["candidate_grid"] = None
            row["candidate_grid_selected"] = None
            row["candidate_grid_evaluated"] = 0
            row["candidate_evaluations"] = []
        row.update(_post_bundle(post_bundle_args, model, tok, hparams, compute_edit_quality,
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


def _parse_candidate_grid(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _candidate_options() -> set[str]:
    return set(_DEFAULT_CANDIDATE_PROFILES)


def _parse_metric_value(metric: Any) -> float:
    if isinstance(metric, (list, tuple)):
        return float(metric[0]) if metric else 0.0
    if metric is None:
        return 0.0
    try:
        return float(metric)
    except (TypeError, ValueError):
        return 0.0


def _extract_candidate_post_metrics(post: dict[str, Any]) -> dict[str, float]:
    locality = post.get("locality", {}).get("neighborhood_acc")
    rewrite = post.get("rewrite_acc")
    ps_all = post.get("rephrase_all_acc")
    return {
        "rewrite": _parse_metric_value(rewrite),
        "ps_all": _parse_metric_value(ps_all),
        "locality": _parse_metric_value(locality),
    }


def _candidate_reject_reasons(post_metrics: dict[str, float],
                             args) -> list[str]:
    reasons: list[str] = []
    locality_min = getattr(args, "candidate_locality_min", None)
    rewrite_min = getattr(args, "candidate_rewrite_min", None)
    if locality_min is not None and post_metrics["locality"] < locality_min:
        reasons.append("low_locality")
    if rewrite_min is not None and post_metrics["rewrite"] < rewrite_min:
        reasons.append("low_rewrite")
    return reasons


def _candidate_score(post_metrics: dict[str, float], metric_name: str) -> float:
    if metric_name == "rewrite":
        return post_metrics["rewrite"]
    if metric_name == "rewrite_then_psall":
        return post_metrics["rewrite"] * 0.7 + post_metrics["ps_all"] * 0.3
    return post_metrics["ps_all"]


def _select_candidate(
    candidates: list[str],
    idx: int,
    args,
    fact,
    record,
    model,
    tok,
    hparams,
    compute_edit_quality,
    test_prediction_acc,
    device_id: int,
    editor,
    pre: dict[str, Any],
) -> dict[str, Any]:
    unknown = [name for name in candidates if name not in _candidate_options()]
    if unknown:
        raise ValueError(f"Unknown candidate profile(s): {', '.join(sorted(unknown))}")

    best_score = float("-inf")
    best_name: str | None = None
    best_args = copy.deepcopy(args)
    best_backup = {"lm_backup": {}, "emb_backup": {}}
    best_edit_time = 0.0
    best_budget = None
    best_proposed_nt = None

    candidate_rows: list[dict[str, Any]] = []
    for name in candidates:
        variant_args = copy.deepcopy(args)
        variant_args.candidate_grid = ""
        variant_args.candidate_rank = idx
        for key, value in _DEFAULT_CANDIDATE_PROFILES[name].items():
            setattr(variant_args, key, value)
        backup, edit_time = apply_one(editor, variant_args, fact, record)
        budget = evaluate_edit_budget(editor, variant_args, fact, backup)
        if budget and budget["no_commit"]:
            proposed_nt = edit_nt_metrics(editor, backup, tok.eos_token_id)
            editor.rollback(backup)
        else:
            proposed_nt = None
        post = _post_bundle(
            variant_args, model, tok, hparams, compute_edit_quality,
            test_prediction_acc, record, pre, device_id, fluency=False)
        edit_status = _budget_status(budget)
        post_metrics = _extract_candidate_post_metrics(post["post"])
        reasons = _candidate_reject_reasons(post_metrics, args)
        candidate_row = {
            "name": name,
            "positive_profile": getattr(variant_args, "positive_profile", "off"),
            "anti_profile": getattr(variant_args, "anti_profile", "off"),
            "rewrite": post_metrics["rewrite"],
            "ps_all": post_metrics["ps_all"],
            "locality": post_metrics["locality"],
            "edit_time_s": round(edit_time, 4),
            "reasons": reasons,
            "edit_status": edit_status["edit_status"] if edit_status else "unknown",
        }
        if budget:
            candidate_row["budget_reasons"] = budget["reasons"]
        candidate_row["rejected"] = bool(reasons)
        candidate_row["score"] = _candidate_score(
            post_metrics, getattr(args, "candidate_rerank_metric", "psall_guarded"),
        )
        if not candidate_row["rejected"]:
            if candidate_row["score"] > best_score:
                best_score = candidate_row["score"]
                best_name = name
                best_args = variant_args
                best_backup = backup
                best_edit_time = edit_time
                best_budget = budget
                best_proposed_nt = proposed_nt
        candidate_rows.append(candidate_row)
        editor.rollback(backup)

    if best_name is None:
        fallback_name = candidates[0]
        fallback = candidate_rows[0]
        fallback["rejected"] = True
        fallback_args = copy.deepcopy(args)
        fallback_args.candidate_grid = ""
        for key, value in _DEFAULT_CANDIDATE_PROFILES[fallback_name].items():
            setattr(fallback_args, key, value)
        fallback_backup, fallback_edit_time = apply_one(editor, fallback_args, fact, record)
        fallback_budget = evaluate_edit_budget(editor, fallback_args, fact, fallback_backup)
        if fallback_budget and fallback_budget["no_commit"]:
            fallback_nt = edit_nt_metrics(editor, fallback_backup, tok.eos_token_id)
            editor.rollback(fallback_backup)
        else:
            fallback_nt = None
        return {
            "name": fallback_name,
            "backup": fallback_backup,
            "edit_time": fallback_edit_time,
            "budget": fallback_budget,
            "candidates": candidate_rows,
            "edit_status": {
                "edit_status": "no_commit",
                "candidate_rejected_reason": "all_candidates_rejected_by_thresholds",
            },
            "nt": fallback_nt,
            "fallback": True,
            "selected_profile": fallback_name,
            "args": fallback_args,
        }

    return {
        "name": best_name,
        "backup": best_backup,
        "edit_time": best_edit_time,
        "budget": best_budget,
        "candidates": candidate_rows,
        "edit_status": _budget_status(best_budget),
        "nt": best_proposed_nt,
        "fallback": False,
        "selected_profile": best_name,
        "args": best_args,
    }


def _retention_checkpoint(
    args_by_idx: list[Any],
    model, tok, hparams, facts, records, pres, relation_profiles,
    edit_times, edit_statuses,
    compute_edit_quality, test_prediction_acc, device_id, edit_idx: int,
) -> dict[str, Any]:
    retention_rows = []
    for idx, (fact, record, pre, relation_profile) in enumerate(
        zip(
            facts[:edit_idx],
            records[:edit_idx],
            pres[:edit_idx],
            relation_profiles[:edit_idx],
        )
    ):
        row = _base_row(fact, record, idx, edit_times[idx], edit_statuses[idx])
        row["relation_profile"] = relation_profile
        row.update(_post_bundle(
            args_by_idx[idx], model, tok, hparams, compute_edit_quality,
            test_prediction_acc, record, copy.deepcopy(pre),
            device_id, fluency=False
        ))
        retention_rows.append(jsonable(row))
    return {
        "n_edits_applied": edit_idx,
        "n_evaluated": len(retention_rows),
        "summary": summarize_official(retention_rows),
        "case_ids": [fact.get("case_id") for fact in facts[:edit_idx]],
    }


def _evaluate_accumulated(
    args_by_idx: list[Any],
    model, tok, hparams, facts, records, pres,
    edit_times, edit_statuses, relation_profiles,
    compute_edit_quality, test_prediction_acc, device_id,
) -> list[dict[str, Any]]:
    metrics = []
    for idx, (fact, record, pre, relation_profile) in enumerate(
        zip(facts, records, pres, relation_profiles),
    ):
        row = _base_row(fact, record, idx, edit_times[idx], edit_statuses[idx])
        row["relation_profile"] = relation_profile
        row.update(_post_bundle(args_by_idx[idx], model, tok, hparams, compute_edit_quality,
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
