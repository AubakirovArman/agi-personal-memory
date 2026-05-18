"""Payload serialization for EasyEdit-compatible runs."""
from __future__ import annotations

import sys
from typing import Any

import torch
import transformers

from agim.eval.easyedit_counterfact import git_sha

from .easyedit_failures import failure_summary
from .easyedit_run_metadata import (
    ARTIFACT_SCHEMA_VERSION,
    atoms_digest,
    base_model_digest,
    method_profile_id,
    parse_failure_families,
)
from .easyedit_utils import parse_retention_steps


def build_payload(
    *,
    args,
    metrics: list[dict[str, Any]],
    retention: dict[str, Any],
    summary: dict[str, Any],
    elapsed: float,
    dataset_sha256: str,
    all_facts: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    locality_limit: int | None,
    model=None,
    editor=None,
) -> dict[str, Any]:
    failure_families = parse_failure_families(args.failure_families)
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "n": len(metrics),
        "model": args.model,
        "method_profile_id": method_profile_id(args),
        "device": args.device,
        "git_sha": git_sha(),
        "command": " ".join(sys.argv),
        "base_model_digest": base_model_digest(model, args),
        "atoms_digest": atoms_digest(editor),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "easyedit": _easyedit_metadata(args),
        "dataset": _dataset_metadata(args, dataset_sha256, all_facts, facts, locality_limit),
        "hyperparams": _hyperparams(args, len(metrics)),
        "summary": summary,
        "failure_analysis": failure_summary(metrics, failure_families),
        "retention": retention,
        "time_s": round(elapsed, 2),
        "time_per_edit_s": round(elapsed / max(len(metrics), 1), 4),
        "metrics": metrics,
    }


def _easyedit_metadata(args) -> dict[str, Any]:
    return {
        "root": str(args.easyedit_root),
        "functions": [
            "easyeditor.evaluate.evaluate.compute_edit_quality",
            "easyeditor.evaluate.evaluate_utils.test_prediction_acc",
        ],
        "aggregation": "EasyEdit BaseEditor-style pre/post locality comparison",
        "teacher_forcing_metric": "token_em",
        "generation_metric": "vanilla_generation token equality",
        "contextual_generation_metric": (
            "greedy token equality against prompt + space + target suffix ids"
        ),
    }


def _dataset_metadata(args, dataset_sha256, all_facts, facts, locality_limit) -> dict[str, Any]:
    return {
        "source": args.dataset,
        "sha256": dataset_sha256,
        "total_size": len(all_facts),
        "sample_policy": args.sample_policy,
        "seed": args.seed,
        "case_ids": [fact.get("case_id") for fact in facts],
        "relation_ids": [
            fact.get("requested_rewrite", {}).get("relation_id")
            for fact in facts
        ],
        "locality_prompts": "all" if locality_limit is None else locality_limit,
        "rephrase_prompt": "first",
        "rephrase_prompts": "all",
    }


def _hyperparams(args, n_records: int) -> dict[str, Any]:
    return {
        "clamp_lm": args.clamp_lm,
        "clamp_embed": args.clamp_embed,
        "clamp_eos": args.clamp_eos,
        "clamp_anti": args.clamp_anti,
        "clamp_old": args.clamp_old,
        "target_token_mode": args.target_token_mode,
        "use_positive_prompts": args.use_positive_prompts,
        "positive_prompt_limit": args.positive_prompt_limit,
        "positive_key_weight": args.positive_key_weight,
        "positive_constraint_mode": args.positive_constraint_mode,
        "failure_families": parse_failure_families(args.failure_families),
        "use_neg_prompts": args.use_neg_prompts,
        "neg_prompt_limit": args.neg_prompt_limit,
        "neg_projection_strength": args.neg_projection_strength,
        "history_projection_strength": args.history_projection_strength,
        "embed_history_projection_strength": args.embed_history_projection_strength,
        "projection_mode": args.projection_mode,
        "history_slot_mode": args.history_slot_mode,
        "max_history_keys": args.max_history_keys,
        "wal_encode_updates": args.wal_encode_updates,
        "nt_sample_mode": "deterministic_lcg",
        "nt_sample_size": args.nt_sample_size,
        "probability_metrics": args.probability_metrics,
        "test_fluency": args.test_fluency,
        "sequential_edit": args.sequential_edit,
        "retention_steps": parse_retention_steps(args.retention_steps, n_records),
    }
