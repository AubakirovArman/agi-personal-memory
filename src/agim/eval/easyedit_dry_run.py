"""Dry-run summaries for EasyEdit-compatible CounterFact samples."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .easyedit_utils import jsonable


def dry_run_payload(
    *,
    args,
    dataset_sha256: str,
    all_facts: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    records: list[dict[str, Any]],
    locality_limit: int | None,
) -> dict[str, Any]:
    relation_ids = [
        fact.get("requested_rewrite", {}).get("relation_id")
        for fact in facts
    ]
    return {
        "mode": "dry_run_summary",
        "command": " ".join(sys.argv),
        "model": args.model,
        "device": args.device,
        "dataset": {
            "source": args.dataset,
            "sha256": dataset_sha256,
            "total_size": len(all_facts),
            "sample_policy": args.sample_policy,
            "seed": args.seed,
            "n_requested": args.n,
            "n_selected": len(facts),
            "case_ids": [fact.get("case_id") for fact in facts],
            "relation_counts": dict(sorted(Counter(relation_ids).items())),
            "locality_prompts": "all" if locality_limit is None else locality_limit,
        },
        "record_stats": _record_stats(records),
        "sample": [_sample_row(fact, record) for fact, record in zip(facts[:10], records[:10])],
        "would_load_model": False,
        "would_load_easyedit": False,
    }


def write_dry_run_summary(args, payload: dict[str, Any]) -> Path:
    output = _dry_run_output_path(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False))
    return output


def _dry_run_output_path(args) -> Path:
    if args.dry_run_output:
        return Path(args.dry_run_output)
    output = Path(args.output)
    return output.with_name(f"{output.stem}.dry_run{output.suffix or '.json'}")


def _record_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    rephrase_counts = [len(record.get("rephrase_prompts", [])) for record in records]
    locality_counts = [
        len(record.get("locality", {}).get("neighborhood", {}).get("prompt", []))
        for record in records
    ]
    return {
        "rephrase_prompt_counts": _count_summary(rephrase_counts),
        "locality_prompt_counts": _count_summary(locality_counts),
    }


def _count_summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": int(min(values)),
        "max": int(max(values)),
        "mean": round(float(sum(values)) / len(values), 4),
    }


def _sample_row(fact: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    rw = fact.get("requested_rewrite", {})
    locality = record.get("locality", {}).get("neighborhood", {})
    return {
        "case_id": fact.get("case_id"),
        "relation_id": rw.get("relation_id"),
        "subject": rw.get("subject"),
        "prompt": record.get("prompt"),
        "target_new": rw.get("target_new", {}).get("str"),
        "target_true": rw.get("target_true", {}).get("str"),
        "n_rephrase_prompts": len(record.get("rephrase_prompts", [])),
        "n_locality_prompts": len(locality.get("prompt", [])),
    }
