"""Failure triage helpers for EasyEdit-compatible result rows."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .easyedit_utils import jsonable


def collect_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for row in rows:
        modes = failure_modes(row)
        if modes:
            failures.append(_failure_record(row, modes))
    return failures


def failure_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = collect_failures(rows)
    counts: dict[str, int] = {}
    relation_counts: dict[str, int] = {}
    for failure in failures:
        for mode in failure["failure_modes"]:
            counts[mode] = counts.get(mode, 0) + 1
        relation = failure.get("relation_id")
        if relation is not None:
            relation_counts[str(relation)] = relation_counts.get(str(relation), 0) + 1
    return {
        "n_failed_cases": len(failures),
        "failure_mode_counts": dict(sorted(counts.items())),
        "failed_by_relation_id": dict(sorted(relation_counts.items())),
    }


def write_failures_only(args, rows: list[dict[str, Any]], summary: dict[str, Any]) -> Path:
    output = _failures_output_path(args)
    payload = {
        "mode": "failures_only",
        "source_output": str(args.output),
        "summary": failure_summary(rows),
        "run_summary": summary,
        "failures": collect_failures(rows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False))
    return output


def failure_modes(row: dict[str, Any]) -> list[str]:
    modes = []
    checks = {
        "tf_rewrite": _metric(row.get("post", {}), "rewrite_acc"),
        "tf_rephrase": _metric(row.get("post", {}), "rephrase_acc"),
        "tf_ps_all": _metric(row.get("post", {}), "rephrase_all_acc"),
        "tf_locality": _locality(row.get("post", {})),
        "gen_rewrite": _metric(row.get("generation", {}), "rewrite_acc"),
        "gen_rephrase": _metric(row.get("generation", {}), "rephrase_acc"),
        "gen_ps_all": _metric(row.get("generation", {}), "rephrase_all_acc"),
        "ctx_gen_rewrite": _metric(row.get("contextual_generation", {}), "rewrite_acc"),
        "prob_locality": _prob_locality(row.get("probability", {})),
    }
    for name, value in checks.items():
        if value is not None and value < 1.0:
            modes.append(name)
    return modes


def _failure_record(row: dict[str, Any], modes: list[str]) -> dict[str, Any]:
    rewrite = row.get("requested_rewrite", {})
    target_new = rewrite.get("target_new", {})
    target_true = rewrite.get("target_true", {})
    return {
        "case_id": row.get("case_id"),
        "relation_id": row.get("relation_id"),
        "subject": rewrite.get("subject"),
        "prompt": rewrite.get("prompt"),
        "target_new": target_new.get("str"),
        "target_true": target_true.get("str"),
        "failure_modes": modes,
        "metrics": {
            "post": _metric_group(row.get("post", {})),
            "generation": _metric_group(row.get("generation", {})),
            "contextual_generation": _metric_group(row.get("contextual_generation", {})),
            "probability": _prob_group(row.get("probability", {})),
        },
    }


def _metric_group(group: dict[str, Any]) -> dict[str, float | None]:
    return {
        "rewrite_acc": _metric(group, "rewrite_acc"),
        "rephrase_acc": _metric(group, "rephrase_acc"),
        "rephrase_all_acc": _metric(group, "rephrase_all_acc"),
        "locality_acc": _locality(group),
    }


def _prob_group(group: dict[str, Any]) -> dict[str, float | None]:
    return {
        "rewrite_acc": _metric(group, "rewrite_acc"),
        "rephrase_acc": _metric(group, "rephrase_acc"),
        "rephrase_all_acc": _metric(group, "rephrase_all_acc"),
        "locality_acc": _prob_locality(group),
    }


def _metric(group: dict[str, Any], key: str) -> float | None:
    if key not in group:
        return None
    value = group[key]
    if isinstance(value, list):
        return float(np.mean(value)) if value else None
    return float(value)


def _locality(group: dict[str, Any]) -> float | None:
    locality = group.get("locality", {})
    if "neighborhood_acc" not in locality:
        return None
    return float(np.mean(locality["neighborhood_acc"]))


def _prob_locality(group: dict[str, Any]) -> float | None:
    values = []
    for value in group.get("locality", {}).values():
        values.append(float(np.mean(value)))
    return float(np.mean(values)) if values else None


def _failures_output_path(args) -> Path:
    if args.failures_output:
        return Path(args.failures_output)
    output = Path(args.output)
    return output.with_name(f"{output.stem}.failures{output.suffix or '.json'}")
