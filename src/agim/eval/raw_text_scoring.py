"""Scoring helpers for raw-text edit proposal outputs."""
from __future__ import annotations

from typing import Any

import numpy as np


def scored_raw_text_payload(
    adapter_payload: dict[str, Any],
    output_payload: dict[str, Any],
    source: str = "",
) -> dict[str, Any]:
    proposals = adapter_payload.get("proposals", [])
    outputs = _outputs_by_case(output_payload)
    rows = [
        score_raw_text_case(proposal, outputs.get(str(idx), {}), idx)
        for idx, proposal in enumerate(proposals)
    ]
    return {
        "artifact_schema_version": "raw_text_scored_outputs.v1",
        "source": source,
        "adapter_schema_version": adapter_payload.get("artifact_schema_version"),
        "output_schema_version": output_payload.get("artifact_schema_version"),
        "n": len(rows),
        "summary": summarize_scored_raw_text_rows(rows),
        "rows": rows,
        "caveat": (
            "Scored raw-text proposal outputs; this is a benchmark result only "
            "if the output payload came from a documented model-editing run."
        ),
    }


def score_raw_text_case(
    proposal: dict[str, Any],
    output: dict[str, Any],
    case_id: int | None = None,
) -> dict[str, Any]:
    target = str(proposal.get("target_new", ""))
    direct = _contains(_output_text(output.get("direct_output", "")), target)
    prompt_score = _optional_contains(output.get("prompt_output"), target)
    service_ready = _status_score(output.get("patch_status"))
    canary_pass_rate = _canary_pass_rate(output.get("canary_results"))
    rollback_verified = _optional_bool(output.get("rollback_verified"))
    composite = _mean([
        direct,
        prompt_score,
        service_ready,
        canary_pass_rate,
        rollback_verified,
    ])
    return {
        "case_id": output.get("case_id", case_id),
        "subject": proposal.get("subject"),
        "relation_id": proposal.get("relation_id"),
        "parser": proposal.get("parser"),
        "parse_confidence": float(proposal.get("confidence", 0.0)),
        "direct_success": direct,
        "prompt_success": prompt_score,
        "service_ready": service_ready,
        "canary_pass_rate": canary_pass_rate,
        "rollback_verified": rollback_verified,
        "raw_text_composite_acc": composite,
    }


def summarize_scored_raw_text_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "n": len(rows),
        "rewrite_acc": _mean([row["direct_success"] for row in rows]),
        "prompt_acc": _mean([row["prompt_success"] for row in rows]),
        "service_ready_rate": _mean([row["service_ready"] for row in rows]),
        "canary_pass_rate": _mean([row["canary_pass_rate"] for row in rows]),
        "rollback_verified_rate": _mean([row["rollback_verified"] for row in rows]),
        "raw_text_composite_acc": _mean([
            row["raw_text_composite_acc"] for row in rows
        ]),
    }


def _outputs_by_case(output_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for idx, item in enumerate(output_payload.get("cases", [])):
        outputs[str(item.get("case_id", idx))] = item
    return outputs


def _status_score(status: Any) -> float | None:
    if status is None:
        return None
    return float(str(status).lower() in {"materialized", "approved", "applied"})


def _canary_pass_rate(canaries: Any) -> float | None:
    if canaries is None:
        return None
    if isinstance(canaries, dict):
        values = canaries.values()
    elif isinstance(canaries, list):
        values = canaries
    else:
        return float(bool(canaries))
    scores = [_optional_bool(value) for value in values]
    return _mean(scores)


def _optional_contains(value: Any, expected: str) -> float | None:
    if value is None:
        return None
    return float(_contains(_output_text(value), expected))


def _contains(generated: str, expected: str) -> bool:
    return bool(expected) and expected.lower() in generated.lower()


def _output_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("text") or value.get("output") or "")
    if isinstance(value, list):
        return " ".join(_output_text(item) for item in value)
    return str(value)


def _optional_bool(value: Any) -> float | None:
    if value is None:
        return None
    return float(bool(value))


def _mean(values) -> float:
    if values is None:
        return 0.0
    if not isinstance(values, list):
        return float(values)
    clean = [float(value) for value in values if value is not None]
    return round(float(np.mean(clean)), 6) if clean else 0.0
