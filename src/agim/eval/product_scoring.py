"""Scoring helpers for product benchmark adapter outputs."""
from __future__ import annotations

from typing import Any

import numpy as np


def scored_product_payload(adapter_payload: dict[str, Any],
                           output_payload: dict[str, Any],
                           source: str = "") -> dict[str, Any]:
    cases = adapter_payload.get("cases", [])
    outputs = {str(item.get("case_id")): item for item in output_payload.get("cases", [])}
    rows = [score_product_case(case, outputs.get(str(case.get("case_id")), {}))
            for case in cases]
    return {
        "artifact_schema_version": "product_scored_outputs.v1",
        "source": source,
        "adapter_schema_version": adapter_payload.get("artifact_schema_version"),
        "output_schema_version": output_payload.get("artifact_schema_version"),
        "n": len(rows),
        "summary": summarize_scored_product_rows(rows),
        "rows": rows,
        "caveat": (
            "Scored product adapter outputs; this is a benchmark result only "
            "if the output payload came from a documented model-editing run."
        ),
    }


def score_product_case(case: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    direct = _contains(_output_text(output.get("direct_output", "")),
                       case.get("request", {}).get("target_new", ""))
    locality = _score_groups(case.get("locality", {}), output.get("locality_outputs", {}))
    portability = _score_groups(
        case.get("portability", {}), output.get("portability_outputs", {}))
    return {
        "case_id": case.get("case_id"),
        "direct_success": direct,
        "locality_acc": locality,
        "portability_acc": portability,
        "product_composite_acc": _mean([direct, locality, portability]),
    }


def summarize_scored_product_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "n": len(rows),
        "rewrite_acc": _mean([row["direct_success"] for row in rows]),
        "locality_acc": _mean([row["locality_acc"] for row in rows]),
        "portability_acc": _mean([row["portability_acc"] for row in rows]),
        "product_composite_acc": _mean([row["product_composite_acc"] for row in rows]),
    }


def _score_groups(groups: dict[str, Any], outputs: dict[str, Any]) -> float:
    scores = []
    for name, group in groups.items():
        truths = group.get("ground_truth", [])
        group_outputs = outputs.get(name, [])
        scores.extend(
            _contains(_output_at(group_outputs, idx), truth)
            for idx, truth in enumerate(truths)
        )
    return _mean(scores)


def _contains(generated: str, expected: str) -> bool:
    return bool(expected) and expected.lower() in generated.lower()


def _output_at(outputs: list[Any], idx: int) -> str:
    if idx >= len(outputs):
        return ""
    return _output_text(outputs[idx])


def _output_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("text") or value.get("output") or "")
    return str(value)


def _mean(values) -> float:
    if values is None:
        return 0.0
    if not isinstance(values, list):
        return float(values)
    clean = [float(value) for value in values if value is not None]
    return round(float(np.mean(clean)), 6) if clean else 0.0
