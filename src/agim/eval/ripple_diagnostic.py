"""Ripple-style post-hoc diagnostics for EasyEdit artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .easyedit_utils import jsonable


SCHEMA_VERSION = "ripple_style_diagnostic.v1"
DATASET_SCHEMA_VERSION = "ripple_dataset_adapter.v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="EasyEdit artifact JSON")
    source.add_argument("--dataset-input", help="RippleEdits-style dataset JSON")
    source.add_argument("--score-adapter", help="Ripple adapter payload JSON")
    parser.add_argument("--score-output", help="Model output JSON for --score-adapter")
    parser.add_argument("--output", help="Output JSON path")
    return parser


def diagnostic_payload(artifact: dict[str, Any], source: str = "") -> dict[str, Any]:
    rows = artifact.get("metrics", [])
    summary = summarize_ripple_rows(rows)
    return {
        "artifact_schema_version": SCHEMA_VERSION,
        "source": source,
        "source_schema_version": artifact.get("artifact_schema_version"),
        "source_method_profile_id": artifact.get("method_profile_id"),
        "n": len(rows),
        "summary": summary,
        "by_relation_id": summarize_ripple_by_relation(rows),
        "caveat": (
            "Post-hoc Ripple-style diagnostic over EasyEdit locality/relation "
            "fields; not an official RippleEdits dataset result."
        ),
    }


def ripple_dataset_payload(records: list[dict[str, Any]],
                           source: str = "") -> dict[str, Any]:
    """Normalize RippleEdits-style records into related-fact cases."""
    cases = [normalize_ripple_record(r, i) for i, r in enumerate(records)]
    return {
        "artifact_schema_version": DATASET_SCHEMA_VERSION,
        "source": source,
        "n": len(cases),
        "cases": cases,
        "caveat": (
            "RippleEdits dataset adapter payload only; this is not a scored "
            "RippleEdits benchmark result."
        ),
    }


def scored_ripple_payload(
    adapter_payload: dict[str, Any],
    output_payload: dict[str, Any],
    source: str = "",
) -> dict[str, Any]:
    cases = adapter_payload.get("cases", [])
    outputs = {str(item.get("case_id")): item for item in output_payload.get("cases", [])}
    rows = [score_ripple_case(case, outputs.get(str(case.get("case_id")), {}))
            for case in cases]
    return {
        "artifact_schema_version": "ripple_scored_outputs.v1",
        "source": source,
        "adapter_schema_version": adapter_payload.get("artifact_schema_version"),
        "output_schema_version": output_payload.get("artifact_schema_version"),
        "n": len(rows),
        "summary": summarize_scored_ripple_rows(rows),
        "rows": rows,
        "caveat": (
            "Scored Ripple adapter outputs; this is a benchmark result only if "
            "the output payload came from a documented model-editing run."
        ),
    }


def score_ripple_case(case: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    direct = _contains(_output_text(output.get("direct_output", "")),
                       case.get("request", {}).get("target_new", ""))
    related_outputs = output.get("related_outputs", [])
    related_scores = [_contains(_output_at(related_outputs, idx), item.get("ground_truth", ""))
                      for idx, item in enumerate(case.get("related_facts", []))]
    locality_outputs = output.get("locality_outputs", [])
    locality_prompts = case.get("locality", {}).get("neighborhood", {}).get("prompt", [])
    locality_scores = [bool(_output_at(locality_outputs, idx)) for idx, _ in enumerate(locality_prompts)]
    return {
        "case_id": case.get("case_id"),
        "direct_success": direct,
        "related_scores": related_scores,
        "locality_scores": locality_scores,
        "related_acc": _mean(related_scores),
        "locality_response_rate": _mean(locality_scores),
        "ripple_break": bool(direct and related_scores and not all(related_scores)),
    }


def summarize_scored_ripple_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "n": len(rows),
        "direct_rewrite_acc": _mean([row["direct_success"] for row in rows]),
        "related_acc": _mean([row["related_acc"] for row in rows]),
        "locality_response_rate": _mean([row["locality_response_rate"] for row in rows]),
        "ripple_break_rate": _mean([row["ripple_break"] for row in rows]),
    }


def normalize_ripple_record(record: dict[str, Any], case_id: int) -> dict[str, Any]:
    rewrite = record.get("requested_rewrite") or record
    related = (
        record.get("related_facts")
        or record.get("ripple_prompts")
        or record.get("implication_prompts")
        or []
    )
    return {
        "case_id": record.get("case_id", case_id),
        "request": _normalize_rewrite(rewrite),
        "related_facts": [_normalize_related(item) for item in _as_list(related)],
        "locality": _normalize_locality(record),
        "source_record_id": record.get("id") or record.get("case_id"),
    }


def summarize_ripple_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    direct = [_metric(row.get("post", {}), "rewrite_acc") for row in rows]
    related = [_tf_locality(row) for row in rows]
    prob_related = [_prob_locality(row) for row in rows]
    direct = [value for value in direct if value is not None]
    related = [value for value in related if value is not None]
    prob_related = [value for value in prob_related if value is not None]
    return {
        "n": len(rows),
        "direct_rewrite_acc": _mean(direct),
        "related_preservation_acc": _mean(related),
        "prob_related_preservation_acc": _mean(prob_related),
        "ripple_break_rate": _ripple_break_rate(rows),
    }


def summarize_ripple_by_relation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        relation = row.get("relation_id")
        if relation is not None:
            grouped.setdefault(str(relation), []).append(row)
    return {
        relation: summarize_ripple_rows(rel_rows)
        for relation, rel_rows in sorted(grouped.items())
    }


def _ripple_break_rate(rows: list[dict[str, Any]]) -> float:
    values = []
    for row in rows:
        direct = _metric(row.get("post", {}), "rewrite_acc")
        related = _tf_locality(row)
        if direct is None or related is None:
            continue
        values.append(float(direct >= 1.0 and related < 1.0))
    return _mean(values)


def _tf_locality(row: dict[str, Any]) -> float | None:
    locality = row.get("post", {}).get("locality", {})
    if "neighborhood_acc" not in locality:
        return None
    return _mean(locality["neighborhood_acc"])


def _prob_locality(row: dict[str, Any]) -> float | None:
    values = []
    for value in row.get("probability", {}).get("locality", {}).values():
        values.append(_mean(value))
    return _mean([value for value in values if value is not None])


def _metric(group: dict[str, Any], key: str) -> float | None:
    if key not in group:
        return None
    return _mean(group[key])


def _mean(values) -> float:
    if values is None:
        return 0.0
    if not isinstance(values, list):
        return float(values)
    clean = [float(value) for value in values if value is not None]
    return round(float(np.mean(clean)), 6) if clean else 0.0


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


def _normalize_rewrite(item: dict[str, Any]) -> dict[str, Any]:
    subject = str(item.get("subject", ""))
    prompt = str(item.get("prompt", ""))
    return {
        "subject": subject,
        "prompt": prompt.format(subject) if "{}" in prompt else prompt,
        "relation_id": str(item.get("relation_id") or item.get("relation") or ""),
        "target_new": _target_text(item.get("target_new", "")),
        "target_true": _target_text(
            item.get("target_true") or item.get("ground_truth") or ""),
    }


def _normalize_related(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        return {"prompt": item, "ground_truth": "", "relation_id": ""}
    return {
        "prompt": str(item.get("prompt") or item.get("question") or ""),
        "ground_truth": _target_text(
            item.get("ground_truth") or item.get("answer") or ""),
        "relation_id": str(item.get("relation_id") or item.get("relation") or ""),
    }


def _normalize_locality(record: dict[str, Any]) -> dict[str, Any]:
    locality = record.get("locality")
    if isinstance(locality, dict):
        return locality
    prompts = _as_texts(record.get("neighborhood_prompts"))
    return {"neighborhood": {"prompt": prompts, "ground_truth": []}} if prompts else {}


def _target_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("str", ""))
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _as_texts(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value)]


def main() -> int:
    args = build_parser().parse_args()
    if args.dataset_input:
        source = Path(args.dataset_input)
        records = json.loads(source.read_text())
        payload = ripple_dataset_payload(records, str(source))
        default_suffix = ".ripple_dataset"
    elif args.score_adapter:
        if not args.score_output:
            raise SystemExit("--score-output is required with --score-adapter")
        source = Path(args.score_adapter)
        adapter = json.loads(source.read_text())
        outputs = json.loads(Path(args.score_output).read_text())
        payload = scored_ripple_payload(adapter, outputs, str(source))
        default_suffix = ".ripple_scored"
    else:
        source = Path(args.input)
        artifact = json.loads(source.read_text())
        payload = diagnostic_payload(artifact, str(source))
        default_suffix = ".ripple_style"
    output = Path(args.output) if args.output else source.with_name(
        f"{source.stem}{default_suffix}{source.suffix or '.json'}"
    )
    output.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False))
    print(f"Ripple-style diagnostic saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
