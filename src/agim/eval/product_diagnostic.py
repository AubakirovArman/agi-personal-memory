"""KnowEdit-inspired product diagnostics for EasyEdit artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .easyedit_utils import jsonable


SCHEMA_VERSION = "product_diagnostic.v1"
DATASET_SCHEMA_VERSION = "product_dataset_adapter.v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="EasyEdit artifact JSON")
    source.add_argument("--dataset-input", help="KnowEdit/UniEdit-style dataset JSON")
    parser.add_argument("--benchmark-name", default="knowedit")
    parser.add_argument("--output", help="Output JSON path")
    return parser


def diagnostic_payload(artifact: dict[str, Any], source: str = "") -> dict[str, Any]:
    rows = artifact.get("metrics", [])
    return {
        "artifact_schema_version": SCHEMA_VERSION,
        "source": source,
        "source_schema_version": artifact.get("artifact_schema_version"),
        "source_method_profile_id": artifact.get("method_profile_id"),
        "n": len(rows),
        "summary": summarize_product_rows(rows),
        "by_relation_id": summarize_product_by_relation(rows),
        "caveat": (
            "KnowEdit-inspired product diagnostic over EasyEdit artifact fields; "
            "not an external KnowEdit/UniEdit/ScEdit leaderboard result."
        ),
    }


def product_dataset_payload(
    records: list[dict[str, Any]],
    source: str = "",
    benchmark_name: str = "knowedit",
) -> dict[str, Any]:
    cases = [
        normalize_product_record(record, case_id, benchmark_name)
        for case_id, record in enumerate(records)
    ]
    return {
        "artifact_schema_version": DATASET_SCHEMA_VERSION,
        "benchmark_name": benchmark_name,
        "source": source,
        "n": len(cases),
        "cases": cases,
        "caveat": (
            "Product benchmark adapter payload only; this is not a scored "
            "external leaderboard result."
        ),
    }


def normalize_product_record(
    record: dict[str, Any],
    case_id: int,
    benchmark_name: str = "knowedit",
) -> dict[str, Any]:
    subject = str(record.get("subject") or record.get("concept") or "")
    target_new = _target_text(record.get("target_new") or record.get("labels") or "")
    target_true = _target_text(record.get("ground_truth") or record.get("target_true") or "")
    prompt = str(record.get("prompt") or record.get("text") or "")
    return {
        "case_id": record.get("case_id", case_id),
        "benchmark_name": benchmark_name,
        "request": {
            "subject": subject,
            "prompt": _format_prompt(prompt, subject),
            "relation_id": str(record.get("relation_id") or record.get("relation") or ""),
            "target_new": target_new,
            "target_true": target_true,
        },
        "locality": _normalize_groups(record.get("locality", {})),
        "portability": _normalize_groups(record.get("portability", {})),
        "source_record_id": record.get("id") or record.get("case_id"),
    }


def summarize_product_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    signals = {
        "rewrite_acc": [_metric(row.get("post", {}), "rewrite_acc") for row in rows],
        "paraphrase_all_acc": [
            _metric(row.get("post", {}), "rephrase_all_acc") for row in rows
        ],
        "locality_acc": [_locality(row) for row in rows],
        "portability_acc": [_portability(row) for row in rows],
    }
    summary: dict[str, float | int] = {"n": len(rows)}
    available = []
    for key, values in signals.items():
        clean = [value for value in values if value is not None]
        score = _mean(clean)
        summary[key] = score
        if clean:
            available.append(score)
    summary["product_composite_acc"] = _mean(available)
    return summary


def summarize_product_by_relation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        relation = row.get("relation_id")
        if relation is not None:
            grouped.setdefault(str(relation), []).append(row)
    return {
        relation: summarize_product_rows(rel_rows)
        for relation, rel_rows in sorted(grouped.items())
    }


def _locality(row: dict[str, Any]) -> float | None:
    locality = row.get("post", {}).get("locality", {})
    if "neighborhood_acc" not in locality:
        return None
    return _mean(locality["neighborhood_acc"])


def _portability(row: dict[str, Any]) -> float | None:
    portability = row.get("post", {}).get("portability", {})
    if not portability:
        return None
    values = [_mean(value) for value in portability.values()]
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


def _normalize_groups(groups: Any) -> dict[str, dict[str, list[str]]]:
    if not isinstance(groups, dict):
        return {}
    normalized: dict[str, dict[str, list[str]]] = {}
    for name, value in groups.items():
        items = value if isinstance(value, list) else [value]
        prompts: list[str] = []
        answers: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            prompt = item.get("prompt") or item.get("question") or item.get("New Question")
            answer = (
                item.get("ground_truth")
                or item.get("answer")
                or item.get("New Answer")
                or item.get("target")
            )
            if prompt:
                prompts.extend(_as_text_list(prompt))
                answer_values = _as_text_list(answer)
                if len(answer_values) == 1 and len(_as_text_list(prompt)) > 1:
                    answer_values = answer_values * len(_as_text_list(prompt))
                answers.extend(answer_values)
        if prompts:
            normalized[str(name)] = {
                "prompt": prompts,
                "ground_truth": answers[:len(prompts)],
            }
    return normalized


def _format_prompt(prompt: str, subject: str) -> str:
    return prompt.format(subject) if "{}" in prompt else prompt


def _target_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("str", ""))
    if isinstance(value, list):
        if not value:
            return ""
        if isinstance(value[0], list):
            return str(value[0][0]) if value[0] else ""
        return str(value[0])
    return str(value)


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return [str(item[0]) if item else "" for item in value]
        return [str(item) for item in value]
    return [str(value)]


def main() -> int:
    args = build_parser().parse_args()
    if args.dataset_input:
        source = Path(args.dataset_input)
        records = json.loads(source.read_text())
        payload = product_dataset_payload(records, str(source), args.benchmark_name)
        default_suffix = ".product_dataset"
    else:
        source = Path(args.input)
        artifact = json.loads(source.read_text())
        payload = diagnostic_payload(artifact, str(source))
        default_suffix = ".product"
    output = Path(args.output) if args.output else source.with_name(
        f"{source.stem}{default_suffix}{source.suffix or '.json'}"
    )
    output.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False))
    print(f"Product diagnostic saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
