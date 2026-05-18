"""MQuAKE-style post-hoc diagnostics for portability artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .easyedit_utils import jsonable


SCHEMA_VERSION = "mquake_style_diagnostic.v1"
DATASET_SCHEMA_VERSION = "mquake_dataset_adapter.v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="EasyEdit artifact JSON")
    source.add_argument("--dataset-input", help="MQuAKE dataset JSON")
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
        "summary": summarize_mquake_rows(rows),
        "by_relation_id": summarize_mquake_by_relation(rows),
        "caveat": (
            "Post-hoc MQuAKE-style diagnostic over EasyEdit portability fields; "
            "not an official MQuAKE dataset result."
        ),
    }


def mquake_dataset_payload(records: list[dict[str, Any]],
                           source: str = "") -> dict[str, Any]:
    """Normalize MQuAKE-style raw records into auditable benchmark cases."""
    cases = [
        normalize_mquake_record(record, case_id)
        for case_id, record in enumerate(records)
    ]
    return {
        "artifact_schema_version": DATASET_SCHEMA_VERSION,
        "source": source,
        "n": len(cases),
        "cases": cases,
        "caveat": (
            "MQuAKE dataset adapter payload only; running and scoring these "
            "cases still requires a model editor evaluation pass."
        ),
    }


def normalize_mquake_record(record: dict[str, Any], case_id: int) -> dict[str, Any]:
    rewrites = record.get("requested_rewrite") or []
    if isinstance(rewrites, dict):
        rewrites = [rewrites]
    requests = [_normalize_rewrite(item) for item in rewrites]
    prompts = _as_list(record.get("questions") or record.get("portability_prompt"))
    answers = _answers_for(
        prompts,
        record.get("new_answer") or record.get("portability_ground_truth"),
    )
    return {
        "case_id": record.get("case_id", case_id),
        "requests": requests,
        "portability": {
            "multi_hop": {
                "prompt": prompts,
                "ground_truth": answers,
            }
        } if prompts else {},
        "source_record_id": record.get("id") or record.get("case_id"),
    }


def summarize_mquake_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    direct = [_metric(row.get("post", {}), "rewrite_acc") for row in rows]
    hop = [_portability(row) for row in rows]
    direct = [value for value in direct if value is not None]
    hop = [value for value in hop if value is not None]
    direct_acc = _mean(direct)
    hop_acc = _mean(hop)
    return {
        "n": len(rows),
        "direct_rewrite_acc": direct_acc,
        "multi_hop_acc": hop_acc,
        "composite_acc": round((direct_acc + hop_acc) / 2, 6),
        "direct_success_hop_fail_rate": _direct_success_hop_fail_rate(rows),
    }


def summarize_mquake_by_relation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        relation = row.get("relation_id")
        if relation is not None:
            grouped.setdefault(str(relation), []).append(row)
    return {
        relation: summarize_mquake_rows(rel_rows)
        for relation, rel_rows in sorted(grouped.items())
    }


def _direct_success_hop_fail_rate(rows: list[dict[str, Any]]) -> float:
    values = []
    for row in rows:
        direct = _metric(row.get("post", {}), "rewrite_acc")
        hop = _portability(row)
        if direct is None or hop is None:
            continue
        values.append(float(direct >= 1.0 and hop < 1.0))
    return _mean(values)


def _portability(row: dict[str, Any]) -> float | None:
    portability = row.get("post", {}).get("portability", {})
    values = []
    for value in portability.values():
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


def _target_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("str", ""))
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _answers_for(prompts: list[str], value: Any) -> list[str]:
    answers = _as_list(value)
    if len(answers) == 1 and len(prompts) > 1:
        answers = answers * len(prompts)
    return answers[:len(prompts)]


def main() -> int:
    args = build_parser().parse_args()
    if args.dataset_input:
        source = Path(args.dataset_input)
        records = json.loads(source.read_text())
        payload = mquake_dataset_payload(records, str(source))
        default_suffix = ".mquake_dataset"
    else:
        source = Path(args.input)
        artifact = json.loads(source.read_text())
        payload = diagnostic_payload(artifact, str(source))
        default_suffix = ".mquake_style"
    output = Path(args.output) if args.output else source.with_name(
        f"{source.stem}{default_suffix}{source.suffix or '.json'}"
    )
    output.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False))
    print(f"MQuAKE-style diagnostic saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
