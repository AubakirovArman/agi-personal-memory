"""MQuAKE-style post-hoc diagnostics for portability artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .easyedit_utils import jsonable


SCHEMA_VERSION = "mquake_style_diagnostic.v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="EasyEdit artifact JSON")
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


def main() -> int:
    args = build_parser().parse_args()
    source = Path(args.input)
    artifact = json.loads(source.read_text())
    payload = diagnostic_payload(artifact, str(source))
    output = Path(args.output) if args.output else source.with_name(
        f"{source.stem}.mquake_style{source.suffix or '.json'}"
    )
    output.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False))
    print(f"MQuAKE-style diagnostic saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
