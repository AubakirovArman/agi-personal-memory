"""KnowEdit-inspired product diagnostics for EasyEdit artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .easyedit_utils import jsonable


SCHEMA_VERSION = "product_diagnostic.v1"


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
        "summary": summarize_product_rows(rows),
        "by_relation_id": summarize_product_by_relation(rows),
        "caveat": (
            "KnowEdit-inspired product diagnostic over EasyEdit artifact fields; "
            "not an external KnowEdit/UniEdit/ScEdit leaderboard result."
        ),
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


def main() -> int:
    args = build_parser().parse_args()
    source = Path(args.input)
    artifact = json.loads(source.read_text())
    payload = diagnostic_payload(artifact, str(source))
    output = Path(args.output) if args.output else source.with_name(
        f"{source.stem}.product{source.suffix or '.json'}"
    )
    output.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False))
    print(f"Product diagnostic saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
