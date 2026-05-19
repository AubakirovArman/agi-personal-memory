#!/usr/bin/env python3
"""Build a relation->profile override map for PS@All ablation sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build relation-aware profile map from an official results payload.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to official payload JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/easyedit_official/ablations/relation_profile_map_seed42.json"),
        help="Output relation profile JSON path",
    )
    parser.add_argument(
        "--ps-threshold",
        type=float,
        default=0.30,
        help="Map relation to positive profile when TF PS@All < threshold",
    )
    parser.add_argument(
        "--locality-threshold",
        type=float,
        default=0.95,
        help="Map relation to anti profile when TF locality < threshold",
    )
    parser.add_argument("--min-count", type=int, default=1, help="Min facts per relation")
    parser.add_argument("--positive-profile", default="w025")
    parser.add_argument("--anti-profile", default="target_low")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print map as JSON and skip writing file",
    )
    return parser.parse_args()


def _coerce_float(value: object) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_relation_profile_map(
    payload: dict,
    ps_threshold: float,
    locality_threshold: float,
    min_count: int,
    positive_profile: str,
    anti_profile: str,
) -> dict:
    metrics_by_relation = payload.get("summary", {}).get("metrics_by_relation_id", {})
    if not isinstance(metrics_by_relation, dict):
        raise ValueError("Input payload does not contain summary.metrics_by_relation_id")

    relation_profile_map: dict[str, dict[str, str]] = {}
    for relation_id, relation_metrics in sorted(metrics_by_relation.items(), key=lambda item: str(item[0])):
        if not isinstance(relation_metrics, dict):
            continue
        n = int(relation_metrics.get("n", 0) or 0)
        if n < min_count:
            continue

        ps_all = _coerce_float(relation_metrics.get("rephrase_all_acc"))
        locality = _coerce_float(relation_metrics.get("locality_acc"))

        profile: dict[str, str] = {}
        if ps_all < ps_threshold:
            profile["positive_profile"] = positive_profile
        if locality < locality_threshold:
            profile["anti_profile"] = anti_profile
        if profile:
            relation_profile_map[str(relation_id)] = profile

    return relation_profile_map


def main() -> int:
    args = _parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input payload not found: {args.input}")

    with args.input.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in payload: {args.input}")

    relation_profile_map = build_relation_profile_map(
        payload=payload,
        ps_threshold=args.ps_threshold,
        locality_threshold=args.locality_threshold,
        min_count=args.min_count,
        positive_profile=args.positive_profile,
        anti_profile=args.anti_profile,
    )

    json_text = json.dumps(
        relation_profile_map,
        ensure_ascii=False,
        indent=2 if args.pretty else None,
        sort_keys=True,
    )

    if args.print_only:
        print(json_text)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json_text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
