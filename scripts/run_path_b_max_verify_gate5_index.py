#!/usr/bin/env python3
"""Verify PATH_B_MAX Gate 5 public index records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify public index for PATH_B_MAX gate 5 releases",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("results/easyedit_official/governance/path_b_max_gate5_public_index.json"),
    )
    parser.add_argument("--tenant", default=None, help="Optional tenant filter")
    parser.add_argument(
        "--channel",
        default=None,
        help="Optional channel filter",
    )
    return parser.parse_args()


def normalize(records: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(records, list):
        return []
    return [r for r in records if isinstance(r, dict)]


def main() -> int:
    args = parse_args()
    if not args.index_path.exists():
        print(f"Missing index file: {args.index_path}")
        return 1
    payload = read_json(args.index_path)
    if payload.get("index_schema_version") != "path_b_max_gate5_index.v1":
        print("Unexpected index_schema_version")
        return 1

    ok = True
    matched_records = 0
    for rec in normalize(payload.get("records")):
        if args.tenant is not None and rec.get("tenant_scope") != args.tenant:
            continue
        if args.channel is not None and rec.get("channel") != args.channel:
            continue
        matched_records += 1
        release_path = rec.get("release_path")
        if not release_path:
            print("Index record missing release_path")
            ok = False
            continue
        release_file = Path(release_path)
        if not release_file.exists():
            print(f"Missing release file for index record: {release_path}")
            ok = False
            continue
        digest = sha256_file(release_file)
        if rec.get("release_sha256") != digest:
            print(f"release_sha256 mismatch for {release_path}")
            ok = False

    if matched_records == 0:
        print(
            f"Index has no matching records for tenant={args.tenant}, channel={args.channel}"
        )
        ok = False

    if ok:
        print(f"PATH_B_MAX Gate5 index OK: {args.index_path}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
