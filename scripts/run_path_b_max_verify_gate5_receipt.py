#!/usr/bin/env python3
"""Verify immutable receipt for PATH_B_MAX Gate 5 public artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify PATH_B_MAX Gate 5 receipt")
    parser.add_argument(
        "--receipt-path",
        type=Path,
        default=Path("results/easyedit_official/governance/path_b_max_gate5_public_receipt.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.receipt_path.exists():
        print(f"Missing receipt: {args.receipt_path}")
        return 1

    payload = read_json(args.receipt_path)
    if payload.get("receipt_schema_version") != "path_b_max_gate5_receipt.v1":
        print("Unexpected receipt schema version")
        return 1

    index_path = Path(payload.get("index_path", ""))
    if not index_path.exists():
        print(f"Missing index in receipt: {index_path}")
        return 1

    ok = True
    if payload.get("index_sha256") != sha256_file(index_path):
        print("Receipt index sha mismatch")
        ok = False

    records = payload.get("records", [])
    if not records:
        print("Receipt has no records")
        return 1

    for rec in records:
        release_path = Path(rec.get("release_path", ""))
        if not release_path.exists():
            ok = False
            print(f"Missing release file: {release_path}")
            continue
        if rec.get("release_file_sha256") != sha256_file(release_path):
            ok = False
            print(f"Release file sha mismatch: {release_path}")

    if payload.get("record_count") != len(records):
        ok = False
        print("record_count mismatch in receipt")

    if ok:
        print(f"PATH_B_MAX Gate5 receipt OK: {args.receipt_path}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
