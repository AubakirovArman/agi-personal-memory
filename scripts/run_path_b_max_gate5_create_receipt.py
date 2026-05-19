#!/usr/bin/env python3
"""Create an immutable receipt for PATH_B_MAX Gate 5 public artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create immutable PATH_B_MAX Gate 5 public artifact receipt",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("results/easyedit_official/governance/path_b_max_gate5_public_index.json"),
        help="Path to public index JSON",
    )
    parser.add_argument(
        "--receipt-path",
        type=Path,
        default=Path("results/easyedit_official/governance/path_b_max_gate5_public_receipt.json"),
        help="Output path for generated receipt",
    )
    parser.add_argument("--tenant", default="public")
    parser.add_argument("--channel", default="public")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.index_path.exists():
        print(f"Missing index: {args.index_path}")
        return 1

    payload = read_json(args.index_path)
    if payload.get("index_schema_version") != "path_b_max_gate5_index.v1":
        print("Unexpected index schema version")
        return 1

    records = payload.get("records", [])
    if not isinstance(records, list):
        print("Index records must be array")
        return 1

    filtered: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if rec.get("tenant_scope") != args.tenant:
            continue
        if rec.get("channel") != args.channel:
            continue
        filtered.append(rec)

    if not filtered:
        print(f"No matching records for tenant={args.tenant}, channel={args.channel}")
        return 1

    index_sha = sha256_file(args.index_path)
    receipt = {
        "receipt_schema_version": "path_b_max_gate5_receipt.v1",
        "tenant_scope": args.tenant,
        "channel": args.channel,
        "index_path": str(args.index_path),
        "index_sha256": index_sha,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "records": [],
    }

    for rec in filtered:
        release_path = Path(rec.get("release_path", ""))
        if not release_path.exists():
            print(f"Missing release in receipt: {rec.get('release_path')}")
            return 1
        release_file_sha = sha256_file(release_path)
        release_index_sha = rec.get("release_sha256")
        if release_index_sha != release_file_sha:
            print(f"Release sha mismatch for {release_path}: index={release_index_sha}, file={release_file_sha}")
            return 1
        item = {
            "release_path": str(release_path),
            "release_sha256": release_index_sha,
            "release_file_sha256": release_file_sha,
            "proof_sha256": rec.get("proof_sha256"),
            "claim_chain_sha256": rec.get("claim_chain_sha256"),
            "claim_chain_len": rec.get("claim_chain_len"),
        }
        receipt["records"].append(item)

    receipt["record_count"] = len(receipt["records"])

    args.receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt_path.open("w", encoding="utf-8") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"created receipt: {args.receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
