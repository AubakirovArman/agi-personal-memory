#!/usr/bin/env python3
"""Create an immutable PATH_B_MAX Gate 5 public artifact bundle."""

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


def file_size(path: Path) -> int:
    return path.stat().st_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create immutable public artifact bundle for PATH_B_MAX Gate 5",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("results/easyedit_official/governance/path_b_max_gate5_public_index.json"),
    )
    parser.add_argument(
        "--receipt-path",
        type=Path,
        default=Path("results/easyedit_official/governance/path_b_max_gate5_public_receipt.json"),
    )
    parser.add_argument(
        "--bundle-path",
        type=Path,
        default=Path("results/easyedit_official/governance/path_b_max_gate5_public_bundle.json"),
    )
    parser.add_argument("--tenant", default="public")
    parser.add_argument("--channel", default="public")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.index_path.exists():
        print(f"Missing index: {args.index_path}")
        return 1
    if not args.receipt_path.exists():
        print(f"Missing receipt: {args.receipt_path}")
        return 1

    index_payload = read_json(args.index_path)
    receipt_payload = read_json(args.receipt_path)
    if index_payload.get("index_schema_version") != "path_b_max_gate5_index.v1":
        print("Unexpected index schema version")
        return 1
    if receipt_payload.get("receipt_schema_version") != "path_b_max_gate5_receipt.v1":
        print("Unexpected receipt schema version")
        return 1

    artifacts: List[Dict[str, Any]] = []
    bundle_files = [
        ("index", args.index_path),
        ("receipt", args.receipt_path),
    ]

    for rec in index_payload.get("records", []):
        if not isinstance(rec, dict):
            continue
        if rec.get("tenant_scope") != args.tenant:
            continue
        if rec.get("channel") != args.channel:
            continue
        release_path = Path(rec.get("release_path", ""))
        if not release_path.exists():
            print(f"Missing release file: {release_path}")
            return 1
        bundle_files.append((f"release:{release_path}", release_path))

    release_count = 0
    for kind, path in bundle_files:
        artifacts.append(
            {
                "kind": kind,
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": file_size(path),
            }
        )
        if kind.startswith("release:"):
            release_count += 1

    if release_count == 0:
        print(f"No release records for tenant={args.tenant}, channel={args.channel}")
        return 1

    bundle = {
        "bundle_schema_version": "path_b_max_gate5_bundle.v1",
        "tenant_scope": args.tenant,
        "channel": args.channel,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "index_path": str(args.index_path),
        "index_sha256": sha256_file(args.index_path),
        "receipt_path": str(args.receipt_path),
        "receipt_sha256": sha256_file(args.receipt_path),
        "release_count": release_count,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }

    args.bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with args.bundle_path.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"created bundle: {args.bundle_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
