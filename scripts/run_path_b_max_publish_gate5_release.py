#!/usr/bin/env python3
"""Publish PATH B MAX Gate 5 release packets into a tenant-scoped index."""

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


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_entry(
    release_path: Path,
    tenant: str,
    channel: str,
    note: str | None,
) -> Dict[str, Any]:
    release = read_json(release_path)
    return {
        "tenant_scope": tenant,
        "channel": channel,
        "release_path": str(release_path),
        "release_sha256": sha256_bytes(release_path.read_bytes()),
        "proof_sha256": release.get("proof_sha256"),
        "claim_chain_sha256": release.get("claim_lock", {}).get("audit_chain_sha256"),
        "claim_chain_len": release.get("claim_lock", {}).get("audit_chain_len"),
        "signature": release.get("signature"),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": note,
        "source_path": release.get("source_path"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish PATH_B_MAX gate-5 public release packets",
    )
    parser.add_argument(
        "--release-path",
        type=Path,
        required=True,
        help="Release packet to publish",
    )
    parser.add_argument(
        "--tenant",
        default="public",
        help="Tenant scope for the published index entry",
    )
    parser.add_argument(
        "--channel",
        default="public",
        help="Distribution channel name for governance publish",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("results/easyedit_official/governance/path_b_max_gate5_public_index.json"),
        help="Path to tenant index",
    )
    parser.add_argument(
        "--note",
        default=None,
        help="Optional note stored with the published release entry",
    )
    return parser.parse_args()


def load_index(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("index payload must be a JSON object")
    records = payload.get("records")
    if records is None:
        raise ValueError("index payload missing records field")
    if not isinstance(records, list):
        raise ValueError("index records field must be list")
    return records


def main() -> int:
    args = parse_args()
    if not args.release_path.exists():
        print(f"Missing release packet: {args.release_path}")
        return 1

    index_payload = {
        "index_schema_version": "path_b_max_gate5_index.v1",
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "records": [],
    }
    try:
        index_payload["records"] = load_index(args.index_path)
    except FileNotFoundError:
        pass

    record = build_entry(args.release_path, args.tenant, args.channel, args.note)
    existing: List[Dict[str, Any]] = index_payload["records"]
    release_key = str(args.release_path)
    new_records = [
        r
        for r in existing
        if not (
            r.get("tenant_scope") == args.tenant
            and r.get("channel") == args.channel
            and r.get("release_path") == release_key
        )
    ]
    new_records.append(record)

    index_payload["records"] = new_records
    write_json(args.index_path, index_payload)
    print(f"published {args.release_path} to {args.index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
