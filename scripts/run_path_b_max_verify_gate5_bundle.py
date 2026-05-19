#!/usr/bin/env python3
"""Verify PATH_B_MAX Gate 5 public artifact bundle integrity."""

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
    parser = argparse.ArgumentParser(description="Verify PATH_B_MAX Gate 5 artifact bundle")
    parser.add_argument(
        "--bundle-path",
        type=Path,
        default=Path("results/easyedit_official/governance/path_b_max_gate5_public_bundle.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.bundle_path.exists():
        print(f"Missing bundle: {args.bundle_path}")
        return 1

    payload = read_json(args.bundle_path)
    if payload.get("bundle_schema_version") != "path_b_max_gate5_bundle.v1":
        print("Unexpected bundle schema version")
        return 1

    ok = True
    artifacts = payload.get("artifacts", [])
    if not isinstance(artifacts, list) or not artifacts:
        print("Bundle has no artifacts")
        return 1

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            ok = False
            print("Invalid artifact entry")
            continue
        path = Path(artifact.get("path", ""))
        expected = artifact.get("sha256")
        if not path.exists():
            ok = False
            print(f"Missing artifact file: {path}")
            continue
        if artifact.get("size_bytes") != path.stat().st_size:
            ok = False
            print(f"Artifact size mismatch: {path}")
        if expected != sha256_file(path):
            ok = False
            print(f"Artifact sha mismatch: {path}")

    index_path = Path(payload.get("index_path", ""))
    if not index_path.exists() or payload.get("index_sha256") != sha256_file(index_path):
        ok = False
        print("Index file mismatch")
    receipt_path = Path(payload.get("receipt_path", ""))
    if not receipt_path.exists() or payload.get("receipt_sha256") != sha256_file(receipt_path):
        ok = False
        print("Receipt file mismatch")

    if payload.get("bundle_schema_version") != "path_b_max_gate5_bundle.v1":
        ok = False

    if ok:
        print(f"PATH_B_MAX Gate5 bundle OK: {args.bundle_path}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
