#!/usr/bin/env python3
"""Verify PATH_B_MAX Gate 5 transport manifest integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify PATH_B_MAX Gate 5 transport manifest.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("results/easyedit_official/governance/path_b_max_gate5_public_transport_manifest.json"),
    )
    parser.add_argument(
        "--expected-manifest-schema-version",
        default="path_b_max_gate5_transport_manifest.v1",
    )
    parser.add_argument(
        "--require-production-external",
        action="store_true",
        help=(
            "Require production-oriented transport claims: all external URIs present "
            "and non-local immutability mode."
        ),
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _storage_provider_is_disallowed(provider: Any) -> bool:
    normalized = str(provider or "").strip().lower()
    return normalized in {"", "filesystem", "file_system", "filesystem_storage", "local", "local_filesystem", "mock-object-store"}


def _is_local_url(url: Any) -> bool:
    base = str(url or "").strip().lower()
    return (
        base.startswith("http://localhost")
        or base.startswith("https://localhost")
        or base.startswith("http://127.0.0.1")
        or base.startswith("https://127.0.0.1")
        or base.startswith("http://0.0.0.0")
        or base.startswith("https://0.0.0.0")
        or base.startswith("http://[::1]")
        or base.startswith("https://[::1]")
    )


def _verify() -> list[str]:
    args = parse_args()
    manifest = _load_json(args.manifest_path)
    errors: list[str] = []

    if manifest.get("transport_schema_version") != args.expected_manifest_schema_version:
        errors.append(
            "transport_schema_version mismatch: "
            f"{manifest.get('transport_schema_version')} != {args.expected_manifest_schema_version}"
        )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("manifest artifacts missing or not a list")
        return errors

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            errors.append("artifact entry not a dict")
            continue
        kind = artifact.get("kind")
        path = artifact.get("path")
        sha = artifact.get("sha256")
        size_bytes = artifact.get("size_bytes")
        schema_version = artifact.get("schema_version")
        if not isinstance(kind, str):
            errors.append("artifact entry missing kind")
            continue
        if not isinstance(path, str):
            errors.append(f"{kind}: artifact path missing")
            continue
        if not isinstance(schema_version, str) or not schema_version:
            errors.append(f"{kind}: schema_version missing")
        file_path = Path(path)
        if not file_path.exists():
            errors.append(f"{kind}: missing file at {path}")
            continue
        if not isinstance(sha, str) or _sha256_file(file_path) != sha:
            errors.append(f"{kind}: sha256 mismatch at {path}")
        if not isinstance(size_bytes, int) or file_path.stat().st_size != size_bytes:
            errors.append(f"{kind}: size_bytes mismatch at {path}")
        if args.require_production_external and not artifact.get("external_uri"):
            errors.append(f"{kind}: external_uri missing while --require-production-external is set")

    if args.require_production_external:
        coverage = manifest.get("coverage")
        if not isinstance(coverage, dict):
            errors.append("coverage missing or not a dict while --require-production-external is set")
        else:
            if not coverage.get("has_external_uris"):
                errors.append(
                    "coverage.has_external_uris must be true when --require-production-external is set"
                )
            if not coverage.get("has_all_schema_versions"):
                errors.append(
                    "coverage.has_all_schema_versions must be true when --require-production-external is set"
                )
            if not coverage.get("production_ready_candidate"):
                errors.append(
                    "coverage.production_ready_candidate must be true when --require-production-external is set"
                )
            if coverage.get("artifact_count") != len(artifacts):
                errors.append("coverage.artifact_count mismatch")
        transport_metadata = manifest.get("transport_metadata")
        if not isinstance(transport_metadata, dict):
            errors.append("transport_metadata missing or not a dict while --require-production-external is set")
        else:
            immutability_mode = transport_metadata.get("immutability_mode")
            if immutability_mode in {None, "", "none", "local_only"}:
                errors.append("transport_metadata.immutability_mode must be non-local when production mode required")
            if not transport_metadata.get("storage_provider"):
                errors.append("transport_metadata.storage_provider missing when production mode required")
            elif _storage_provider_is_disallowed(transport_metadata.get("storage_provider")):
                errors.append(
                    f"transport_metadata.storage_provider '{transport_metadata.get('storage_provider')}' "
                    "is not allowed for production mode"
                )
            if not transport_metadata.get("public_base_url"):
                errors.append("transport_metadata.public_base_url missing when production mode required")
            elif _is_local_url(transport_metadata.get("public_base_url")):
                errors.append(
                    "transport_metadata.public_base_url must be non-local when production mode required"
                )
        if not all(
            isinstance(artifact, dict) and bool(artifact.get("schema_version")) and bool(artifact.get("external_uri"))
            for artifact in artifacts
        ):
            errors.append(
                "all artifacts must provide schema_version and external_uri when --require-production-external is set"
            )

    return errors


def main() -> int:
    errors = _verify()
    if errors:
        print("Transport manifest verification failed:")
        for issue in errors:
            print(f"- {issue}")
        return 1
    print("PATH_B_MAX Gate5 transport manifest verification OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
