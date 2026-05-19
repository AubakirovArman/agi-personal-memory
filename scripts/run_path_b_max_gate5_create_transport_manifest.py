#!/usr/bin/env python3
"""Create immutable transport manifest for PATH_B_MAX Gate 5 public artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_FIELDS = {
    "release": "release_schema_version",
    "index": "index_schema_version",
    "receipt": "receipt_schema_version",
    "bundle": "bundle_schema_version",
}
DISALLOWED_PRODUCTION_STORAGE_PROVIDERS = {
    "",
    "filesystem",
    "file_system",
    "filesystem_storage",
    "local",
    "local_filesystem",
    "mock-object-store",
}


def _storage_provider_is_disallowed(provider: Optional[str]) -> bool:
    normalized = (provider or "").strip().lower()
    return normalized in DISALLOWED_PRODUCTION_STORAGE_PROVIDERS


def _is_local_base_url(url: Optional[str]) -> bool:
    normalized = (url or "").strip().lower()
    return (
        normalized.startswith("http://localhost")
        or normalized.startswith("https://localhost")
        or normalized.startswith("http://127.0.0.1")
        or normalized.startswith("https://127.0.0.1")
        or normalized.startswith("http://0.0.0.0")
        or normalized.startswith("https://0.0.0.0")
        or normalized.startswith("http://[::1]")
        or normalized.startswith("https://[::1]")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create transport manifest for PATH_B_MAX Gate 5 publication artifacts.",
    )
    parser.add_argument(
        "--release-path",
        type=Path,
        default=Path("results/easyedit_official/governance/path_b_max_gate5_public_release.json"),
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
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("results/easyedit_official/governance/path_b_max_gate5_public_transport_manifest.json"),
    )
    parser.add_argument("--tenant", default="public")
    parser.add_argument("--channel", default="public")
    parser.add_argument(
        "--storage-provider",
        default="local_filesystem",
        help="Storage type label used in manifest metadata",
    )
    parser.add_argument(
        "--immutability-mode",
        default="local_only",
        help="Logical immutability mode reported in manifest (local_only, object_lock, versioned_bucket, signed_pointers)",
    )
    parser.add_argument(
        "--retention-policy",
        default="workspace-local",
        help="Retention policy label captured in manifest metadata",
    )
    parser.add_argument(
        "--public-base-url",
        default=None,
        help="Optional base URL for derived external URIs (if external transport is available)",
    )
    parser.add_argument(
        "--require-production-external",
        action="store_true",
        help=(
            "Require full production-oriented transport: all artifact schemas present, all "
            "external URIs present, non-local immutability mode, and storage metadata"
        ),
    )
    parser.add_argument("--external-index-uri", default=None)
    parser.add_argument("--external-release-uri", default=None)
    parser.add_argument("--external-receipt-uri", default=None)
    parser.add_argument("--external-bundle-uri", default=None)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _schema_for_kind(kind: str, payload: Dict[str, Any]) -> Optional[str]:
    key = SCHEMA_FIELDS[kind]
    value = payload.get(key)
    if isinstance(value, str):
        return value
    return None


def _artifact_entry(
    kind: str,
    path: Path,
    external_uri: Optional[str],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {kind} artifact at {path}")
    size = path.stat().st_size
    return {
        "kind": kind,
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": size,
        "schema_version": _schema_for_kind(kind, payload),
        "external_uri": external_uri,
    }


def _default_external_uri(base_url: Optional[str], kind: str, release_sha: Optional[str]) -> Optional[str]:
    if not base_url:
        return None
    base = base_url.rstrip("/")
    if kind == "release":
        if not release_sha:
            return None
        return f"{base}/gate5/releases/{release_sha}.json"
    if kind == "index":
        return f"{base}/gate5/index.json"
    if kind == "receipt":
        return f"{base}/gate5/receipt.json"
    if kind == "bundle":
        return f"{base}/gate5/bundle.json"
    return None


def _build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    release_payload = _load_json(args.release_path)
    index_payload = _load_json(args.index_path)
    receipt_payload = _load_json(args.receipt_path)
    bundle_payload = _load_json(args.bundle_path)

    release_sha = _sha256_file(args.release_path)
    artifacts: List[Dict[str, Any]] = [
        _artifact_entry(
            "release",
            args.release_path,
            args.external_release_uri
            or _default_external_uri(args.public_base_url, "release", release_sha),
            release_payload,
        ),
        _artifact_entry(
            "index",
            args.index_path,
            args.external_index_uri
            or _default_external_uri(args.public_base_url, "index", release_sha),
            index_payload,
        ),
        _artifact_entry(
            "receipt",
            args.receipt_path,
            args.external_receipt_uri
            or _default_external_uri(args.public_base_url, "receipt", release_sha),
            receipt_payload,
        ),
        _artifact_entry(
            "bundle",
            args.bundle_path,
            args.external_bundle_uri
            or _default_external_uri(args.public_base_url, "bundle", release_sha),
            bundle_payload,
        ),
    ]

    has_external = all(bool(a.get("external_uri")) for a in artifacts)
    has_all_schema = all(bool(a.get("schema_version")) for a in artifacts)
    immutability_mode = (args.immutability_mode or "").strip().lower()
    has_storage_provider = bool((args.storage_provider or "").strip())
    has_nonlocal_public_base_url = bool(args.public_base_url) and not _is_local_base_url(args.public_base_url)
    production_ready = (
        bool(has_external)
        and bool(has_all_schema)
        and has_storage_provider
        and not _storage_provider_is_disallowed(args.storage_provider)
        and has_nonlocal_public_base_url
        and immutability_mode not in {"", "none", "local_only"}
    )

    if args.require_production_external:
        if _storage_provider_is_disallowed(args.storage_provider):
            raise ValueError(
                f"Production manifest requires real external storage provider, got '{args.storage_provider}'"
            )
        if not args.public_base_url:
            raise ValueError("Production manifest requires public_base_url")
        if _is_local_base_url(args.public_base_url):
            raise ValueError(
                "Production manifest requires a non-local public_base_url (localhost/loopback not allowed)"
            )
        if not has_all_schema:
            raise ValueError("Production manifest requires schema_version for every artifact")
        if not has_external:
            missing_kinds = [a.get("kind", "unknown") for a in artifacts if not a.get("external_uri")]
            raise ValueError(f"Production manifest requires external_uri for all artifacts, missing: {missing_kinds}")
        if immutability_mode in {"", "none", "local_only"}:
            raise ValueError(
                "Production manifest requires non-local immutability_mode (for example: object_lock, versioned_bucket, signed_pointers)"
            )
        if not has_storage_provider:
            raise ValueError("Production manifest requires storage_provider when production mode is requested")

    return {
        "transport_schema_version": "path_b_max_gate5_transport_manifest.v1",
        "tenant_scope": args.tenant,
        "channel": args.channel,
        "published_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "transport_metadata": {
            "storage_provider": args.storage_provider,
            "immutability_mode": args.immutability_mode,
            "retention_policy": args.retention_policy,
            "public_base_url": args.public_base_url,
        },
        "artifacts": artifacts,
        "coverage": {
            "has_external_uris": bool(has_external),
            "has_all_schema_versions": bool(has_all_schema),
            "production_ready_candidate": bool(production_ready),
            "artifact_count": len(artifacts),
        },
    }


def main() -> int:
    args = parse_args()
    manifest = _build_manifest(args)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"written transport manifest to {args.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
