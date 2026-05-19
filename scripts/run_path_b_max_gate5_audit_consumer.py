#!/usr/bin/env python3
"""Audit consumer for PATH_B_MAX Gate 5 public API."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
import hashlib
from typing import Any, Dict, List


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consume and verify PATH_B_MAX Gate 5 public API artifacts",
    )
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:8010",
        help="Base URL of Gate 5 API",
    )
    parser.add_argument("--tenant", default="public")
    parser.add_argument("--channel", default="public")
    parser.add_argument(
        "--check-receipt",
        action="store_true",
        help="Fetch /gate5/receipt.json and validate JSON payload",
    )
    parser.add_argument(
        "--check-bundle",
        action="store_true",
        help="Fetch /gate5/bundle.json and validate JSON payload",
    )
    parser.add_argument(
        "--check-transport-manifest",
        action="store_true",
        help="Fetch /gate5/transport-manifest.json and validate JSON payload",
    )
    parser.add_argument(
        "--require-production-external",
        action="store_true",
        help="Require production-oriented transport guarantees in manifest (external URIs + non-local immutability)",
    )
    parser.add_argument(
        "--expected-bundle-sha256",
        default=None,
        help="Optional SHA256 to compare against /gate5/bundle.json bytes",
    )
    parser.add_argument(
        "--expected-transport-manifest-sha256",
        default=None,
        help="Optional SHA256 to compare against /gate5/transport-manifest.json bytes",
    )
    parser.add_argument(
        "--expected-receipt-sha256",
        default=None,
        help="Optional SHA256 to compare against /gate5/receipt.json bytes",
    )
    parser.add_argument(
        "--expected-release-sha256",
        default=None,
        help="Optional SHA256 to compare against /gate5/releases/<sha>.json bytes",
    )
    parser.add_argument(
        "--expected-release-schema-version",
        default="path_b_max_gate5_release.v1",
        help="Expected release schema version for /gate5/releases/<sha>.json",
    )
    parser.add_argument(
        "--expected-receipt-schema-version",
        default="path_b_max_gate5_receipt.v1",
        help="Expected receipt schema version for /gate5/receipt.json",
    )
    parser.add_argument(
        "--expected-bundle-schema-version",
        default="path_b_max_gate5_bundle.v1",
        help="Expected bundle schema version for /gate5/bundle.json",
    )
    parser.add_argument(
        "--expected-transport-manifest-schema-version",
        default="path_b_max_gate5_transport_manifest.v1",
        help="Expected transport manifest schema version for /gate5/transport-manifest.json",
    )
    return parser.parse_args()


def read_json_url(url: str) -> tuple[bytes, Dict[str, Any]]:
    with urllib.request.urlopen(url, timeout=10) as r:
        if r.status != 200:
            raise RuntimeError(f"{url} returned status {r.status}")
        payload = r.read()
    return payload, json.loads(payload.decode("utf-8"))


def _storage_provider_is_disallowed(provider: Any) -> bool:
    normalized = str(provider or "").strip().lower()
    return normalized in {"", "filesystem", "file_system", "filesystem_storage", "local", "local_filesystem", "mock-object-store"}


def main() -> int:
    args = parse_args()
    index_url = f"{args.api_base.rstrip('/')}/gate5/index.json"

    try:
        _, index_payload = read_json_url(index_url)
    except (urllib.error.URLError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"failed to read index: {exc}")
        return 1

    if index_payload.get("index_schema_version") != "path_b_max_gate5_index.v1":
        print("unexpected index schema version")
        return 1

    records = index_payload.get("records", [])
    expected = [
        r
        for r in records
        if isinstance(r, dict)
        and r.get("tenant_scope") == args.tenant
        and r.get("channel") == args.channel
    ]
    if not expected:
        print(f"no matching records for tenant={args.tenant} channel={args.channel}")
        return 1

    ok = True
    released: List[str] = []
    for rec in expected:
        release_sha = rec.get("release_sha256")
        release_path = rec.get("release_path")
        if not release_sha or not release_path:
            ok = False
            print("index record missing release_sha256/release_path")
            continue
        release_url = f"{args.api_base.rstrip('/')}/gate5/releases/{release_sha}.json"
        try:
            release_bytes, release_payload = read_json_url(release_url)
        except (urllib.error.URLError, RuntimeError, json.JSONDecodeError) as exc:
            ok = False
            print(f"release fetch failed for {release_sha}: {exc}")
            continue
        fetched_sha = sha256_bytes(release_bytes)
        if fetched_sha != release_sha:
            ok = False
            print(
                f"release hash mismatch for {release_sha}: "
                f"expected {release_sha}, fetched {fetched_sha}"
            )
            continue
        if args.expected_release_sha256 is not None and fetched_sha != args.expected_release_sha256:
            ok = False
            print(
                f"release API hash mismatch for {release_sha}: "
                f"expected {args.expected_release_sha256}, fetched {fetched_sha}"
            )
            continue
        claim_lock = release_payload.get("claim_lock")
        if not isinstance(claim_lock, dict):
            ok = False
            print(f"release payload missing claim_lock for {release_sha}")
            continue
        if rec.get("claim_chain_sha256") is not None:
            if claim_lock.get("audit_chain_sha256") != rec.get("claim_chain_sha256"):
                ok = False
                print(
                    f"claim_chain_sha256 mismatch for {release_sha}: "
                    f"index={rec.get('claim_chain_sha256')} release={claim_lock.get('audit_chain_sha256')}"
                )
                continue
        if rec.get("claim_chain_len") is not None:
            if claim_lock.get("audit_chain_len") != rec.get("claim_chain_len"):
                ok = False
                print(
                    f"claim_chain_len mismatch for {release_sha}: "
                    f"index={rec.get('claim_chain_len')} release={claim_lock.get('audit_chain_len')}"
                )
                continue
        proof = release_payload.get("proof")
        if proof is not None and not isinstance(proof, dict):
            ok = False
            print(f"release payload proof object invalid for {release_sha}")
            continue
        release_proof_sha = release_payload.get("proof_sha256")
        if release_proof_sha is None and isinstance(proof, dict):
            release_proof_sha = proof.get("proof_sha256")
        if rec.get("proof_sha256") is not None:
            if release_proof_sha != rec.get("proof_sha256"):
                ok = False
                print(
                    f"proof_sha256 mismatch for {release_sha}: "
                    f"index={rec.get('proof_sha256')} release={release_proof_sha}"
                )
                continue
        if (
            args.expected_release_schema_version
            and release_payload.get("release_schema_version") != args.expected_release_schema_version
        ):
            ok = False
            print(
                f"release schema mismatch for {release_sha}: expected "
                f"{args.expected_release_schema_version}, got {release_payload.get('release_schema_version')}"
            )
            continue
        released.append(release_path)

        if args.check_receipt:
            receipt_url = f"{args.api_base.rstrip('/')}/gate5/receipt.json"
            try:
                receipt_bytes, receipt_payload = read_json_url(receipt_url)
            except (urllib.error.URLError, RuntimeError, json.JSONDecodeError) as exc:
                ok = False
                print(f"receipt fetch failed: {exc}")
            else:
                if (
                    args.expected_receipt_schema_version
                    and receipt_payload.get("receipt_schema_version") != args.expected_receipt_schema_version
                ):
                    ok = False
                    print(
                        "receipt schema mismatch: expected "
                        f"{args.expected_receipt_schema_version}, got "
                        f"{receipt_payload.get('receipt_schema_version')}"
                    )
                if args.expected_receipt_sha256 is not None:
                    actual_sha = sha256_bytes(receipt_bytes)
                    if actual_sha != args.expected_receipt_sha256:
                        ok = False
                        print(
                            f"receipt hash mismatch: expected {args.expected_receipt_sha256}, "
                            f"actual {actual_sha}"
                        )

    if args.check_bundle:
        bundle_url = f"{args.api_base.rstrip('/')}/gate5/bundle.json"
        try:
            bundle_bytes, bundle_payload = read_json_url(bundle_url)
        except (urllib.error.URLError, RuntimeError, json.JSONDecodeError) as exc:
            ok = False
            print(f"bundle fetch failed: {exc}")
        else:
            if (
                args.expected_bundle_schema_version
                and bundle_payload.get("bundle_schema_version") != args.expected_bundle_schema_version
            ):
                ok = False
                print(
                    "bundle schema mismatch: expected "
                    f"{args.expected_bundle_schema_version}, got "
                    f"{bundle_payload.get('bundle_schema_version')}"
                )
            if args.expected_bundle_sha256 is not None:
                actual_sha = sha256_bytes(bundle_bytes)
                if actual_sha != args.expected_bundle_sha256:
                    ok = False
                    print(
                        f"bundle hash mismatch: expected {args.expected_bundle_sha256}, actual {actual_sha}"
                    )

    if args.check_transport_manifest:
        transport_url = f"{args.api_base.rstrip('/')}/gate5/transport-manifest.json"
        try:
            transport_bytes, transport_payload = read_json_url(transport_url)
        except (urllib.error.URLError, RuntimeError, json.JSONDecodeError) as exc:
            ok = False
            print(f"transport-manifest fetch failed: {exc}")
        else:
            if (
                args.expected_transport_manifest_schema_version
                and transport_payload.get("transport_schema_version")
                != args.expected_transport_manifest_schema_version
            ):
                ok = False
                print(
                    "transport-manifest schema mismatch: expected "
                    f"{args.expected_transport_manifest_schema_version}, got "
                    f"{transport_payload.get('transport_schema_version')}"
                )
            if args.expected_transport_manifest_sha256 is not None:
                actual_transport_sha = sha256_bytes(transport_bytes)
                if actual_transport_sha != args.expected_transport_manifest_sha256:
                    ok = False
                    print(
                        f"transport-manifest hash mismatch: expected "
                        f"{args.expected_transport_manifest_sha256}, actual {actual_transport_sha}"
                    )
            if args.require_production_external:
                coverage = transport_payload.get("coverage")
                if not isinstance(coverage, dict):
                    ok = False
                    print("transport-manifest coverage missing or not a dict while --require-production-external is set")
                else:
                    if not coverage.get("has_external_uris"):
                        ok = False
                        print(
                            "transport-manifest production mismatch: coverage.has_external_uris must be true"
                        )
                    if not coverage.get("has_all_schema_versions"):
                        ok = False
                        print(
                            "transport-manifest production mismatch: coverage.has_all_schema_versions must be true"
                        )
                    if not coverage.get("production_ready_candidate"):
                        ok = False
                        print(
                            "transport-manifest production readiness mismatch: "
                            "coverage.production_ready_candidate must be true"
                        )
                    if coverage.get("artifact_count") != len(transport_payload.get("artifacts", [])):
                        ok = False
                        print("transport-manifest production mismatch: coverage.artifact_count mismatch")
                transport_metadata = transport_payload.get("transport_metadata")
                if not isinstance(transport_metadata, dict):
                    ok = False
                    print(
                        "transport-manifest transport_metadata missing while --require-production-external is set"
                    )
                else:
                    immutability_mode = transport_metadata.get("immutability_mode")
                    if immutability_mode in {None, "", "none", "local_only"}:
                        ok = False
                        print(
                            "transport-manifest immutability_mode must be non-local when --require-production-external is set"
                        )
                    if not transport_metadata.get("storage_provider"):
                        ok = False
                        print(
                            "transport-manifest storage_provider missing while --require-production-external is set"
                        )
                    elif _storage_provider_is_disallowed(transport_metadata.get("storage_provider")):
                        ok = False
                        print(
                            "transport-manifest storage_provider must be production-grade while "
                            "--require-production-external is set"
                        )
                    if not transport_metadata.get("public_base_url"):
                        ok = False
                        print(
                            "transport-manifest public_base_url missing while --require-production-external is set"
                        )
                    else:
                        base = str(transport_metadata.get("public_base_url", "")).strip().lower()
                        if (
                            base.startswith("http://localhost")
                            or base.startswith("https://localhost")
                            or base.startswith("http://127.0.0.1")
                            or base.startswith("https://127.0.0.1")
                            or base.startswith("http://0.0.0.0")
                            or base.startswith("https://0.0.0.0")
                            or base.startswith("http://[::1]")
                            or base.startswith("https://[::1]")
                        ):
                            ok = False
                            print(
                                "transport-manifest public_base_url must be non-local while --require-production-external is set"
                            )
                artifacts = transport_payload.get("artifacts")
                if not isinstance(artifacts, list) or not artifacts:
                    ok = False
                    print("transport-manifest artifacts missing when --require-production-external is set")
                else:
                    for artifact in artifacts:
                        if not isinstance(artifact, dict):
                            ok = False
                            print("transport-manifest artifact entry is not a dict")
                            continue
                        if not artifact.get("schema_version"):
                            ok = False
                            print(
                                f"transport-manifest artifact {artifact.get('kind', 'unknown')} "
                                "missing schema_version while production mode is required"
                            )
                        if not artifact.get("external_uri"):
                            ok = False
                            print(
                                f"transport-manifest artifact {artifact.get('kind', 'unknown')} "
                                "missing external_uri while production mode is required"
                            )

    if ok:
        print(
            f"PATH_B_MAX Gate5 API consumer OK: "
            f"tenant={args.tenant} channel={args.channel} matched={len(expected)}"
        )
        return 0

    print(f"release paths processed (verified sha): {released}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
