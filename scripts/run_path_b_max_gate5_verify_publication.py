#!/usr/bin/env python3
"""Run a full Gate 5 public publication verification flow."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full PATH_B_MAX Gate 5 public publication verification",
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
    parser.add_argument("--tenant", default="public")
    parser.add_argument("--channel", default="public")
    parser.add_argument("--api-base", default=None, help="Optional API base for external consumer checks")
    parser.add_argument(
        "--expected-release-schema-version",
        default="path_b_max_gate5_release.v1",
        help="Expected release schema version for API consumer checks",
    )
    parser.add_argument("--expected-release-sha256", default=None)
    parser.add_argument(
        "--expected-receipt-schema-version",
        default="path_b_max_gate5_receipt.v1",
        help="Expected receipt schema version for API consumer checks",
    )
    parser.add_argument(
        "--expected-bundle-schema-version",
        default="path_b_max_gate5_bundle.v1",
        help="Expected bundle schema version for API consumer checks",
    )
    parser.add_argument("--expected-receipt-sha256", default=None)
    parser.add_argument("--expected-bundle-sha256", default=None)
    parser.add_argument("--expected-transport-manifest-sha256", default=None)
    parser.add_argument("--transport-manifest-path", type=Path, default=None)
    parser.add_argument(
        "--expected-transport-manifest-schema-version",
        default="path_b_max_gate5_transport_manifest.v1",
        help="Expected transport manifest schema version for API checks",
    )
    parser.add_argument(
        "--check-transport-manifest",
        action="store_true",
        help="Verify transport manifest through API consumer when --api-base is set",
    )
    parser.add_argument(
        "--require-production-external",
        action="store_true",
        help=(
            "Require production-grade transport guarantees in manifest "
            "(all external URIs and non-local immutability metadata)."
        ),
    )
    return parser.parse_args()


def _run(cmd: List[str]) -> int:
    return subprocess.run(cmd, check=False).returncode


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"{path} does not contain a JSON object")


def _release_proof_sha(release_payload: Dict[str, Any]) -> Optional[str]:
    proof_sha = release_payload.get("proof_sha256")
    if proof_sha is not None:
        return proof_sha
    proof = release_payload.get("proof")
    if isinstance(proof, dict):
        return proof.get("proof_sha256")
    return None


def _publication_consistency_errors(
    index_path: Path,
    release_path: Path,
    tenant: str,
    channel: str,
    expected_release_sha256: Optional[str],
) -> List[str]:
    errors: List[str] = []

    try:
        index_payload = _load_json(index_path)
        release_payload = _load_json(release_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [str(exc)]

    records = index_payload.get("records")
    if not isinstance(records, list):
        return ["index payload has no records list"]

    if expected_release_sha256 is None:
        return ["expected release hash is required for local publication consistency checks"]

    matching_records = [
        r
        for r in records
        if isinstance(r, dict)
        and r.get("tenant_scope") == tenant
        and r.get("channel") == channel
        and r.get("release_sha256") == expected_release_sha256
    ]
    if not matching_records:
        return [
            "no index records match tenant/channel/release_sha256="
            f"{tenant}/{channel}/{expected_release_sha256}"
        ]

    if not isinstance(release_payload.get("claim_lock"), dict):
        errors.append("release payload missing claim_lock")

    proof_sha = _release_proof_sha(release_payload)
    if proof_sha is None:
        errors.append("release payload missing proof_sha256")

    for rec in matching_records:
        expected_fields = ("proof_sha256", "claim_chain_sha256", "claim_chain_len")
        for field in expected_fields:
            if rec.get(field) is None:
                errors.append(f"index record for release {expected_release_sha256} missing {field}")

        claim_lock = release_payload.get("claim_lock")
        if rec.get("claim_chain_sha256") is not None and isinstance(claim_lock, dict):
            if claim_lock.get("audit_chain_sha256") != rec.get("claim_chain_sha256"):
                errors.append(
                    f"claim_chain_sha256 mismatch for release {expected_release_sha256}: "
                    f"index={rec.get('claim_chain_sha256')} release={claim_lock.get('audit_chain_sha256')}"
                )
        if rec.get("claim_chain_len") is not None and isinstance(claim_lock, dict):
            if claim_lock.get("audit_chain_len") != rec.get("claim_chain_len"):
                errors.append(
                    f"claim_chain_len mismatch for release {expected_release_sha256}: "
                    f"index={rec.get('claim_chain_len')} release={claim_lock.get('audit_chain_len')}"
                )

        if rec.get("proof_sha256") is not None and proof_sha is not None:
            if proof_sha != rec.get("proof_sha256"):
                errors.append(
                    f"proof_sha256 mismatch for release {expected_release_sha256}: "
                    f"index={rec.get('proof_sha256')} release={proof_sha}"
                )

    return errors


def main() -> int:
    args = parse_args()
    expected_transport_manifest_sha256 = args.expected_transport_manifest_sha256
    if expected_transport_manifest_sha256 is None and args.transport_manifest_path is not None:
        try:
            expected_transport_manifest_sha256 = _sha256_file(args.transport_manifest_path)
        except OSError as exc:
            print(f"cannot read transport manifest to derive hash: {exc}")
            return 1

    expected_release_sha256 = args.expected_release_sha256
    if expected_release_sha256 is None:
        try:
            expected_release_sha256 = _sha256_file(args.release_path)
        except OSError as exc:
            print(f"cannot read release to derive hash: {exc}")
            return 1

    checks = [
        [
            sys.executable,
            "scripts/run_path_b_max_verify_gate5_release.py",
            "--tenant",
            args.tenant,
            "--release-path",
            str(args.release_path),
        ],
        [
            sys.executable,
            "scripts/run_path_b_max_verify_gate5_index.py",
            "--tenant",
            args.tenant,
            "--channel",
            args.channel,
            "--index-path",
            str(args.index_path),
        ],
        [
            sys.executable,
            "scripts/run_path_b_max_verify_gate5_receipt.py",
            "--receipt-path",
            str(args.receipt_path),
        ],
        [
            sys.executable,
            "scripts/run_path_b_max_verify_gate5_bundle.py",
            "--bundle-path",
            str(args.bundle_path),
        ],
    ]
    if args.transport_manifest_path is not None:
        transport_check = [
            sys.executable,
            "scripts/run_path_b_max_verify_gate5_transport_manifest.py",
            "--manifest-path",
            str(args.transport_manifest_path),
        ]
        if args.require_production_external:
            transport_check.append("--require-production-external")
        checks.append(transport_check)

    if args.api_base is not None:
        if args.expected_receipt_sha256 is None:
            try:
                args.expected_receipt_sha256 = _sha256_file(args.receipt_path)
            except OSError as exc:
                print(f"cannot read receipt to derive hash: {exc}")
                return 1
        if args.expected_bundle_sha256 is None:
            try:
                args.expected_bundle_sha256 = _sha256_file(args.bundle_path)
            except OSError as exc:
                print(f"cannot read bundle to derive hash: {exc}")
                return 1

    ok = True
    for cmd in checks:
        if _run(cmd) != 0:
            print(f"FAILED: {' '.join(cmd)}")
            ok = False

    for issue in _publication_consistency_errors(
        args.index_path,
        args.release_path,
        args.tenant,
        args.channel,
        expected_release_sha256,
    ):
        print(f"FAILED publication consistency: {issue}")
        ok = False

    if args.api_base is not None:
        check_transport_manifest = args.check_transport_manifest or args.transport_manifest_path is not None
        cmd = [
            sys.executable,
            "scripts/run_path_b_max_gate5_audit_consumer.py",
            "--api-base",
            args.api_base,
            "--tenant",
            args.tenant,
            "--channel",
            args.channel,
            "--check-receipt",
            "--check-bundle",
        ]
        if args.expected_receipt_sha256:
            cmd.extend(["--expected-receipt-sha256", args.expected_receipt_sha256])
        if args.expected_receipt_schema_version:
            cmd.extend(["--expected-receipt-schema-version", args.expected_receipt_schema_version])
        if expected_release_sha256 is not None:
            cmd.extend(["--expected-release-sha256", expected_release_sha256])
        if args.expected_bundle_sha256:
            cmd.extend(["--expected-bundle-sha256", args.expected_bundle_sha256])
        if args.expected_release_schema_version:
            cmd.extend(
                ["--expected-release-schema-version", args.expected_release_schema_version]
            )
        if args.expected_bundle_schema_version:
            cmd.extend(["--expected-bundle-schema-version", args.expected_bundle_schema_version])
        if args.transport_manifest_path is not None and check_transport_manifest:
            cmd.append("--check-transport-manifest")
            if args.expected_transport_manifest_schema_version:
                cmd.extend(
                    [
                        "--expected-transport-manifest-schema-version",
                        args.expected_transport_manifest_schema_version,
                    ]
                )
            if expected_transport_manifest_sha256 is not None:
                cmd.extend(
                    ["--expected-transport-manifest-sha256", expected_transport_manifest_sha256]
                )
            if args.require_production_external:
                cmd.append("--require-production-external")
        if _run(cmd) != 0:
            print(f"FAILED: {' '.join(cmd)}")
            ok = False

    if ok:
        print("PATH_B_MAX Gate5 publication verify OK")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
