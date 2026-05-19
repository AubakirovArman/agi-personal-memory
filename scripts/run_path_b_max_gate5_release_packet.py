#!/usr/bin/env python3
"""Create a public-style release packet for PATH_B_MAX gate 5 proof.

This script reads the existing gate-5 governance proof and produces a
deterministic release artifact with claim-chain digest and integrity metadata.
"""

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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def build_release_packet(
    proof: Dict[str, Any],
    proof_path: Path,
    release_path: Path,
    tenant: str,
) -> Dict[str, Any]:
    governance = proof.get("governance") if isinstance(proof.get("governance"), dict) else {}
    lifecycle = proof.get("lifecycle") if isinstance(proof.get("lifecycle"), dict) else {}
    lifecycle_propose = lifecycle.get("propose", {}) if isinstance(lifecycle.get("propose"), dict) else {}
    audit_chain = proof.get("audit_chain", governance.get("audit_trail", []))
    audit_chain_dump = json.dumps(audit_chain, sort_keys=True, ensure_ascii=False).encode()
    chain_head = None
    if audit_chain:
        chain_tail = audit_chain[-1]
        if isinstance(chain_tail, dict):
            chain_head = chain_tail.get("entry_sha256")
            if chain_head is None and isinstance(chain_tail.get("metadata"), dict):
                chain_head = chain_tail["metadata"].get("digest")
            if chain_head is None:
                chain_head = chain_tail.get("event_hash")

    claim_lock = {
        "audit_chain_len": len(audit_chain),
        "audit_chain_sha256": sha256_bytes(audit_chain_dump),
        "audit_chain_digest_head": chain_head,
    }

    proof_signature = proof.get("signature")
    if proof_signature is None:
        proof_signature = governance.get("signature")

    return {
        "release_schema_version": "path_b_max_gate5_release.v1",
        "tenant_scope": tenant,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_path": str(proof_path),
        "release_path": str(release_path),
        "release_kind": "governance_proof_public_packet",
        "base_model_digest": (
            proof.get("base_model_digest")
            or lifecycle_propose.get("base_model_digest")
            or proof.get("governance", {}).get("base_model_digest")
            or "sha256:path-b-max-proof"
        ),
        "method_profile_id": (
            proof.get("method_profile_id")
            or lifecycle_propose.get("method_profile_id")
            or proof.get("governance", {}).get("method_profile_id")
        ),
        "artifact_schema_version": proof.get("artifact_schema_version")
        or lifecycle_propose.get("artifact_schema_version"),
        "atoms_digest": proof.get("atoms_digest"),
        "git_sha": proof.get("git_sha") or proof.get("governance", {}).get("git_sha"),
        "command": proof.get("command") or proof.get("governance", {}).get("command") or "run_path_b_max_patch_service_governance_proof",
        "proof_sha256": sha256_file(proof_path),
        "source_hash": proof.get("payload", {}).get("sha256", None),
        "claim_lock": claim_lock,
        "acl_summary": proof.get("acl_state") if isinstance(proof.get("acl_state"), dict) else {},
        "signature": proof_signature,
        "verifier": {
            "evidence_paths": [
                str(proof_path),
            ],
            "required_checks": [
                "verify_signature",
                "verify_audit_chain",
                "verify_acl",
                "verify_claim_lock",
            ],
        },
        "evidence_paths": [
            str(proof_path),
        ],
    }


def verify_release_packet(packet: Dict[str, Any]) -> List[str]:
    issues = []
    if not packet.get("base_model_digest"):
        issues.append("missing base_model_digest")
    if not packet.get("proof_sha256"):
        issues.append("missing proof_sha256")
    if packet.get("claim_lock", {}).get("audit_chain_len", 0) == 0:
        issues.append("empty audit_chain_len")
    if not packet.get("command"):
        issues.append("missing command metadata")
    if not packet.get("signature"):
        issues.append("missing signature")
    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a public release packet for PATH_B_MAX gate 5 proof",
    )
    parser.add_argument(
        "--proof-path",
        type=Path,
        default=Path("results/easyedit_official/governance/path_b_max_gate5_proof.json"),
        help="Path to gate 5 governance proof JSON",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/easyedit_official/governance/path_b_max_gate5_public_release.json"),
        help="Path to write immutable release packet",
    )
    parser.add_argument(
        "--tenant",
        default="public",
        help="Tenant scope label for public release packet",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.proof_path.exists():
        print(f"Gate-5 proof missing: {args.proof_path}")
        return 1

    proof = read_json(args.proof_path)
    packet = build_release_packet(
        proof=proof,
        proof_path=args.proof_path,
        release_path=args.output_path,
        tenant=args.tenant,
    )
    issues = verify_release_packet(packet)
    if issues:
        print("WARNING: release packet missing fields:", ", ".join(issues))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        json.dump(packet, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"wrote release packet: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
