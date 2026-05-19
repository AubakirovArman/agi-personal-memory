#!/usr/bin/env python3
"""Verify public gate-5 release packet and its claim-lock integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_chain(chain) -> bytes:
    return json.dumps(chain, sort_keys=True, ensure_ascii=False).encode()


def verify_signature_presence(packet: Dict[str, Any]) -> bool:
    return bool(packet.get("signature"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify public gate-5 release packet",
    )
    parser.add_argument(
        "--release-path",
        type=Path,
        default=Path("results/easyedit_official/governance/path_b_max_gate5_public_release.json"),
    )
    parser.add_argument(
        "--tenant",
        default="public",
        help="Expected tenant_scope in release packet",
    )
    args = parser.parse_args()

    if not args.release_path.exists():
        print(f"Missing release packet: {args.release_path}")
        return 1

    packet = read_json(args.release_path)
    ok = True

    evidence = packet.get("evidence_paths", []) or packet.get("verifier", {}).get("evidence_paths", [])
    if not isinstance(evidence, list) or not evidence:
        print("Missing evidence_paths")
        ok = False

    source_path = packet.get("source_path") or (evidence[0] if evidence else None)
    if not source_path:
        print("Missing source_path")
        ok = False

    source_file = Path(source_path) if source_path else None
    if source_file and not source_file.exists():
        print(f"Missing source proof file: {source_file}")
        ok = False

    if source_file and source_file.exists():
        proof = read_json(source_file)
        proof_sha = sha256_bytes(source_file.read_bytes())
        if proof_sha != packet.get("proof_sha256"):
            print("proof_sha256 mismatch")
            ok = False

        governance = proof.get("governance") if isinstance(proof.get("governance"), dict) else {}
        chain = proof.get("audit_chain", governance.get("audit_trail", []))
        claim = packet.get("claim_lock", {})
        if claim.get("audit_chain_sha256") != sha256_bytes(normalize_chain(chain)):
            print("claim_lock.audit_chain_sha256 mismatch")
            ok = False
        if claim.get("audit_chain_len") != len(chain):
            print("claim_lock.audit_chain_len mismatch")
            ok = False
        if chain:
            chain_head = chain[-1].get("entry_sha256")
            if chain_head is None and isinstance(chain[-1], dict):
                chain_head = chain[-1].get("metadata", {}).get("digest")
                if chain_head is None:
                    chain_head = chain[-1].get("event_hash")
            if claim.get("audit_chain_digest_head") != chain_head:
                print("claim_lock.audit_chain_digest_head mismatch")
                ok = False
        if not verify_signature_presence(packet):
            print("Missing signature")
            ok = False

    if packet.get("release_schema_version") != "path_b_max_gate5_release.v1":
        print(f"Unexpected release_schema_version: {packet.get('release_schema_version')}")
        ok = False

    if packet.get("tenant_scope") != args.tenant:
        print(f"Unexpected tenant_scope: {packet.get('tenant_scope')}, expected: {args.tenant}")
        ok = False

    if not packet.get("base_model_digest") or not packet.get("command"):
        print("Missing base metadata (base_model_digest/command)")
        ok = False

    if ok:
        print(f"PATH_B_MAX Gate5 public release packet OK: {args.release_path}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
