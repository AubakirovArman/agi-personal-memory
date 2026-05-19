#!/usr/bin/env python
"""Generate a compact, auditable PatchService/Governance proof packet."""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import torch
from torch import nn

from agim.model.patch_artifact import (
    NormBudgetPolicy,
    PatchArtifact,
    RowPatch,
    conflict_summary,
)
from agim.model.patch_governance import PatchGovernance
from agim.model.patch_service import PatchService


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(4, 3)
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.down_proj = nn.Linear(3, 3, bias=False)
        self.model.layers = nn.ModuleList([layer])
        self.lm_head = nn.Linear(3, 5, bias=False)


def _artifact(
    patch_id: str,
    row_id: int,
    after: torch.Tensor,
    *,
    subject: str | None = None,
    metadata: dict[str, object] | None = None,
) -> PatchArtifact:
    patch_subject = subject if subject is not None else ("France" if "1" in patch_id else "Germany")
    metadata_payload = dict(metadata or {})
    metadata_payload.setdefault("relation_shard", "P36")
    metadata_payload.setdefault("relation_slot_id", "P36:slot-0000")
    metadata_payload.setdefault("relation_slot_buckets", 4)
    return PatchArtifact(
        patch_id=patch_id,
        base_model_digest="sha256:path-b-max-proof",
        method_profile_id="path_b_max_gate5",
        subject=patch_subject,
        relation_id="P36",
        target_new="Paris" if patch_id.endswith("1") else "Berlin",
        rows=[RowPatch.from_tensors("lm_head", row_id, torch.zeros(3), after)],
        metadata=metadata_payload,
    )


def _conflict_metadata() -> dict[str, object]:
    return {
        "subject_token_ids": [101, 102],
        "target_token_ids": [201],
        "control_row_ids": [2],
        "protected_basis_ids": ["P36:protected:0"],
        "relation_shard": "P36",
        "relation_slot_id": "P36:slot-0000",
        "relation_slot_buckets": 4,
    }


def _blocked_probe(action) -> dict[str, object]:
    try:
        action()
    except ValueError as exc:
        return {
            "blocked": True,
            "error": str(exc),
        }
    return {"blocked": False}


def main() -> int:
    output = Path("results/easyedit_official/governance/path_b_max_gate5_proof.json")
    model = _TinyModel()
    service = PatchService(
        enforce_conflicts=True,
        enforce_budget=True,
        budget_policy=NormBudgetPolicy(max_rows=2, max_row_delta_norm=2.0),
    )
    shared_budget_service = PatchService(
        enforce_conflicts=False,
        enforce_budget=True,
        budget_policy=NormBudgetPolicy(
            max_rows=2,
            max_row_delta_norm=2.0,
            max_shared_row_delta_norm=1.0,
        ),
    )
    governance = PatchGovernance()
    output.parent.mkdir(parents=True, exist_ok=True)

    governance.grant("proof-1", "alice", "approve")
    governance.grant("proof-1", "alice", "apply")
    governance.grant("proof-1", "alice", "rollback")

    a = _artifact(
        "proof-1",
        1,
        torch.tensor([1.0, 0.0, 0.0]),
        metadata=_conflict_metadata(),
    )
    b = _artifact("proof-2", 2, torch.tensor([0.0, 1.0, 0.0]))

    packet: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "proof_id": "path_b_max_gate_5",
        "lifecycle": {},
        "governance": {},
    }

    packet["lifecycle"]["propose"] = service.propose_patch(a)
    packet["lifecycle"]["inspect_after_propose"] = service.inspect_patch("proof-1")
    packet["lifecycle"]["simulate"] = service.simulate_patch("proof-1")
    packet["lifecycle"]["run_canaries"] = service.run_canaries("proof-1", {"rewrite": True})
    packet["lifecycle"]["approve"] = service.approve_patch("proof-1", "alice")
    packet["lifecycle"]["apply"] = service.apply_patch("proof-1", model)
    packet["lifecycle"]["inspect_after_apply"] = service.inspect_patch("proof-1")

    conflict_patch = _artifact(
        "proof-conflict-1",
        1,
        torch.tensor([0.5, 0.0, 0.0]),
        subject="France",
        metadata=_conflict_metadata(),
    )
    packet["lifecycle"]["strict_conflict_summary"] = conflict_summary(a, conflict_patch)
    packet["lifecycle"]["strict_conflict_guard"] = _blocked_probe(
        lambda: service.propose_patch(conflict_patch)
    )

    packet["lifecycle"]["rollback"] = service.rollback_patch("proof-1", model)
    packet["lifecycle"]["inspect_after_rollback"] = service.inspect_patch("proof-1")

    packet["lifecycle"]["strict_budget_guard"] = _blocked_probe(
        lambda: service.propose_patch(
            _artifact("proof-budget", 3, torch.tensor([3.0, 0.0, 0.0]))
        )
    )
    shared_budget_service.propose_patch(
        _artifact("proof-shared-1", 4, torch.tensor([0.6, 0.0, 0.0]))
    )
    shared_budget_service.run_canaries("proof-shared-1", {"rewrite": True})
    shared_budget_service.approve_patch("proof-shared-1", "alice")
    shared_budget_service.apply_patch("proof-shared-1", model)
    packet["lifecycle"]["shared_row_budget_guard"] = _blocked_probe(
        lambda: shared_budget_service.propose_patch(
            _artifact("proof-shared-2", 4, torch.tensor([0.6, 0.0, 0.0]))
        )
    )
    packet["lifecycle"]["shared_row_budget_inspect"] = shared_budget_service.inspect_patch(
        "proof-shared-1"
    )
    shared_budget_service.rollback_patch("proof-shared-1", model)
    packet["lifecycle"]["shared_row_budget_after_rollback"] = (
        shared_budget_service.inspect_patch("proof-shared-1")
    )
    shared_budget_service.propose_patch(
        _artifact("proof-shared-2", 4, torch.tensor([0.6, 0.0, 0.0]))
    )
    shared_budget_service.run_canaries("proof-shared-2", {"rewrite": True})
    shared_budget_service.approve_patch("proof-shared-2", "alice")
    shared_budget_service.apply_patch("proof-shared-2", model)
    packet["lifecycle"]["shared_row_budget_after_reapply"] = (
        shared_budget_service.inspect_patch("proof-shared-2")
    )

    service.propose_patch(b)
    packet["lifecycle"]["diff"] = service.diff_patch("proof-1", "proof-2")
    packet["lifecycle"]["list_patches"] = service.list_patches()

    signature = governance.sign_artifact(a, "alice")
    packet["governance"]["signature"] = signature
    packet["governance"]["verify_signature"] = governance.verify_signature(a)
    packet["governance"]["access_alice_approve"] = governance.check_access(
        "proof-1", "alice", "approve"
    )
    packet["governance"]["events"] = [
        governance.audit("propose", "proof-1", "alice", {"patch_id": "proof-1"}).__dict__,
        governance.audit("simulate", "proof-1", "alice", {"status": packet["lifecycle"]["simulate"]["status"]}).__dict__,
        governance.audit("run_canaries", "proof-1", "alice", {"canaries": packet["lifecycle"]["run_canaries"]["canaries"]}).__dict__,
        governance.audit("approve", "proof-1", "alice", {"status": packet["lifecycle"]["approve"]["status"]}).__dict__,
        governance.audit("apply", "proof-1", "alice", {"status": packet["lifecycle"]["apply"]["status"]}).__dict__,
        governance.audit("rollback", "proof-1", "alice", {"status": packet["lifecycle"]["rollback"]["status"]}).__dict__,
        governance.audit("inspect", "proof-1", "alice", {"status": packet["lifecycle"]["inspect_after_rollback"]["status"]}).__dict__,
        governance.audit("diff", "proof-1", "alice", {"compared_with": "proof-2"}).__dict__,
    ]
    packet["governance"]["audit_chain_valid"] = governance.verify_audit_chain()
    packet["governance"]["audit_trail"] = governance.audit_trail()

    output.write_text(json.dumps(packet, indent=2, ensure_ascii=False))
    print(f"PatchService/Governance proof packet saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
