import torch

from agim.model.patch_artifact import PatchArtifact, RowPatch
from agim.model.patch_governance import PatchGovernance, artifact_digest


def _artifact() -> PatchArtifact:
    return PatchArtifact(
        patch_id="p1",
        base_model_digest="sha256:model",
        method_profile_id="single_loc",
        subject="Alice",
        relation_id="P17",
        target_new="Paris",
        rows=[RowPatch.from_tensors("lm_head", 1, torch.zeros(2), torch.ones(2))],
    )


def test_patch_governance_signs_and_verifies_artifact():
    artifact = _artifact()
    governance = PatchGovernance(signing_key="secret")

    signature = governance.sign_artifact(artifact, "alice")

    assert signature["signer"] == "alice"
    assert governance.verify_signature(artifact) is True
    assert governance.verify_audit_chain() is True


def test_patch_governance_detects_signature_tampering():
    artifact = _artifact()
    governance = PatchGovernance(signing_key="secret")
    governance.sign_artifact(artifact, "alice")

    artifact.rows[0].after = [2.0, 2.0]

    assert governance.verify_signature(artifact) is False


def test_patch_governance_acl_and_audit_chain():
    governance = PatchGovernance()
    governance.grant("p1", "alice", "approve")

    event = governance.audit("approve", "p1", "alice", {"ok": True})

    assert governance.check_access("p1", "alice", "approve") is True
    assert governance.check_access("p1", "bob", "approve") is False
    assert event.previous_hash == "0" * 64
    assert governance.audit_trail()[0]["action"] == "approve"
    assert governance.verify_audit_chain() is True


def test_artifact_digest_ignores_signature_metadata():
    artifact = _artifact()
    before = artifact_digest(artifact)
    artifact.metadata["signature"] = {"signature": "abc"}

    assert artifact_digest(artifact) == before
