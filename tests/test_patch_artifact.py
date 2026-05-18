import pytest
import torch

from agim.model.patch_artifact import (
    NormBudgetPolicy,
    PatchArtifact,
    RowPatch,
    conflict_summary,
)


def test_patch_artifact_round_trips_and_summarizes_norms():
    row = RowPatch.from_tensors(
        "lm_head",
        7,
        torch.tensor([1.0, 2.0]),
        torch.tensor([2.0, 4.0]),
    )
    artifact = PatchArtifact(
        patch_id="patch-1",
        base_model_digest="sha256:model",
        method_profile_id="single_loc",
        subject="Alice",
        relation_id="P17",
        target_new="Paris",
        target_true="Rome",
        rows=[row],
    )

    restored = PatchArtifact.from_dict(artifact.to_dict())

    assert restored.patch_id == "patch-1"
    assert restored.row_counts() == {"lm_head": 1}
    assert restored.norm_summary()["patch_delta_norm"] == pytest.approx(5 ** 0.5)
    assert restored.norm_summary()["max_delta_norm"] == pytest.approx(5 ** 0.5)


def test_patch_conflict_summary_reports_overlapping_rows():
    left = PatchArtifact(
        patch_id="left",
        base_model_digest="sha256:model",
        method_profile_id="single_loc",
        subject="Alice",
        relation_id="P17",
        target_new="Paris",
        rows=[
            RowPatch.from_tensors("lm_head", 1, torch.zeros(2), torch.ones(2)),
        ],
    )
    right = PatchArtifact(
        patch_id="right",
        base_model_digest="sha256:model",
        method_profile_id="single_loc",
        subject="Alice",
        relation_id="P17",
        target_new="Berlin",
        rows=[
            RowPatch.from_tensors("lm_head", 1, torch.zeros(2), torch.ones(2)),
            RowPatch.from_tensors("embed_tokens", 2, torch.zeros(2), torch.ones(2)),
        ],
    )

    summary = conflict_summary(left, right)

    assert summary["has_conflict"] is True
    assert summary["overlapping_rows"] == [{"layer": "lm_head", "row_id": 1}]
    assert summary["same_relation"] is True
    assert summary["same_subject"] is True


def test_norm_budget_policy_blocks_no_commit_when_limits_are_exceeded():
    artifact = PatchArtifact(
        patch_id="patch-budget",
        base_model_digest="sha256:model",
        method_profile_id="single_loc",
        subject="Alice",
        relation_id="P17",
        target_new="Paris",
        rows=[
            RowPatch.from_tensors("lm_head", 1, torch.zeros(2), torch.ones(2)),
            RowPatch.from_tensors("embed_tokens", 2, torch.zeros(2), torch.ones(2)),
        ],
    )

    decision = artifact.budget_decision(NormBudgetPolicy(max_rows=1))

    assert decision["allow_commit"] is False
    assert decision["no_commit"] is True
    assert decision["reasons"] == [{"metric": "row_count", "value": 2, "limit": 1}]


def test_norm_budget_policy_allows_patch_within_limits():
    artifact = PatchArtifact(
        patch_id="patch-small",
        base_model_digest="sha256:model",
        method_profile_id="single_loc",
        subject="Alice",
        relation_id="P17",
        target_new="Paris",
        rows=[RowPatch.from_tensors("lm_head", 1, torch.zeros(2), torch.ones(2))],
    )

    decision = artifact.budget_decision(
        NormBudgetPolicy(max_rows=2, max_row_delta_norm=2.0))

    assert decision["allow_commit"] is True
    assert decision["reasons"] == []
