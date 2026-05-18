import pytest
import torch
from torch import nn

from agim.model.patch_artifact import PatchArtifact, RowPatch
from agim.model.patch_service import PatchService


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(4, 3)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Backbone()
        self.lm_head = nn.Linear(3, 5, bias=False)


def _artifact(patch_id: str, row_id: int, after: torch.Tensor) -> PatchArtifact:
    return PatchArtifact(
        patch_id=patch_id,
        base_model_digest="sha256:model",
        method_profile_id="single_loc",
        subject="Alice",
        relation_id="P17",
        target_new="Paris",
        rows=[RowPatch.from_tensors("lm_head", row_id, torch.zeros(3), after)],
    )


def test_patch_service_lifecycle_applies_and_rolls_back():
    model = _TinyModel()
    original = model.lm_head.weight.detach().clone()
    service = PatchService()
    service.propose_patch(_artifact("p1", 1, torch.tensor([1.0, 0.0, 0.0])))

    with pytest.raises(ValueError):
        service.approve_patch("p1", "alice")

    assert service.run_canaries("p1", {"rewrite": True})["passed"] is True
    assert service.approve_patch("p1", "alice")["status"] == "approved"
    assert service.apply_patch("p1", model)["status"] == "applied"
    assert torch.allclose(model.lm_head.weight[1], torch.tensor([1.0, 0.0, 0.0]))

    assert service.rollback_patch("p1", model)["status"] == "rolled_back"
    assert torch.allclose(model.lm_head.weight, original)


def test_patch_service_simulates_and_diffs_conflicts():
    service = PatchService()
    service.propose_patch(_artifact("p1", 1, torch.tensor([1.0, 0.0, 0.0])))
    service.propose_patch(_artifact("p2", 1, torch.tensor([0.0, 1.0, 0.0])))

    diff = service.diff_patch("p1", "p2")
    sim = service.simulate_patch("p1")

    assert diff["has_conflict"] is True
    assert diff["risk_flags"] == ["row_overlap", "same_relation", "same_subject"]
    assert sim["row_counts"] == {"lm_head": 1}


def test_patch_service_rejects_failed_canaries_before_approval():
    service = PatchService()
    service.propose_patch(_artifact("p1", 1, torch.ones(3)))
    service.run_canaries("p1", {"rewrite": True, "locality": False})

    with pytest.raises(ValueError):
        service.approve_patch("p1", "alice")


class _FakeEditor:
    def __init__(self, model):
        self.model = model
        self.calls = []

    def apply_edit(self, **kwargs):
        self.calls.append(kwargs)
        before = self.model.lm_head.weight.data[1, :].detach().clone()
        self.model.lm_head.weight.data[1, :] = torch.tensor([2.0, 3.0, 4.0])
        return {
            "lm_backup": {1: before},
            "metadata": {"target_token_ids": [1]},
        }

    def rollback(self, backup):
        for row_id, before in backup["lm_backup"].items():
            self.model.lm_head.weight.data[row_id, :] = before


def test_patch_service_materializes_requested_rewrite_draft_then_applies():
    model = _TinyModel()
    before = model.lm_head.weight.detach().clone()
    service = PatchService()
    draft = PatchArtifact(
        patch_id="draft-1",
        base_model_digest="sha256:model",
        method_profile_id="raw_text_proposal",
        subject="France",
        relation_id="P36",
        target_new="Berlin",
        metadata={
            "requires_backend_materialization": True,
            "requested_rewrite": {
                "subject": "France",
                "prompt": "The capital of {} is",
                "relation_id": "P36",
                "target_new": {"str": "Berlin"},
                "target_true": {"str": "Paris"},
            },
        },
    )
    editor = _FakeEditor(model)

    service.propose_patch(draft)
    materialized = service.materialize_patch("draft-1", editor)

    assert materialized["status"] == "materialized"
    assert materialized["row_counts"] == {"lm_head": 1}
    assert torch.allclose(model.lm_head.weight, before)
    assert editor.calls[0]["prompt"] == "The capital of France is"

    service.run_canaries("draft-1", {"rewrite": True})
    service.approve_patch("draft-1", "alice")
    service.apply_patch("draft-1", model)

    assert torch.allclose(model.lm_head.weight[1], torch.tensor([2.0, 3.0, 4.0]))
