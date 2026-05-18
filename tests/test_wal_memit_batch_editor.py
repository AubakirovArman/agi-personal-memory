import torch
from torch import nn

from agim.model.patch_artifact import PatchArtifact, RowPatch
from agim.model.wal_memit_batch_editor import WALMemitBatchEditor


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(3, 3, bias=False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _MLP()


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(4, 3)
        self.layers = nn.ModuleList([_Layer()])


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Backbone()
        self.lm_head = nn.Linear(3, 5, bias=False)


def _patch(patch_id: str, layer: str, row_id: int,
           before: torch.Tensor, after: torch.Tensor) -> PatchArtifact:
    return PatchArtifact(
        patch_id=patch_id,
        base_model_digest="sha256:model",
        method_profile_id="single_loc",
        subject=f"subject-{patch_id}",
        relation_id="P17",
        target_new=f"target-{patch_id}",
        rows=[RowPatch.from_tensors(layer, row_id, before, after)],
    )


def test_wal_memit_batch_consolidates_overlapping_row_deltas():
    first = _patch(
        "a", "lm_head", 1, torch.zeros(3), torch.tensor([1.0, 0.0, 0.0]))
    second = _patch(
        "b", "lm_head", 1, torch.zeros(3), torch.tensor([0.0, 2.0, 0.0]))
    editor = WALMemitBatchEditor()
    editor.extend([first, second])

    artifact = editor.consolidate("batch-1")

    assert artifact.patch_id == "batch-1"
    assert artifact.method_profile_id == "wal_memit_batch"
    assert artifact.row_counts() == {"lm_head": 1}
    assert artifact.metadata["source_patch_ids"] == ["a", "b"]
    assert artifact.metadata["row_conflicts"] == [
        {"layer": "lm_head", "row_id": 1, "count": 2}
    ]
    assert torch.allclose(torch.tensor(artifact.rows[0].after),
                          torch.tensor([1.0, 2.0, 0.0]))


def test_wal_memit_batch_applies_and_rolls_back_rows():
    model = _TinyModel()
    before_lm = model.lm_head.weight.detach().clone()
    before_ffn = model.model.layers[0].mlp.down_proj.weight.detach().clone()
    lm_patch = _patch(
        "lm", "lm_head", 2, torch.zeros(3), torch.tensor([3.0, 0.0, 0.0]))
    ffn_patch = _patch(
        "ffn", "model.layers.0.mlp.down_proj", 1,
        torch.zeros(3), torch.tensor([0.0, 4.0, 0.0]))
    editor = WALMemitBatchEditor()
    editor.extend([lm_patch, ffn_patch])
    artifact = editor.consolidate("batch-apply")

    backup = editor.apply(model, artifact)

    assert torch.allclose(model.lm_head.weight[2], torch.tensor([3.0, 0.0, 0.0]))
    assert torch.allclose(
        model.model.layers[0].mlp.down_proj.weight[1],
        torch.tensor([0.0, 4.0, 0.0]),
    )
    assert set(backup) == {"lm_backup", "emb_backup", "ffn_backup"}

    editor.rollback(model, backup)

    assert torch.allclose(model.lm_head.weight, before_lm)
    assert torch.allclose(model.model.layers[0].mlp.down_proj.weight, before_ffn)
