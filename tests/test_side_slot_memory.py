import torch

from agim.model.patch_artifact import PatchArtifact, RowPatch
from agim.model.side_slot_memory import SideSlotMemory


class _Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(4, 3)


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Backbone()
        self.lm_head = torch.nn.Linear(3, 5, bias=False)


def _artifact(patch_id: str, subject: str, relation_id: str,
              row_id: int, delta: torch.Tensor) -> PatchArtifact:
    return PatchArtifact(
        patch_id=patch_id,
        base_model_digest="sha256:model",
        method_profile_id="single_loc",
        subject=subject,
        relation_id=relation_id,
        target_new="Paris",
        rows=[RowPatch.from_tensors("lm_head", row_id, torch.zeros(3), delta)],
    )


def test_side_slot_memory_selects_by_subject_and_relation():
    memory = SideSlotMemory()
    memory.add_patch(_artifact("a", "Alice", "P17", 1, torch.ones(3)))
    memory.add_patch(_artifact("b", "Bob", "P19", 2, torch.ones(3)))

    selected = memory.select(subject="Alice", relation_id="P17")

    assert [slot.slot_id for slot in selected] == ["a"]
    assert memory.summary() == {"slots": 2, "enabled": 2}
    assert memory.relation_slot_summary() == {
        "P17": {"slots": 1, "enabled": 1},
        "P19": {"slots": 1, "enabled": 1},
    }


def test_side_slot_memory_builds_overlay_for_selected_slots():
    model = _TinyModel()
    hidden = torch.tensor([[[1.0, 0.0, 0.0]]])
    baseline = model.lm_head(hidden)
    memory = SideSlotMemory()
    memory.add_patch(_artifact("a", "Alice", "P17", 1, torch.tensor([1.0, 0.0, 0.0])))
    memory.add_patch(_artifact("b", "Bob", "P19", 2, torch.tensor([1.0, 0.0, 0.0])))

    with memory.overlay_for(model, subject="Alice", relation_id="P17") as overlay:
        patched = model.lm_head(hidden)

    assert overlay.lm_deltas.keys() == {1}
    assert patched[..., 1] == baseline[..., 1] + 1.0
    assert patched[..., 2] == baseline[..., 2]


def test_side_slot_memory_disable_excludes_slot_from_overlay():
    model = _TinyModel()
    memory = SideSlotMemory()
    memory.add_patch(_artifact("a", "Alice", "P17", 1, torch.ones(3)))
    memory.disable("a")

    overlay = memory.overlay_for(model, subject="Alice", relation_id="P17")

    assert overlay.lm_deltas == {}
    assert memory.summary() == {"slots": 1, "enabled": 0}


def test_side_slot_memory_isolates_relation_slots():
    model = _TinyModel()
    hidden = torch.tensor([[[1.0, 0.0, 0.0]]])
    baseline = model.lm_head(hidden)
    memory = SideSlotMemory()
    memory.add_patch(_artifact(
        "a", "Alice", "P17", 1, torch.tensor([1.0, 0.0, 0.0])))
    memory.add_patch(_artifact(
        "b", "Alice", "P19", 2, torch.tensor([1.0, 0.0, 0.0])))

    with memory.overlay_for(model, subject="Alice", relation_slot_id="P17") as overlay:
        patched = model.lm_head(hidden)

    assert overlay.lm_deltas.keys() == {1}
    assert patched[..., 1] == baseline[..., 1] + 1.0
    assert patched[..., 2] == baseline[..., 2]


def test_side_slot_memory_can_disable_whole_relation_slot():
    memory = SideSlotMemory()
    memory.add_patch(_artifact("a", "Alice", "P17", 1, torch.ones(3)))
    memory.add_patch(_artifact("b", "Bob", "P17", 2, torch.ones(3)))
    memory.add_patch(_artifact("c", "Alice", "P19", 3, torch.ones(3)))

    memory.disable_relation_slot("P17")

    assert memory.select(relation_slot_id="P17") == []
    assert [slot.slot_id for slot in memory.select(relation_slot_id="P19")] == ["c"]
    assert memory.relation_slot_summary()["P17"] == {"slots": 2, "enabled": 0}
