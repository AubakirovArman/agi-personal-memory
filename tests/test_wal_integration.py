"""Integration tests: AGIM + real WAL encoder."""
import tempfile

import torch

from agim.core.system import AGIMSystem
from agim.model.wal_backend import WALWeightEditor
from agim.model.memory_overlay import MemoryOverlay


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)

    def forward(self, x):
        return self.linear(x)


def test_wal_vocabulary_build():
    model = SimpleModel()
    editor = WALWeightEditor(model, K=64, lmax=5, device="cpu")
    editor.build_vocabulary("linear.weight")
    assert editor.vocabulary_is_frozen
    assert editor._atom_table is not None
    assert editor._atom_table.shape[0] == 64


def test_wal_encode_weight():
    model = SimpleModel()
    editor = WALWeightEditor(model, K=64, lmax=5, device="cpu")
    editor.build_vocabulary("linear.weight")
    weight = model.linear.weight.data.clone()
    prog, recon = editor.encode_weight(weight)
    assert prog.N == weight.numel()
    assert recon.shape == weight.reshape(-1).shape


def test_wal_edit_and_verify():
    model = SimpleModel()
    editor = WALWeightEditor(model, K=64, lmax=5, device="cpu")
    editor.build_vocabulary("linear.weight")

    original = model.linear.weight.data.clone()
    delta = torch.randn_like(original) * 0.01
    editor.snapshot_layer("linear.weight")

    ok = editor.edit_weight("linear.weight", delta)
    assert ok
    assert editor.edit_count == 1

    assert not torch.equal(model.linear.weight.data, original)

    ok = editor.rollback_edit("linear.weight")
    assert ok
    assert torch.equal(model.linear.weight.data, original)


def test_memory_overlay_with_agim():
    with tempfile.TemporaryDirectory() as tmp:
        agim = AGIMSystem(workdir=tmp)
        overlay = MemoryOverlay(agim)
        assert not overlay.model_available

        overlay.teach("X?", "Y")
        resp = overlay.ask("X?")
        assert resp.answer == "Y"
        assert resp.source == "wal_recipe"

        resp2 = overlay.ask("Unknown?")
        assert resp2.source == "none"


def test_full_wal_agim_cycle():
    """Test: AGIM memory → WAL encode → model edit → verify → rollback."""
    model = SimpleModel()
    editor = WALWeightEditor(model, K=64, lmax=5, device="cpu")
    editor.build_vocabulary("linear.weight")

    original = model.linear.weight.data.clone()
    delta = torch.randn_like(original) * 0.01

    ok = editor.edit_weight("linear.weight", delta)
    assert ok
    assert not torch.equal(model.linear.weight.data, original)

    ok = editor.rollback_edit("linear.weight")
    assert ok
    assert torch.equal(model.linear.weight.data, original)
