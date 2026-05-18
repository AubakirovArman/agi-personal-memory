import torch

from agim.model.wal_dual_editor import WALDualLayerEditor


def test_wal_dual_state_namespaces_isolate_history_basis():
    editor = WALDualLayerEditor(object(), object(), device="cpu")
    editor._edit_key_basis.append(torch.tensor([1.0, 0.0]))
    editor._sync_active_state()

    editor._activate_state_namespace("tenant-a")
    assert editor._edit_key_basis == []
    editor._edit_key_basis.append(torch.tensor([0.0, 1.0]))
    editor._sync_active_state()

    editor._activate_state_namespace("default")
    assert torch.equal(editor._edit_key_basis[0], torch.tensor([1.0, 0.0]))

    editor._activate_state_namespace("tenant-a")
    assert torch.equal(editor._edit_key_basis[0], torch.tensor([0.0, 1.0]))


def test_wal_dual_rollback_uses_backup_namespace():
    editor = WALDualLayerEditor(object(), object(), device="cpu")
    editor._activate_state_namespace("tenant-a")
    editor._edit_key_basis.append(torch.tensor([1.0, 0.0]))
    editor._relation_key_basis["P17"] = [torch.tensor([0.0, 1.0])]
    editor._relation_protected_basis["P17"] = [torch.tensor([1.0, 1.0])]
    editor._sync_active_state()

    editor._activate_state_namespace("tenant-b")
    editor._edit_key_basis.append(torch.tensor([2.0, 0.0]))
    editor._sync_active_state()

    editor.rollback({
        "state_namespace": "tenant-a",
        "history_len": 0,
        "relation_key": "P17",
        "relation_history_len": 0,
        "relation_protected_len": 0,
    })

    assert editor._active_state_namespace == "tenant-a"
    assert editor._edit_key_basis == []
    assert editor._relation_key_basis["P17"] == []
    assert editor._relation_protected_basis["P17"] == []

    editor._activate_state_namespace("tenant-b")
    assert torch.equal(editor._edit_key_basis[0], torch.tensor([2.0, 0.0]))
