"""Mutable-state namespaces for WALDualLayerEditor."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class WALDualMutableState:
    edit_key_basis: list[torch.Tensor] = field(default_factory=list)
    relation_key_basis: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    relation_protected_basis: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    edit_count: int = 0


def init_state_namespaces(editor) -> None:
    state = WALDualMutableState(
        edit_key_basis=editor._edit_key_basis,
        relation_key_basis=editor._relation_key_basis,
        relation_protected_basis=editor._relation_protected_basis,
        edit_count=editor._edit_count,
    )
    editor._state_namespaces = {"default": state}
    editor._active_state_namespace = "default"


def activate_state_namespace(editor, namespace: str | None) -> str:
    key = str(namespace or "default")
    states = getattr(editor, "_state_namespaces", None)
    if states is None:
        init_state_namespaces(editor)
        states = editor._state_namespaces
    state = states.setdefault(key, WALDualMutableState())
    editor._active_state_namespace = key
    editor._edit_key_basis = state.edit_key_basis
    editor._relation_key_basis = state.relation_key_basis
    editor._relation_protected_basis = state.relation_protected_basis
    editor._edit_count = state.edit_count
    return key


def sync_active_state(editor) -> None:
    key = getattr(editor, "_active_state_namespace", "default")
    state = editor._state_namespaces[key]
    state.edit_key_basis = editor._edit_key_basis
    state.relation_key_basis = editor._relation_key_basis
    state.relation_protected_basis = editor._relation_protected_basis
    state.edit_count = editor._edit_count


def rollback(editor, backup: dict):
    """Exact rollback via clone restoration."""
    if "state_namespace" in backup:
        editor._activate_state_namespace(backup.get("state_namespace"))
    for tid, orig in backup.get("lm_backup", {}).items():
        editor.model.lm_head.weight.data[tid, :] = orig
    for sid, orig in backup.get("emb_backup", {}).items():
        editor.model.model.embed_tokens.weight.data[sid, :] = orig
    if "history_len" in backup:
        editor._edit_key_basis = editor._edit_key_basis[:backup["history_len"]]
    if "relation_key" in backup and "relation_history_len" in backup:
        key = backup["relation_key"]
        editor._relation_key_basis[key] = editor._relation_key_basis.get(key, [])[
            :backup["relation_history_len"]]
    if "relation_key" in backup and "relation_protected_len" in backup:
        key = backup["relation_key"]
        editor._relation_protected_basis[key] = editor._relation_protected_basis.get(
            key, [])[:backup["relation_protected_len"]]
    sync_active_state(editor)
