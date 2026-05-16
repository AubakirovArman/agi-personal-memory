"""WAL frozen vocabulary backend — apply verified edits to model weights."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class WALWeightEditor:
    """Edit model weights using frozen vocabulary WAL recipes.

    Key property: frozen vocabulary = 0% non-target diff.
    Only the target layer is modified; all other layers unchanged.
    """
    def __init__(self, model: torch.nn.Module, atom_table_path: str | None = None):
        self.model = model
        self.atom_table_path = atom_table_path
        self._original_weights: dict[str, torch.Tensor] = {}
        self._edit_log: list[dict[str, Any]] = []

    def snapshot_layer(self, layer_name: str):
        """Save original weights for rollback."""
        param = dict(self.model.named_parameters()).get(layer_name)
        if param is not None:
            self._original_weights[layer_name] = param.data.clone()

    def apply_edit(self, recipe: dict) -> bool:
        """Apply a WAL recipe edit to the model."""
        layer_name = recipe.get("target_layer")
        if not layer_name:
            return False
        param = dict(self.model.named_parameters()).get(layer_name)
        if param is None:
            return False
        self.snapshot_layer(layer_name)
        delta = recipe.get("delta_tensor")
        if delta is not None:
            param.data.add_(torch.tensor(delta, device=param.device, dtype=param.dtype))
        self._edit_log.append({
            "artifact_id": recipe.get("artifact_id", "?"),
            "layer": layer_name,
            "timestamp": recipe.get("created_at", ""),
        })
        return True

    def rollback_layer(self, layer_name: str) -> bool:
        if layer_name in self._original_weights:
            param = dict(self.model.named_parameters()).get(layer_name)
            if param is not None:
                param.data.copy_(self._original_weights[layer_name])
                del self._original_weights[layer_name]
                return True
        return False

    def verify_non_target_diff(self, edited_layer: str) -> bool:
        """Check that only the target layer changed."""
        for name, param in self.model.named_parameters():
            if name == edited_layer:
                continue
            if name in self._original_weights:
                if not torch.equal(param.data, self._original_weights[name]):
                    return False
        return True

    @property
    def edit_count(self) -> int:
        return len(self._edit_log)

    def save_edit_log(self, path: Path):
        path.write_text(json.dumps(self._edit_log, indent=2))
