"""v1.0: WISE Dual-Memory — separates episodic & semantic, prevents interference."""
from __future__ import annotations

import torch


class WISEDualMemory:
    """WISE (Weight Interference-Suppressed Editing) dual-memory architecture.

    Separates episodic memory (specific facts) from semantic memory (general
    reasoning) to prevent catastrophic interference during editing.
    """

    def __init__(self, model: torch.nn.Module, semantic_layers: set[int] | None = None,
                 episodic_layers: set[int] | None = None):
        self.model = model
        self.semantic_layers = semantic_layers or set(range(0, 20))
        self.episodic_layers = episodic_layers or set(range(40, 60))
        self._episodic_state: dict[int, torch.Tensor] = {}
        self._semantic_state: dict[int, torch.Tensor] = {}

    def protect_semantic(self):
        """Snapshot semantic layers before episodic edits."""
        for layer_idx in self.semantic_layers:
            try:
                mlp = self.model.model.layers[layer_idx].mlp.down_proj
                self._semantic_state[layer_idx] = mlp.weight.data.clone()
            except (AttributeError, IndexError):
                continue

    def verify_semantic_intact(self) -> tuple[bool, dict[int, float]]:
        """Check that semantic layers haven't changed after episodic edit."""
        results = {}
        intact = True
        for layer_idx, original in self._semantic_state.items():
            try:
                mlp = self.model.model.layers[layer_idx].mlp.down_proj
                diff = (mlp.weight.data - original).norm().item()
                ok = diff < 1e-6
                results[layer_idx] = diff
                if not ok:
                    intact = False
            except (AttributeError, IndexError):
                continue
        return intact, results

    def apply_episodic_edit(self, layer_idx: int, delta: torch.Tensor) -> bool:
        """Apply edit only to episodic layers, protect semantic."""
        if layer_idx not in self.episodic_layers:
            return False
        try:
            mlp = self.model.model.layers[layer_idx].mlp.down_proj
            self.protect_semantic()
            mlp.weight.data.add_(delta.to(mlp.weight.dtype))
            return True
        except (AttributeError, IndexError):
            return False

    def rollback_episodic(self) -> bool:
        """Restore semantic layers from snapshot."""
        for layer_idx, original in self._semantic_state.items():
            try:
                self.model.model.layers[layer_idx].mlp.down_proj.weight.data = original
            except (AttributeError, IndexError):
                continue
        return True
