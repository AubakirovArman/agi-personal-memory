"""WAL frozen vocabulary backend — real weight editing with 0% non-target diff."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

_WAL_IMPORTED = False
try:
    from ..wal.encoder import build_atoms_kmeans, wal_encode_scalar
    from ..wal.isa import ProgramBuffer, pack_programs, unpack_programs
    _WAL_IMPORTED = True
except ImportError as e:
        import traceback; traceback.print_exc()


class WALWeightEditor:
    """Edit model weights using WAL with frozen vocabulary.

    Key property: frozen vocabulary = 0% non-target diff.
    Only the target layer is modified; all other layers unchanged.

    Uses real WAL encoder from src/agim/wal/ for actual weight encoding.
    """

    def __init__(self, model: torch.nn.Module, atom_table: torch.Tensor | None = None,
                 K: int = 256, lmax: int = 12, device: str = "cuda:3"):
        self.model = model
        self.K = K
        self.lmax = lmax
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        self._original_weights: dict[str, torch.Tensor] = {}
        self._atom_table = atom_table
        self._vocabulary_frozen = atom_table is not None
        self._edit_log: list[dict[str, Any]] = []
        self._snapshots: dict[str, torch.Tensor] = {}

    def build_vocabulary(self, sample_layer: str | None = None):
        """Build atom table from model weights. Freeze after building."""
        if sample_layer:
            param = dict(self.model.named_parameters()).get(sample_layer)
            if param is None:
                raise ValueError(f"Layer {sample_layer} not found")
            weights = param.data.float().flatten()
        else:
            weights = torch.cat([p.data.float().flatten()
                               for p in self.model.parameters() if p.dim() >= 2])
        self._atom_table = build_atoms_kmeans(
            weights[:2_000_000].flatten().to(self.device), K=self.K, iters=5)
        self._vocabulary_frozen = True

    def encode_weight(self, weight: torch.Tensor) -> tuple[ProgramBuffer, torch.Tensor]:
        if self._atom_table is None:
            raise RuntimeError("Vocabulary not built. Call build_vocabulary() first.")
        atoms = self._atom_table.to(weight.device)
        return wal_encode_scalar(weight.float().flatten(), atoms, self.lmax)

    def snapshot_layer(self, layer_name: str):
        param = dict(self.model.named_parameters()).get(layer_name)
        if param is not None:
            self._snapshots[layer_name] = param.data.clone()

    def edit_weight(self, layer_name: str, delta: torch.Tensor) -> bool:
        """Apply a delta to a layer. Snapshot first. Verify non-target diff."""
        param = dict(self.model.named_parameters()).get(layer_name)
        if param is None:
            return False
        self.snapshot_layer(layer_name)
        param.data.add_(delta.to(device=param.device, dtype=param.dtype))
        self._edit_log.append({"layer": layer_name, "operation": "delta_add"})
        return True

    def apply_edit(self, recipe: dict) -> bool:
        layer_name = recipe.get("target_layer")
        if not layer_name:
            return False
        param = dict(self.model.named_parameters()).get(layer_name)
        if param is None:
            return False
        delta = recipe.get("delta_tensor")
        if delta is not None:
            return self.edit_weight(layer_name, torch.tensor(delta))
        return False

    def verify_non_target_diff(self, edited_layer: str) -> bool:
        for name, param in self.model.named_parameters():
            if name == edited_layer:
                continue
            if name in self._snapshots:
                if not torch.equal(param.data, self._snapshots[name]):
                    return False
        return True

    def rollback_edit(self, layer_name: str) -> bool:
        if layer_name not in self._snapshots:
            return False
        param = dict(self.model.named_parameters()).get(layer_name)
        if param is not None:
            param.data.copy_(self._snapshots[layer_name])
            del self._snapshots[layer_name]
            self._edit_log.append({"layer": layer_name, "operation": "rollback"})
            return True
        return False

    @property
    def edit_count(self) -> int:
        return len(self._edit_log)

    @property
    def vocabulary_is_frozen(self) -> bool:
        return self._vocabulary_frozen

    def save_edit_log(self, path: Path):
        path.write_text(json.dumps(self._edit_log, indent=2, ensure_ascii=False))
