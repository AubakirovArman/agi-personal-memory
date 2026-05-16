"""v1.0: MEMIT — Mass-Editing Memory in Transformer. Batch knowledge editing."""
from __future__ import annotations

import torch
from dataclasses import dataclass, field


@dataclass
class MEMITEdit:
    subject: str
    target: str
    relation: str = "is"
    target_layer: int = 5
    applied: bool = False


class MEMITEditor:
    """Batch knowledge editor — updates multiple facts sharing updates across layers.

    Extends ROME's single-fact approach to batch editing with
    shared covariance across all edits for stability.
    """

    def __init__(self, model: torch.nn.Module, tokenizer, device: str = "cuda:0"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._edit_batch: list[MEMITEdit] = []
        self._original_states: dict[str, torch.Tensor] = {}
        self._edit_count = 0

    def add_to_batch(self, subject: str, target: str, relation: str = "is",
                     target_layer: int = 5):
        self._edit_batch.append(MEMITEdit(subject=subject, target=target,
                                          relation=relation, target_layer=target_layer))

    def apply_batch(self, clamp_norm: float = 0.1) -> int:
        """Apply all batched edits with shared covariance for stability."""
        if not self._edit_batch:
            return 0
        layers = set(e.target_layer for e in self._edit_batch)
        applied = 0
        for layer_idx in layers:
            layer_edits = [e for e in self._edit_batch if e.target_layer == layer_idx]
            if not layer_edits:
                continue
            try:
                mlp = self.model.model.layers[layer_idx].mlp
                weight = mlp.down_proj.weight.data
                if f"layer_{layer_idx}" not in self._original_states:
                    self._original_states[f"layer_{layer_idx}"] = weight.clone()

                keys, values = [], []
                for edit in layer_edits:
                    inputs = self.tokenizer(edit.subject, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        hidden = self.model.model.embed_tokens(inputs.input_ids)
                        for i in range(layer_idx):
                            hidden = self.model.model.layers[i](hidden, attention_mask=None)[0]
                        k = hidden[:, -1, :]
                        k = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
                        keys.append(k.squeeze(0))
                        target_ids = self.tokenizer(edit.target, return_tensors="pt").to(self.device)
                        v = torch.zeros(weight.shape[0], device=self.device)
                        tid = target_ids.input_ids[0, -1] % weight.shape[0]
                        v[tid] = 1.0
                        values.append(v)

                if keys:
                    K = torch.stack(keys)
                    V = torch.stack(values)
                    KtK = K.T @ K
                    eye = torch.eye(KtK.shape[0], device=self.device) * 1e-6
                    delta = (V.T @ K) @ torch.inverse(KtK + eye)
                    delta = delta * clamp_norm / (delta.norm() + 1e-8)
                    mlp.down_proj.weight.data = weight + delta.to(weight.dtype)
                    for e in layer_edits:
                        e.applied = True
                    applied += len(layer_edits)
            except Exception:
                continue
        self._edit_count += applied
        self._edit_batch.clear()
        return applied

    def rollback(self) -> bool:
        for name, original in self._original_states.items():
            layer_idx = int(name.split("_")[1])
            self.model.model.layers[layer_idx].mlp.down_proj.weight.data = original
        self._original_states.clear()
        return True

    @property
    def edit_count(self) -> int:
        return self._edit_count
