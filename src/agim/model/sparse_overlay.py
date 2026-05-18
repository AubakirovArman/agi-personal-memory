"""Runtime sparse row overlays for frozen base model weights."""
from __future__ import annotations

import torch


class RuntimeSparseOverlay:
    """Apply row-level lm_head/embed deltas at runtime without mutating weights."""

    def __init__(self, model):
        self.model = model
        self.lm_deltas: dict[int, torch.Tensor] = {}
        self.embed_deltas: dict[int, torch.Tensor] = {}
        self._handles = []

    def add_lm_delta(self, row_id: int, delta: torch.Tensor) -> None:
        self._add_delta(self.lm_deltas, row_id, delta)

    def add_embed_delta(self, row_id: int, delta: torch.Tensor) -> None:
        self._add_delta(self.embed_deltas, row_id, delta)

    def clear(self) -> None:
        self.lm_deltas.clear()
        self.embed_deltas.clear()

    def install(self) -> "RuntimeSparseOverlay":
        if self._handles:
            return self
        self._handles.append(self.model.lm_head.register_forward_hook(self._lm_hook))
        embed = self.model.model.embed_tokens
        self._handles.append(embed.register_forward_hook(self._embed_hook))
        return self

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "RuntimeSparseOverlay":
        return self.install()

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.remove()

    @staticmethod
    def _add_delta(store: dict[int, torch.Tensor],
                   row_id: int, delta: torch.Tensor) -> None:
        moved = delta.detach().float().cpu()
        row_id = int(row_id)
        store[row_id] = store.get(row_id, torch.zeros_like(moved)) + moved

    def _lm_hook(self, _module, inputs, output):
        if not self.lm_deltas:
            return output
        hidden = inputs[0]
        rows = sorted(self.lm_deltas)
        deltas = torch.stack([
            self.lm_deltas[row].to(device=hidden.device, dtype=hidden.dtype)
            for row in rows
        ])
        correction = hidden @ deltas.T
        adjusted = output.clone()
        adjusted[..., rows] = adjusted[..., rows] + correction
        return adjusted

    def _embed_hook(self, _module, inputs, output):
        if not self.embed_deltas:
            return output
        token_ids = inputs[0]
        adjusted = output.clone()
        for row_id, delta in self.embed_deltas.items():
            moved = delta.to(device=output.device, dtype=output.dtype)
            mask = (token_ids == row_id).unsqueeze(-1)
            adjusted = adjusted + mask * moved
        return adjusted
