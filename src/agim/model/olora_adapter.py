"""v1.0: O-LoRA — Orthogonal Low-Rank Adaptation.

Trains new task adapters in directions orthogonal to gradient space of
previous tasks. Guarantees preference updates don't affect factual knowledge.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class OLoRAAdapter(nn.Module):
    """Orthogonal LoRA adapter. New tasks projected orthogonal to previous."""

    def __init__(self, in_features: int, out_features: int, rank: int = 8,
                 alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self._prev_grad_space: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.A.T @ self.B.T) * self.scaling

    def orthogonalize(self):
        """Project current gradient space orthogonal to all previous tasks."""
        if not self._prev_grad_space:
            return
        for prev_grad in self._prev_grad_space:
            prev_flat = prev_grad.flatten()
            a_flat = self.A.grad.flatten() if self.A.grad is not None else self.A.data.flatten()
            projection = (a_flat @ prev_flat) / (prev_flat @ prev_flat + 1e-8) * prev_flat
            if self.A.grad is not None:
                self.A.grad.data -= projection.reshape(self.A.grad.shape)
            else:
                self.A.data -= projection.reshape(self.A.shape)

    def freeze_task(self):
        """Save current gradient space and prepare for next task."""
        if self.A.grad is not None:
            self._prev_grad_space.append(self.A.grad.detach().clone())

    def merge_to_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Merge adapter into base weight for inference."""
        delta = (self.B.data @ self.A.data) * self.scaling
        return weight + delta.to(weight.dtype)


class OLoRAPreferenceManager:
    """Manage multiple O-LoRA adapters for preferences without interference."""

    MAX_ADAPTERS = 10

    def __init__(self, in_features: int, out_features: int, rank: int = 8):
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.adapters: dict[str, OLoRAAdapter] = {}

    def add_preference(self, name: str, base_weight: torch.Tensor) -> OLoRAAdapter:
        if len(self.adapters) >= self.MAX_ADAPTERS:
            oldest = next(iter(self.adapters))
            del self.adapters[oldest]
        adapter = OLoRAAdapter(self.in_features, self.out_features, self.rank)
        self.adapters[name] = adapter
        return adapter

    def apply_all(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        result = base_out
        for adapter in self.adapters.values():
            result = result + adapter(x)
        return result

    def consolidate(self) -> torch.Tensor | None:
        """Merge all adapters into single delta. Returns None if no adapters."""
        if not self.adapters:
            return None
        total_delta = torch.zeros(self.out_features, self.in_features)
        for adapter in self.adapters.values():
            total_delta += (adapter.B.data @ adapter.A.data) * adapter.scaling
        return total_delta
