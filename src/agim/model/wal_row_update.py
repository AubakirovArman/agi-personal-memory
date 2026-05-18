"""Row update helpers for WAL-backed editors."""
from __future__ import annotations

import torch

from ..wal.encoder import wal_encode_scalar_gpu


def add_row_delta(weight: torch.Tensor, row_idx: int, delta: torch.Tensor,
                  atoms: torch.Tensor, lmax: int, wal_encode: bool) -> None:
    """Apply a row delta, optionally round-tripping through WAL encoding."""
    row = weight[row_idx, :]
    updated = row.float().to(delta.device) + delta
    if wal_encode:
        _, _, updated = wal_encode_scalar_gpu(updated, atoms, lmax)
    if updated.shape != row.shape:
        raise RuntimeError(
            f"WAL reconstruction shape mismatch: {updated.shape} != {row.shape}"
        )
    weight[row_idx, :] = updated.to(device=weight.device, dtype=weight.dtype)
