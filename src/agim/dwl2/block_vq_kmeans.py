from __future__ import annotations

import torch
from torch import Tensor


def _id_storage_bytes(codebook_size: int) -> int:
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive")
    if codebook_size <= 256:
        return 1
    if codebook_size <= 65_536:
        return 2
    return 4


def _id_storage_dtype(codebook_size: int) -> torch.dtype:
    if codebook_size <= 256:
        return torch.uint8
    if codebook_size <= 32_767:
        return torch.int16
    return torch.int32


def _reshape_blocks(w_norm: Tensor, block_size: int) -> tuple[Tensor, int, int]:
    if w_norm.ndim != 2:
        raise ValueError("expected 2D weight matrix")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    rows, cols = w_norm.shape
    padded_cols = ((cols + block_size - 1) // block_size) * block_size
    if padded_cols != cols:
        w_norm = torch.nn.functional.pad(w_norm, (0, padded_cols - cols), value=0.0)
    return w_norm.view(rows * (padded_cols // block_size), block_size), rows, padded_cols


def _assign_to_codebook(blocks: Tensor, codebook: Tensor, batch_size: int = 16_384) -> tuple[Tensor, Tensor]:
    if blocks.numel() == 0:
        raise ValueError("blocks must be non-empty")
    blocks = blocks.to(torch.float32)
    codebook = codebook.to(device=blocks.device, dtype=torch.float32)
    code_sq = codebook.square().sum(dim=1)
    ids = torch.empty(blocks.shape[0], dtype=torch.int64, device=blocks.device)
    min_dist = torch.empty(blocks.shape[0], dtype=torch.float32, device=blocks.device)
    for start in range(0, blocks.shape[0], batch_size):
        end = min(start + batch_size, blocks.shape[0])
        chunk = blocks[start:end]
        chunk_sq = chunk.square().sum(dim=1, keepdim=True)
        dist = chunk_sq + code_sq.unsqueeze(0) - 2.0 * (chunk @ codebook.t())
        best_dist, best_ids = dist.min(dim=1)
        ids[start:end] = best_ids
        min_dist[start:end] = best_dist
    return ids, min_dist


def _fit_kmeans(
    blocks: Tensor,
    codebook_size: int,
    iters: int,
    batch_size: int = 16_384,
) -> Tensor:
    if blocks.ndim != 2:
        raise ValueError("expected 2D blocks")
    if blocks.shape[0] < codebook_size:
        raise ValueError("sample set must be at least as large as codebook_size")
    blocks = blocks.to(torch.float32)
    generator = torch.Generator(device=blocks.device)
    generator.manual_seed(0)
    perm = torch.randperm(blocks.shape[0], generator=generator, device=blocks.device)
    codebook = blocks[perm[:codebook_size]].clone()
    for _ in range(iters):
        ids, _ = _assign_to_codebook(blocks, codebook, batch_size=batch_size)
        counts = torch.bincount(ids, minlength=codebook_size)
        sums = torch.zeros_like(codebook)
        sums.index_add_(0, ids, blocks)
        nonzero = counts > 0
        updated = codebook.clone()
        updated[nonzero] = sums[nonzero] / counts[nonzero].to(sums.dtype).unsqueeze(1)
        empty = (~nonzero).nonzero(as_tuple=True)[0]
        if empty.numel() > 0:
            reseed = torch.randint(0, blocks.shape[0], (empty.numel(),), generator=generator, device=blocks.device)
            updated[empty] = blocks[reseed]
        codebook = updated
    return codebook
