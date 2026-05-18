from __future__ import annotations

import math

import torch
from torch import Tensor

from .block_vq_encode import encode_block_residual_vq
from .block_vq_encoding import BlockRVQEncoding, GroupedBlockRVQEncoding


@torch.no_grad()
def encode_grouped_block_residual_vq(
    weight: Tensor,
    *,
    group_rows: int,
    block_size: int = 32,
    codebook_size: int = 256,
    num_stages: int = 2,
    product_splits: int = 1,
    stages_per_split: tuple[int, ...] | None = None,
    normalize_blocks: str = "none",
    transform_kind: str = "none",
    calibrate_stage_scales: bool = False,
    residual_correction: str = "none",
    sample_limit: int = 131_072,
    kmeans_iters: int = 8,
    batch_size: int = 16_384,
) -> GroupedBlockRVQEncoding:
    if group_rows < 1:
        raise ValueError("group_rows must be >= 1")
    groups: list[BlockRVQEncoding] = []
    row_slices: list[tuple[int, int]] = []
    for start in range(0, int(weight.shape[0]), group_rows):
        end = min(start + group_rows, int(weight.shape[0]))
        groups.append(
            encode_block_residual_vq(
                weight[start:end],
                block_size=block_size,
                codebook_size=codebook_size,
                num_stages=num_stages,
                product_splits=product_splits,
                stages_per_split=stages_per_split,
                normalize_blocks=normalize_blocks,
                transform_kind=transform_kind,
                calibrate_stage_scales=calibrate_stage_scales,
                residual_correction=residual_correction,
                sample_limit=sample_limit,
                kmeans_iters=kmeans_iters,
                batch_size=batch_size,
            )
        )
        row_slices.append((start, end))
    return GroupedBlockRVQEncoding(
        groups=tuple(groups),
        row_slices=tuple(row_slices),
        original_shape=(int(weight.shape[0]), int(weight.shape[1])),
    )


@torch.no_grad()
def sample_row_similarity(stage_ids: tuple[Tensor, ...], n_pairs: int = 1024) -> dict[str, float]:
    if len(stage_ids) < 1:
        raise ValueError("need at least one stage")
    rows, blocks_per_row = stage_ids[0].shape
    combined = torch.zeros(rows, blocks_per_row, dtype=torch.int64, device=stage_ids[0].device)
    radix = 1
    for ids in stage_ids:
        max_id = int(ids.max().item()) + 1
        combined += ids.to(torch.int64) * radix
        radix *= max(max_id, 1)
    generator = torch.Generator(device=combined.device)
    generator.manual_seed(0)
    a = torch.randint(0, rows, (n_pairs,), generator=generator, device=combined.device)
    b = torch.randint(0, rows, (n_pairs,), generator=generator, device=combined.device)
    sim = (combined[a] == combined[b]).float().mean(dim=1).cpu()
    return {
        "mean_sim": float(sim.mean().item()),
        "p50_sim": float(sim.median().item()),
        "p90_sim": float(sim.quantile(0.90).item()),
        "p99_sim": float(sim.quantile(0.99).item()),
    }


@torch.no_grad()
def sample_grouped_row_similarity(groups: tuple[BlockRVQEncoding, ...], n_pairs_per_group: int = 256) -> dict[str, float]:
    sims = []
    weights = []
    for group in groups:
        if group.stage_ids[0].shape[0] < 2:
            continue
        stats = sample_row_similarity(group.stage_ids, n_pairs=n_pairs_per_group)
        sims.append(stats)
        weights.append(group.stage_ids[0].shape[0])
    if not sims:
        return {"mean_sim": 0.0, "p50_sim": 0.0, "p90_sim": 0.0, "p99_sim": 0.0}
    total = float(sum(weights))
    return {
        key: float(sum(stats[key] * w for stats, w in zip(sims, weights)) / total)
        for key in sims[0]
    }


def storage_megabytes(num_bytes: int) -> float:
    return float(num_bytes) / 1e6


def dense_bf16_storage_bytes(weight: Tensor) -> int:
    return int(weight.numel()) * 2


def ideal_id_bits_per_weight(block_size: int, codebook_size: int, num_stages: int) -> float:
    return num_stages * math.ceil(math.log2(max(codebook_size, 2))) / float(block_size)
