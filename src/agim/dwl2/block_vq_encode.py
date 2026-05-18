from __future__ import annotations

import torch
from torch import Tensor

from .block_vq_encoding import BlockRVQEncoding
from .block_vq_kmeans import _assign_to_codebook, _fit_kmeans, _id_storage_dtype, _reshape_blocks
from .block_vq_transforms import (
    _fit_pca_transform,
    _pack_sign_bits,
    _polar_transform,
    _sign_correction_matrix,
    _transform_matrix,
)


@torch.no_grad()
def encode_block_residual_vq(
    weight: Tensor,
    *,
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
) -> BlockRVQEncoding:
    if weight.ndim != 2:
        raise ValueError("expected 2D weight")
    if num_stages < 1:
        raise ValueError("num_stages must be >= 1")
    if product_splits < 1 or block_size % product_splits != 0:
        raise ValueError("product_splits must divide block_size and be >= 1")
    if stages_per_split is None:
        split_stage_counts = tuple(int(num_stages) for _ in range(product_splits))
    else:
        split_stage_counts = tuple(int(count) for count in stages_per_split)
        if len(split_stage_counts) != product_splits:
            raise ValueError("stages_per_split must have exactly product_splits entries")
        if any(count < 1 for count in split_stage_counts):
            raise ValueError("each stages_per_split entry must be >= 1")

    row_scale = weight.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8).to(torch.float16)
    w_norm = (weight.to(torch.float32) / row_scale.to(torch.float32))
    blocks, rows, padded_cols = _reshape_blocks(w_norm, block_size)
    num_blocks_per_row = padded_cols // block_size
    transform = _transform_matrix(transform_kind, block_size, device=blocks.device)
    transform_bias = None
    if transform_kind == "pca":
        if sample_limit < codebook_size:
            sample_limit = codebook_size
        sample_count = min(int(sample_limit), blocks.shape[0])
        generator = torch.Generator(device=blocks.device)
        generator.manual_seed(0)
        pick_for_pca = torch.randperm(blocks.shape[0], generator=generator, device=blocks.device)[:sample_count]
        transform, transform_bias = _fit_pca_transform(blocks[pick_for_pca])
        blocks = blocks - transform_bias
    if transform_kind == "polar":
        blocks = _polar_transform(blocks)
    elif transform is not None:
        blocks = blocks @ transform.t()

    block_scale = None
    if normalize_blocks == "amax":
        block_scale = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        blocks = blocks / block_scale
    elif normalize_blocks == "l2":
        block_scale = blocks.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
        blocks = blocks / block_scale
    elif normalize_blocks != "none":
        raise ValueError("normalize_blocks must be one of: none, amax, l2")

    residual = blocks.clone()
    if sample_limit < codebook_size:
        sample_limit = codebook_size
    sample_count = min(int(sample_limit), residual.shape[0])
    generator = torch.Generator(device=residual.device)
    generator.manual_seed(0)
    pick = torch.randperm(residual.shape[0], generator=generator, device=residual.device)[:sample_count]
    sample_residual = residual[pick].clone()

    stage_ids: list[Tensor] = []
    codebooks: list[Tensor] = []
    stage_value_dims: list[int] = []

    if product_splits == 1:
        for _ in range(split_stage_counts[0]):
            codebook = _fit_kmeans(sample_residual, codebook_size=codebook_size, iters=kmeans_iters, batch_size=batch_size)
            full_ids, _ = _assign_to_codebook(residual, codebook, batch_size=batch_size)
            residual -= codebook[full_ids]
            sample_ids, _ = _assign_to_codebook(sample_residual, codebook, batch_size=batch_size)
            sample_residual -= codebook[sample_ids]
            stage_ids.append(full_ids.view(rows, num_blocks_per_row).to(_id_storage_dtype(codebook_size)))
            codebooks.append(codebook.to(torch.float16))
            stage_value_dims.append(block_size)
    else:
        sub_dim = block_size // product_splits
        for split, split_stage_count in enumerate(split_stage_counts):
            lo = split * sub_dim
            hi = lo + sub_dim
            residual_slice = residual[:, lo:hi]
            sample_slice = sample_residual[:, lo:hi]
            for _ in range(split_stage_count):
                codebook_slice = _fit_kmeans(sample_slice, codebook_size=codebook_size, iters=kmeans_iters, batch_size=batch_size)
                full_ids, _ = _assign_to_codebook(residual_slice, codebook_slice, batch_size=batch_size)
                residual_slice -= codebook_slice[full_ids]
                sample_ids, _ = _assign_to_codebook(sample_slice, codebook_slice, batch_size=batch_size)
                sample_slice -= codebook_slice[sample_ids]
                codebook_full = torch.zeros(codebook_size, block_size, dtype=torch.float32, device=residual.device)
                codebook_full[:, lo:hi] = codebook_slice
                stage_ids.append(full_ids.view(rows, num_blocks_per_row).to(_id_storage_dtype(codebook_size)))
                codebooks.append(codebook_full.to(torch.float16))
                stage_value_dims.append(sub_dim)
            residual[:, lo:hi] = residual_slice
            sample_residual[:, lo:hi] = sample_slice

    if residual_correction not in {"none", "sign"}:
        raise ValueError("residual_correction must be one of: none, sign")

    stage_scales = None
    full_stage_blocks = [
        codebook[ids.reshape(-1).long()].to(torch.float32)
        for ids, codebook in zip(stage_ids, codebooks)
    ]
    baseline_recon_blocks = torch.zeros_like(blocks, dtype=torch.float32)
    for stage in full_stage_blocks:
        baseline_recon_blocks = baseline_recon_blocks + stage

    def _weight_rel_mse(block_recon: Tensor) -> Tensor:
        restored = block_recon
        if block_scale is not None:
            restored = restored * block_scale.to(torch.float32)
        if transform is not None:
            restored = restored @ transform.to(torch.float32)
        if transform_bias is not None:
            restored = restored + transform_bias.to(torch.float32)
        weight_recon = restored.view(rows, num_blocks_per_row * block_size)[:, : int(weight.shape[1])]
        weight_recon = weight_recon * row_scale.to(torch.float32)
        return (weight.to(torch.float32) - weight_recon).square().mean() / weight.to(torch.float32).square().mean().clamp_min(1e-12)

    baseline_weight_loss = _weight_rel_mse(baseline_recon_blocks)
    if calibrate_stage_scales and full_stage_blocks:
        gram = torch.empty(len(full_stage_blocks), len(full_stage_blocks), dtype=torch.float32, device=blocks.device)
        rhs = torch.empty(len(full_stage_blocks), dtype=torch.float32, device=blocks.device)
        target = blocks.to(torch.float32)
        for i, stage_i in enumerate(full_stage_blocks):
            rhs[i] = torch.sum(stage_i * target)
            for j, stage_j in enumerate(full_stage_blocks[i:], start=i):
                value = torch.sum(stage_i * stage_j)
                gram[i, j] = value
                gram[j, i] = value
        damp = 1e-4 * torch.eye(len(full_stage_blocks), device=blocks.device, dtype=torch.float32)
        solved = torch.linalg.solve(gram + damp, rhs)
        candidate_stage_scales = solved.clamp_min(0.0)
        if float(candidate_stage_scales.sum().item()) > 0.0:
            candidate_recon_blocks = torch.zeros_like(blocks, dtype=torch.float32)
            for idx, stage in enumerate(full_stage_blocks):
                candidate_recon_blocks = candidate_recon_blocks + stage * candidate_stage_scales[idx]
            candidate_weight_loss = _weight_rel_mse(candidate_recon_blocks)
            if bool(candidate_weight_loss < baseline_weight_loss):
                stage_scales = candidate_stage_scales
    if stage_scales is not None:
        recon_blocks = torch.zeros_like(blocks, dtype=torch.float32)
        for idx, stage in enumerate(full_stage_blocks):
            recon_blocks = recon_blocks + stage * stage_scales[idx]
    else:
        recon_blocks = baseline_recon_blocks
    residual_signs = None
    residual_scale = None
    current_weight_loss = _weight_rel_mse(recon_blocks)
    if residual_correction == "sign":
        residual = blocks.to(torch.float32) - recon_blocks
        correction_matrix = _sign_correction_matrix(block_size, device=blocks.device)
        projected_residual = residual @ correction_matrix.t()
        candidate_scale = projected_residual.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
        candidate_signs = projected_residual >= 0
        candidate_correction = (candidate_signs.to(torch.float32) * 2.0 - 1.0) * candidate_scale
        candidate_correction = candidate_correction @ correction_matrix.to(torch.float32)
        candidate_weight_loss = _weight_rel_mse(recon_blocks + candidate_correction)
        if bool(candidate_weight_loss < current_weight_loss):
            residual_signs = _pack_sign_bits(candidate_signs)
            residual_scale = candidate_scale.to(torch.float16)
            recon_blocks = recon_blocks + candidate_correction
    sample_rel_mse = float(
        ((blocks[pick].to(torch.float32) - recon_blocks[pick]).square().mean() / blocks[pick].square().mean().clamp_min(1e-12)).item()
    )
    return BlockRVQEncoding(
        stage_ids=tuple(stage_ids),
        codebooks=tuple(codebooks),
        stage_value_dims=tuple(stage_value_dims),
        stages_per_split=split_stage_counts,
        stage_scales=None if stage_scales is None else stage_scales.to(torch.float16),
        residual_correction="none" if residual_signs is None else residual_correction,
        residual_signs=None if residual_signs is None else residual_signs.contiguous(),
        residual_scale=None if residual_scale is None else residual_scale.contiguous(),
        row_scale=row_scale,
        block_scale=None if block_scale is None else block_scale.view(rows, num_blocks_per_row).to(torch.float16),
        transform_kind=transform_kind,
        transform_matrix=None if transform is None or transform_kind != "pca" else transform.to(torch.float16),
        transform_bias=None if transform_bias is None else transform_bias.to(torch.float16),
        product_splits=product_splits,
        original_shape=(int(weight.shape[0]), int(weight.shape[1])),
        padded_cols=padded_cols,
        block_size=block_size,
        sample_rel_mse=sample_rel_mse,
    )
