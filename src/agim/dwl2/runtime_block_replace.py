from __future__ import annotations

import torch
from torch import nn

from .block_vq import GroupedBlockRVQEncoding, encode_grouped_block_residual_vq
from .eager_layers import EagerBlockRVQLinear
from .grouped_block_rvq_linear import PackedGroupedBlockRVQLinear
from .route_encoder import rel_mse
from .runtime_common import _set_submodule


@torch.no_grad()
def replace_with_eager_block_rvq(
    model: nn.Module,
    target_module_names: tuple[str, ...],
    *,
    group_rows: int = 2048,
    block_size: int = 32,
    codebook_size: int = 256,
    num_stages: int = 4,
    product_splits: int = 1,
    normalize_blocks: str = "none",
    transform_kind: str = "none",
    calibrate_stage_scales: bool = False,
    residual_correction: str = "none",
    sample_limit: int = 65_536,
    kmeans_iters: int = 8,
    batch_size: int = 16_384,
    target_dtype: torch.dtype = torch.bfloat16,
) -> list[dict[str, float]]:
    target_set = set(target_module_names)
    stats = []
    for name, module in model.named_modules():
        if name not in target_set or not isinstance(module, nn.Linear):
            continue
        original_weight = module.weight.detach()
        enc = encode_grouped_block_residual_vq(
            original_weight,
            group_rows=group_rows,
            block_size=block_size,
            codebook_size=codebook_size,
            num_stages=num_stages,
            product_splits=product_splits,
            normalize_blocks=normalize_blocks,
            transform_kind=transform_kind,
            calibrate_stage_scales=calibrate_stage_scales,
            residual_correction=residual_correction,
            sample_limit=sample_limit,
            kmeans_iters=kmeans_iters,
            batch_size=batch_size,
        )
        approx_weight = enc.reconstruct(out_dtype=target_dtype)
        packed = EagerBlockRVQLinear(approx_weight, module.bias, target_dtype=target_dtype)
        _set_submodule(model, name, packed)
        try:
            if hasattr(module, "_parameters"):
                module._parameters["weight"] = None
                module._parameters["bias"] = None
        except (AttributeError, TypeError, KeyError):
            pass
        stats.append(
            {
                "name": name,
                "tensor_name": f"{name}.weight",
                "runtime_decision": "eager_block_rvq",
                "rel_mse": float(rel_mse(original_weight.float(), approx_weight.float()).item()),
                "storage_bytes": int(enc.storage_bytes()),
                "bits_per_weight": float(enc.bits_per_weight()),
                "group_rows": int(group_rows),
                "block_size": int(block_size),
                "codebook_size": int(codebook_size),
                "num_stages": int(num_stages),
                "product_splits": int(product_splits),
                "normalize_blocks": normalize_blocks,
                "transform_kind": transform_kind,
                "calibrate_stage_scales": bool(calibrate_stage_scales),
                "residual_correction": residual_correction,
            }
        )
    return stats
def replace_with_packed_block_rvq(
    model: nn.Module,
    target_module_names: tuple[str, ...],
    *,
    group_rows: int = 2048,
    block_size: int = 32,
    codebook_size: int = 256,
    num_stages: int = 4,
    product_splits: int = 1,
    normalize_blocks: str = "none",
    transform_kind: str = "none",
    calibrate_stage_scales: bool = False,
    residual_correction: str = "none",
    sample_limit: int = 65_536,
    kmeans_iters: int = 8,
    batch_size: int = 16_384,
    matmul_strategy: str = "per_group",
    matmul_chunk_rows: int | None = None,
    local_palette_group_cols: int | None = None,
    hot_topk: int | None = None,
    hot_score_mode: str = "row_scale_norm",
    hot_min_stage_share: float = 0.0,
    hot_score_threshold_ratio: float = 0.0,
) -> list[dict[str, float]]:
    target_set = set(target_module_names)
    stats = []
    for name, module in model.named_modules():
        if name not in target_set or not isinstance(module, nn.Linear):
            continue
        original_weight = module.weight.detach()
        enc = encode_grouped_block_residual_vq(
            original_weight,
            group_rows=group_rows,
            block_size=block_size,
            codebook_size=codebook_size,
            num_stages=num_stages,
            product_splits=product_splits,
            normalize_blocks=normalize_blocks,
            transform_kind=transform_kind,
            calibrate_stage_scales=calibrate_stage_scales,
            residual_correction=residual_correction,
            sample_limit=sample_limit,
            kmeans_iters=kmeans_iters,
            batch_size=batch_size,
        )
        approx_weight = enc.reconstruct(out_dtype=torch.bfloat16)
        packed = PackedGroupedBlockRVQLinear(
            enc,
            module.bias,
            matmul_strategy=matmul_strategy,
            matmul_chunk_rows=matmul_chunk_rows,
            local_palette_group_cols=local_palette_group_cols,
            hot_topk=hot_topk,
            hot_score_mode=hot_score_mode,
            hot_min_stage_share=hot_min_stage_share,
            hot_score_threshold_ratio=hot_score_threshold_ratio,
        ).to(original_weight.device)
        _set_submodule(model, name, packed)
        try:
            if hasattr(module, "_parameters"):
                module._parameters["weight"] = None
                module._parameters["bias"] = None
        except (AttributeError, TypeError, KeyError):
            pass
        stats.append(
            {
                "name": name,
                "tensor_name": f"{name}.weight",
                "runtime_decision": "packed_block_rvq",
                "matmul_strategy": matmul_strategy,
                "matmul_chunk_rows": int(matmul_chunk_rows) if matmul_chunk_rows is not None else None,
                "local_palette_group_cols": int(local_palette_group_cols) if local_palette_group_cols is not None else None,
                "hot_topk": int(hot_topk) if hot_topk is not None else None,
                "hot_score_mode": hot_score_mode,
                "hot_min_stage_share": float(hot_min_stage_share),
                "hot_score_threshold_ratio": float(hot_score_threshold_ratio),
                "rel_mse": float(rel_mse(original_weight.float(), approx_weight.float()).item()),
                "storage_bytes": int(enc.storage_bytes()),
                "bits_per_weight": float(enc.bits_per_weight()),
                "group_rows": int(group_rows),
                "block_size": int(block_size),
                "codebook_size": int(codebook_size),
                "num_stages": int(num_stages),
                "product_splits": int(product_splits),
                "normalize_blocks": normalize_blocks,
                "transform_kind": transform_kind,
                "calibrate_stage_scales": bool(calibrate_stage_scales),
                "residual_correction": residual_correction,
            }
        )
    return stats
def replace_with_preencoded_packed_block_rvq(
    model: nn.Module,
    encodings_by_name: dict[str, GroupedBlockRVQEncoding],
    *,
    matmul_strategy: str = "per_group",
    matmul_chunk_rows: int | None = None,
    local_palette_group_cols: int | None = None,
    hot_topk: int | None = None,
    hot_score_mode: str = "row_scale_norm",
    hot_min_stage_share: float = 0.0,
    hot_score_threshold_ratio: float = 0.0,
) -> list[dict[str, float]]:
    stats = []
    for name, module in model.named_modules():
        if name not in encodings_by_name or not isinstance(module, nn.Linear):
            continue
        original_weight = module.weight.detach()
        enc = encodings_by_name[name]
        approx_weight = enc.reconstruct(out_dtype=torch.bfloat16).to(original_weight.device)
        packed = PackedGroupedBlockRVQLinear(
            enc,
            module.bias,
            matmul_strategy=matmul_strategy,
            matmul_chunk_rows=matmul_chunk_rows,
            local_palette_group_cols=local_palette_group_cols,
            hot_topk=hot_topk,
            hot_score_mode=hot_score_mode,
            hot_min_stage_share=hot_min_stage_share,
            hot_score_threshold_ratio=hot_score_threshold_ratio,
        ).to(original_weight.device)
        _set_submodule(model, name, packed)
        try:
            if hasattr(module, "_parameters"):
                module._parameters["weight"] = None
                module._parameters["bias"] = None
        except (AttributeError, TypeError, KeyError):
            pass
        stats.append(
            {
                "name": name,
                "tensor_name": f"{name}.weight",
                "runtime_decision": "preencoded_packed_block_rvq",
                "matmul_strategy": matmul_strategy,
                "matmul_chunk_rows": int(matmul_chunk_rows) if matmul_chunk_rows is not None else None,
                "local_palette_group_cols": int(local_palette_group_cols) if local_palette_group_cols is not None else None,
                "hot_topk": int(hot_topk) if hot_topk is not None else None,
                "hot_score_mode": hot_score_mode,
                "hot_min_stage_share": float(hot_min_stage_share),
                "hot_score_threshold_ratio": float(hot_score_threshold_ratio),
                "rel_mse": float(rel_mse(original_weight.float(), approx_weight.float()).item()),
                "storage_bytes": int(enc.storage_bytes()),
                "bits_per_weight": float(enc.bits_per_weight()),
            }
        )
    return stats
