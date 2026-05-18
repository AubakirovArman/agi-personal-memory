from __future__ import annotations

import torch
from torch import Tensor, nn

from .block_vq import BlockRVQEncoding
from .block_rvq_group_hot_cache import PackedBlockRVQHotCacheMixin
from .block_rvq_group_hot_reconstruct import PackedBlockRVQHotReconstructMixin
from .block_rvq_group_matmul import PackedBlockRVQMatmulMixin
from .block_rvq_group_reconstruct import PackedBlockRVQReconstructMixin
from .block_rvq_group_stage_local import PackedBlockRVQStageLocalMixin


class PackedBlockRVQGroup(
    PackedBlockRVQStageLocalMixin,
    PackedBlockRVQMatmulMixin,
    PackedBlockRVQHotReconstructMixin,
    PackedBlockRVQHotCacheMixin,
    PackedBlockRVQReconstructMixin,
    nn.Module,
):
    def __init__(self, enc: BlockRVQEncoding) -> None:
        super().__init__()
        self.block_size = int(enc.block_size)
        self.in_features = int(enc.original_shape[1])
        self.out_features = int(enc.original_shape[0])
        self.padded_cols = int(enc.padded_cols)
        self.num_stages = len(enc.stage_ids)
        # M21: stages live in split-major order. We keep stages_per_split so
        # callers can request "drop last residual stage uniformly across splits".
        self.product_splits = int(enc.product_splits)
        self.stages_per_split = tuple(int(x) for x in enc.stages_per_split)
        self.transform_kind = enc.transform_kind
        for idx, ids in enumerate(enc.stage_ids):
            self.register_buffer(f"stage_ids_{idx}", ids.contiguous())
        for idx, codebook in enumerate(enc.codebooks):
            self.register_buffer(f"codebook_{idx}", codebook.contiguous().to(torch.float16))
        if enc.stage_scales is not None:
            self.register_buffer("stage_scales", enc.stage_scales.to(torch.float16).contiguous())
        else:
            self.stage_scales = None
        self.residual_correction = enc.residual_correction
        if enc.residual_signs is not None:
            self.register_buffer("residual_signs", enc.residual_signs.contiguous())
        else:
            self.residual_signs = None
        if enc.residual_scale is not None:
            self.register_buffer("residual_scale", enc.residual_scale.to(torch.float16).contiguous())
        else:
            self.residual_scale = None
        self.register_buffer("row_scale", enc.row_scale.to(torch.float16).contiguous())
        if enc.block_scale is not None:
            self.register_buffer("block_scale", enc.block_scale.to(torch.float16).contiguous())
        else:
            self.block_scale = None
        if enc.transform_matrix is not None:
            self.register_buffer("transform_matrix", enc.transform_matrix.to(torch.float16).contiguous())
        else:
            self.transform_matrix = None
        if enc.transform_bias is not None:
            self.register_buffer("transform_bias", enc.transform_bias.to(torch.float16).contiguous())
        else:
            self.transform_bias = None
        self._triton_stage_ids: Tensor | None = None
        self._triton_codebooks: Tensor | None = None
        self._triton_cache_device: torch.device | None = None
        self._triton_stage_local_hot_ids: Tensor | None = None
        self._triton_stage_local_full_codebooks: Tensor | None = None
        self._triton_stage_local_hot_codebooks: Tensor | None = None
        self._triton_stage_local_hot_lut: Tensor | None = None
        self._triton_stage_local_b1_stage_ids: Tensor | None = None
        self._triton_stage_local_b1_full_codebooks: Tensor | None = None
        self._triton_stage_local_b1_hot_ids: Tensor | None = None
        self._triton_stage_local_b1_hot_codebooks: Tensor | None = None
        self._triton_stage_local_b1_row_scale: Tensor | None = None
        self._triton_stage_local_b1_cache_device: torch.device | None = None
        self._triton_stage_local_b1_cache_dtype: torch.dtype | None = None
        self._triton_stage_local_b1_cache_topk: int | None = None
        self._triton_stage_local_b1_cache_score_mode: str | None = None
        self._triton_stage_local_b1_cache_min_stage_share: float | None = None
        self._triton_stage_local_b1_cache_score_threshold_ratio: float | None = None
        self._triton_stage_local_cache_device: torch.device | None = None
        self._triton_stage_local_cache_dtype: torch.dtype | None = None
        self._triton_stage_local_cache_topk: int | None = None
        self._triton_stage_local_cache_score_mode: str | None = None
        self._triton_stage_local_cache_min_stage_share: float | None = None
        self._triton_stage_local_cache_score_threshold_ratio: float | None = None
        # M20 fast-path caches: built lazily on first call of reconstruct_weight_fast
        self._fast_codebooks_cached: list[Tensor] | None = None
        self._fast_row_scale_cached: Tensor | None = None
        self._fast_cache_dtype: torch.dtype | None = None
        self._fast_cache_device: torch.device | None = None
        # M21: variable-stage decoding. effective_stages_per_split controls how
        # many residual stages we sum *per split*. Range: [1, max(stages_per_split)].
        # Defaults to max(stages_per_split) (no change in behavior).
        self.effective_stages_per_split: int = max(self.stages_per_split) if self.stages_per_split else self.num_stages
        # Cached active sub-stage indices, rebuilt lazily when effective_stages_per_split changes.
        self._active_stage_indices: tuple[int, ...] | None = None
        # M23: stage-local hot/cold codebook caches.
        self._hot_codebooks_cached: list[Tensor | None] | None = None
        self._hot_top_ids_cached: list[Tensor | None] | None = None
        self._hot_id_to_slot_cached: list[Tensor | None] | None = None
        self._hot_positions_cached: list[Tensor | None] | None = None
        self._hot_slots_cached: list[Tensor | None] | None = None
        self._cold_positions_cached: list[Tensor | None] | None = None
        self._cold_ids_cached: list[Tensor | None] | None = None
        self._hot_cache_dtype: torch.dtype | None = None
        self._hot_cache_device: torch.device | None = None
        self._hot_cache_topk: int | None = None
        self._hot_cache_score_mode: str | None = None
        self._hot_cache_min_stage_share: float | None = None
        self._hot_cache_score_threshold_ratio: float | None = None
        self._hot_stage_share_cached: list[float] | None = None
        # M26 stage 1: cached recon buffer reused by reconstruct_weight_hot_v2.
        self._hot_recon_buffer_cached: Tensor | None = None
        self._hot_recon_buffer_dtype: torch.dtype | None = None
        self._hot_recon_buffer_device: torch.device | None = None

    def _stage_tensors(self) -> list[tuple[Tensor, Tensor]]:
        return [
            (getattr(self, f"stage_ids_{idx}"), getattr(self, f"codebook_{idx}"))
            for idx in range(self.num_stages)
        ]

    def _compute_active_stage_indices(self) -> tuple[int, ...]:
        """M21: return sub-stage indices that should be summed under
        ``effective_stages_per_split``. Stages are split-major:
        for product_splits=ps and stages_per_split=(s,)*ps, the layout is
        [split0_stage0..s-1, split1_stage0..s-1, ...]. We keep the first
        ``effective_stages_per_split`` stages of every split.
        """
        if self._active_stage_indices is not None:
            return self._active_stage_indices
        keep = max(1, int(self.effective_stages_per_split))
        idxs: list[int] = []
        offset = 0
        for split_count in self.stages_per_split:
            local_keep = min(keep, int(split_count))
            for j in range(local_keep):
                idxs.append(offset + j)
            offset += int(split_count)
        if not idxs:
            idxs = [0]
        self._active_stage_indices = tuple(idxs)
        return self._active_stage_indices

    def set_effective_stages_per_split(self, k: int) -> None:
        """Set per-split active stage count and invalidate the cached index list."""
        self.effective_stages_per_split = int(k)
        self._active_stage_indices = None
