from __future__ import annotations

import torch
from torch import Tensor


class PackedBlockRVQReconstructMixin:
    def reconstruct_weight(self, out_dtype: torch.dtype) -> Tensor:
        rows, blocks_per_row = getattr(self, "stage_ids_0").shape
        flat_blocks = int(rows) * int(blocks_per_row)
        recon = torch.zeros(flat_blocks, self.block_size, dtype=torch.float32, device=self.row_scale.device)
        stage_scales = None if self.stage_scales is None else self.stage_scales.to(torch.float32)
        for idx, (ids, codebook) in enumerate(self._stage_tensors()):
            stage = codebook[ids.reshape(-1).long()].to(torch.float32)
            if stage_scales is not None:
                stage = stage * stage_scales[idx]
            recon += stage
        if self.residual_correction != "none" and self.residual_signs is not None and self.residual_scale is not None:
            from .block_vq import _sign_correction_matrix, _unpack_sign_bits

            signs = _unpack_sign_bits(self.residual_signs, self.block_size).to(torch.float32)
            signs = signs * 2.0 - 1.0
            correction = signs * self.residual_scale.reshape(-1, 1).to(torch.float32)
            correction = correction @ _sign_correction_matrix(self.block_size, device=recon.device).to(torch.float32)
            recon = recon + correction
        if self.block_scale is not None:
            recon = recon * self.block_scale.reshape(-1, 1).to(torch.float32)
        if self.transform_kind == "polar":
            from .block_vq import _inverse_polar_transform

            recon = _inverse_polar_transform(recon)
        else:
            transform_matrix = self.transform_matrix
            if transform_matrix is None and self.transform_kind != "none":
                from .block_vq import _transform_matrix

                transform_matrix = _transform_matrix(self.transform_kind, self.block_size, device=recon.device)
            if transform_matrix is not None:
                recon = recon @ transform_matrix.to(torch.float32)
        if self.transform_bias is not None:
            recon = recon + self.transform_bias.to(torch.float32)
        weight = recon.view(rows, blocks_per_row * self.block_size)[:, : self.in_features]
        return (weight.to(out_dtype) * self.row_scale.to(out_dtype))

    def supports_fast_reconstruct(self) -> bool:
        """True for the plain RVQ case where bf16 accumulation suffices.

        Skips fp32 casts, torch.zeros allocation, and per-call dtype conversions.
        """
        return (
            self.block_scale is None
            and self.transform_kind == "none"
            and self.transform_matrix is None
            and self.transform_bias is None
            and self.residual_correction == "none"
            and self.residual_signs is None
            and self.residual_scale is None
        )

    def _build_fast_cache(self, out_dtype: torch.dtype, device: torch.device) -> None:
        """Build cached out_dtype codebooks (with stage_scales merged) and bf16 row_scale.

        Note: ids are NOT pre-cast to int64. Doing so would multiply the on-device
        ids cache size by 8x (uint8 -> int64) and dominate VRAM cost. Instead we
        rely on inline `.long()` of a small contiguous tensor which is cheap.
        """
        codebooks_cached = []
        stage_scales = self.stage_scales
        for idx in range(self.num_stages):
            cb = getattr(self, f"codebook_{idx}").to(out_dtype)
            if stage_scales is not None:
                cb = cb * stage_scales[idx].to(out_dtype)
            codebooks_cached.append(cb.contiguous())
        self._fast_codebooks_cached = codebooks_cached
        self._fast_row_scale_cached = self.row_scale.to(out_dtype).contiguous()
        self._fast_cache_dtype = out_dtype
        self._fast_cache_device = device

    def reconstruct_weight_fast(self, out_dtype: torch.dtype) -> Tensor:
        """Allocation-light reconstruction for plain RVQ groups.

        Eliminates: torch.zeros allocation, fp32 codebook casts, fp32 accumulation,
        per-call row_scale dtype cast, per-call stage_scales dtype cast.
        Keeps: inline ids.long() cast (cheap; pre-caching int64 ids would 8x VRAM).
        Accumulates directly in out_dtype (bf16) which is sufficient when
        codebook_size <= 256 and num_stages <= 4.
        """
        if not self.supports_fast_reconstruct():
            return self.reconstruct_weight(out_dtype)
        device = self.row_scale.device
        if (
            self._fast_codebooks_cached is None
            or self._fast_cache_dtype != out_dtype
            or self._fast_cache_device != device
        ):
            self._build_fast_cache(out_dtype, device)
        cb_list = self._fast_codebooks_cached
        # M21: build the split-aware list of active sub-stage indices.
        active_idx = self._compute_active_stage_indices()
        first = active_idx[0]
        ids0 = getattr(self, f"stage_ids_{first}").reshape(-1).to(torch.int64)
        recon = cb_list[first].index_select(0, ids0)
        for idx in active_idx[1:]:
            ids = getattr(self, f"stage_ids_{idx}").reshape(-1).to(torch.int64)
            recon.add_(cb_list[idx].index_select(0, ids))
        rows, blocks_per_row = getattr(self, "stage_ids_0").shape
        weight = recon.view(int(rows), int(blocks_per_row) * self.block_size)[:, : self.in_features]
        weight.mul_(self._fast_row_scale_cached)
        return weight

    def reconstruct_weight_fast_norm(self, out_dtype: torch.dtype) -> Tensor:
        """Fast pre-row-scale reconstruct matching ``reconstruct_weight_fast`` internals."""
        if not self.supports_fast_reconstruct():
            return self.reconstruct_weight_norm(out_dtype)
        device = self.row_scale.device
        if (
            self._fast_codebooks_cached is None
            or self._fast_cache_dtype != out_dtype
            or self._fast_cache_device != device
        ):
            self._build_fast_cache(out_dtype, device)
        cb_list = self._fast_codebooks_cached
        active_idx = self._compute_active_stage_indices()
        first = active_idx[0]
        ids0 = getattr(self, f"stage_ids_{first}").reshape(-1).to(torch.int64)
        recon = cb_list[first].index_select(0, ids0)
        for idx in active_idx[1:]:
            ids = getattr(self, f"stage_ids_{idx}").reshape(-1).to(torch.int64)
            recon.add_(cb_list[idx].index_select(0, ids))
        rows, blocks_per_row = getattr(self, "stage_ids_0").shape
        return recon.view(int(rows), int(blocks_per_row) * self.block_size)[:, : self.in_features]

    def reconstruct_weight_norm(self, out_dtype: torch.dtype) -> Tensor:
        rows, blocks_per_row = getattr(self, "stage_ids_0").shape
        flat_blocks = int(rows) * int(blocks_per_row)
        recon = torch.zeros(flat_blocks, self.block_size, dtype=torch.float32, device=self.row_scale.device)
        stage_scales = None if self.stage_scales is None else self.stage_scales.to(torch.float32)
        for idx, (ids, codebook) in enumerate(self._stage_tensors()):
            stage = codebook[ids.reshape(-1).long()].to(torch.float32)
            if stage_scales is not None:
                stage = stage * stage_scales[idx]
            recon += stage
        if self.residual_correction != "none" and self.residual_signs is not None and self.residual_scale is not None:
            from .block_vq import _sign_correction_matrix, _unpack_sign_bits

            signs = _unpack_sign_bits(self.residual_signs, self.block_size).to(torch.float32)
            signs = signs * 2.0 - 1.0
            correction = signs * self.residual_scale.reshape(-1, 1).to(torch.float32)
            correction = correction @ _sign_correction_matrix(self.block_size, device=recon.device).to(torch.float32)
            recon = recon + correction
        if self.block_scale is not None:
            recon = recon * self.block_scale.reshape(-1, 1).to(torch.float32)
        if self.transform_kind == "polar":
            from .block_vq import _inverse_polar_transform

            recon = _inverse_polar_transform(recon)
        else:
            transform_matrix = self.transform_matrix
            if transform_matrix is None and self.transform_kind != "none":
                from .block_vq import _transform_matrix

                transform_matrix = _transform_matrix(self.transform_kind, self.block_size, device=recon.device)
            if transform_matrix is not None:
                recon = recon @ transform_matrix.to(torch.float32)
        if self.transform_bias is not None:
            recon = recon + self.transform_bias.to(torch.float32)
        weight = recon.view(rows, blocks_per_row * self.block_size)[:, : self.in_features]
        return weight.to(out_dtype)
