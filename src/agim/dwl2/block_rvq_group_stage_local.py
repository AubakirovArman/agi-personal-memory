from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .triton_stage_local_hot_cold import (
    stage_local_hot_cold_matmul,
    stage_local_hot_palette_b2_matmul,
    stage_local_hot_palette_matmul,
)


class PackedBlockRVQStageLocalMixin:
    def _triton_stage_local_hot_cold_state(
        self,
        out_dtype: torch.dtype,
        hot_topk: int,
        score_mode: str,
        min_stage_share: float,
        score_threshold_ratio: float = 0.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if not self.supports_triton_block_matmul():
            raise ValueError("stage_local_hot_cold requires plain uint8 block RVQ encodings")
        device = self.row_scale.device
        if (
            self._fast_codebooks_cached is None
            or self._fast_cache_dtype != out_dtype
            or self._fast_cache_device != device
        ):
            self._build_fast_cache(out_dtype, device)
        if (
            self._hot_codebooks_cached is None
            or self._hot_id_to_slot_cached is None
            or self._hot_cache_dtype != out_dtype
            or self._hot_cache_device != device
            or self._hot_cache_topk != int(hot_topk)
            or self._hot_cache_score_mode != score_mode
            or self._hot_cache_min_stage_share != float(min_stage_share)
            or self._hot_cache_score_threshold_ratio != float(score_threshold_ratio)
        ):
            self._build_hot_cache(out_dtype, device, int(hot_topk), score_mode, float(min_stage_share), float(score_threshold_ratio))
        active_idx = self._compute_active_stage_indices()
        if (
            self._triton_stage_local_hot_ids is None
            or self._triton_stage_local_full_codebooks is None
            or self._triton_stage_local_hot_codebooks is None
            or self._triton_stage_local_hot_lut is None
            or self._triton_stage_local_cache_device != device
            or self._triton_stage_local_cache_dtype != out_dtype
            or self._triton_stage_local_cache_topk != int(hot_topk)
            or self._triton_stage_local_cache_score_mode != score_mode
            or self._triton_stage_local_cache_min_stage_share != float(min_stage_share)
            or self._triton_stage_local_cache_score_threshold_ratio != float(score_threshold_ratio)
            or int(self._triton_stage_local_hot_ids.shape[0]) != len(active_idx)
        ):
            stage_ids = []
            full_codebooks = []
            hot_codebooks = []
            hot_luts = []
            for idx in active_idx:
                full_cb = self._fast_codebooks_cached[idx].contiguous()
                hot_cb = self._hot_codebooks_cached[idx]
                hot_lut = self._hot_id_to_slot_cached[idx]
                stage_ids.append(getattr(self, f"stage_ids_{idx}").contiguous())
                full_codebooks.append(full_cb)
                if hot_cb is None or hot_lut is None:
                    hot_codebooks.append(torch.zeros((int(hot_topk), self.block_size), dtype=out_dtype, device=device))
                    hot_luts.append(torch.full((int(full_cb.shape[0]),), -1, dtype=torch.int16, device=device))
                    continue
                if int(hot_cb.shape[0]) < int(hot_topk):
                    pad = torch.zeros((int(hot_topk) - int(hot_cb.shape[0]), self.block_size), dtype=hot_cb.dtype, device=device)
                    hot_cb = torch.cat([hot_cb, pad], dim=0)
                hot_codebooks.append(hot_cb.contiguous())
                hot_luts.append(hot_lut.contiguous())
            self._triton_stage_local_hot_ids = torch.stack(stage_ids, dim=0).contiguous()
            self._triton_stage_local_full_codebooks = torch.stack(full_codebooks, dim=0).contiguous()
            self._triton_stage_local_hot_codebooks = torch.stack(hot_codebooks, dim=0).contiguous()
            self._triton_stage_local_hot_lut = torch.stack(hot_luts, dim=0).contiguous()
            self._triton_stage_local_cache_device = device
            self._triton_stage_local_cache_dtype = out_dtype
            self._triton_stage_local_cache_topk = int(hot_topk)
            self._triton_stage_local_cache_score_mode = score_mode
            self._triton_stage_local_cache_min_stage_share = float(min_stage_share)
            self._triton_stage_local_cache_score_threshold_ratio = float(score_threshold_ratio)
        return (
            self._triton_stage_local_hot_ids,
            self._triton_stage_local_full_codebooks,
            self._triton_stage_local_hot_codebooks,
            self._triton_stage_local_hot_lut,
            self._fast_row_scale_cached,
        )

    def triton_stage_local_hot_cold_matmul(
        self,
        x_flat: Tensor,
        hot_topk: int,
        score_mode: str = "row_scale_norm",
        min_stage_share: float = 0.0,
        score_threshold_ratio: float = 0.0,
    ) -> Tensor:
        if not x_flat.is_cuda or not self.supports_triton_block_matmul():
            return F.linear(
                x_flat,
                self.reconstruct_weight_hot_v2(
                    out_dtype=x_flat.dtype,
                    hot_topk=hot_topk,
                    score_mode=score_mode,
                    min_stage_share=min_stage_share,
                    score_threshold_ratio=score_threshold_ratio,
                ),
                bias=None,
            )
        stage_ids, full_codebooks, hot_codebooks, hot_lut, row_scale = self._triton_stage_local_hot_cold_state(
            x_flat.dtype,
            hot_topk,
            score_mode,
            min_stage_share,
            score_threshold_ratio,
        )
        return stage_local_hot_cold_matmul(
            x_flat,
            stage_ids,
            full_codebooks,
            hot_codebooks,
            hot_lut,
            row_scale,
        )

    def _triton_stage_local_hot_palette_state(
        self,
        out_dtype: torch.dtype,
        hot_topk: int,
        score_mode: str,
        min_stage_share: float,
        score_threshold_ratio: float = 0.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if not self.supports_triton_block_matmul():
            raise ValueError("stage_local_hot_cold_b1 requires plain uint8 block RVQ encodings")
        device = self.row_scale.device
        if (
            self._fast_codebooks_cached is None
            or self._fast_cache_dtype != out_dtype
            or self._fast_cache_device != device
        ):
            self._build_fast_cache(out_dtype, device)
        if (
            self._hot_codebooks_cached is None
            or self._hot_top_ids_cached is None
            or self._hot_cache_dtype != out_dtype
            or self._hot_cache_device != device
            or self._hot_cache_topk != int(hot_topk)
            or self._hot_cache_score_mode != score_mode
            or self._hot_cache_min_stage_share != float(min_stage_share)
            or self._hot_cache_score_threshold_ratio != float(score_threshold_ratio)
        ):
            self._build_hot_cache(out_dtype, device, int(hot_topk), score_mode, float(min_stage_share), float(score_threshold_ratio))
        active_idx = self._compute_active_stage_indices()
        if (
            self._triton_stage_local_b1_stage_ids is None
            or self._triton_stage_local_b1_full_codebooks is None
            or self._triton_stage_local_b1_hot_ids is None
            or self._triton_stage_local_b1_hot_codebooks is None
            or self._triton_stage_local_b1_row_scale is None
            or self._triton_stage_local_b1_cache_device != device
            or self._triton_stage_local_b1_cache_dtype != out_dtype
            or self._triton_stage_local_b1_cache_topk != int(hot_topk)
            or self._triton_stage_local_b1_cache_score_mode != score_mode
            or self._triton_stage_local_b1_cache_min_stage_share != float(min_stage_share)
            or self._triton_stage_local_b1_cache_score_threshold_ratio != float(score_threshold_ratio)
            or int(self._triton_stage_local_b1_stage_ids.shape[0]) != len(active_idx)
        ):
            stage_ids = []
            full_codebooks = []
            hot_ids = []
            hot_codebooks = []
            for idx in active_idx:
                full_cb = self._fast_codebooks_cached[idx].contiguous()
                stage_ids.append(getattr(self, f"stage_ids_{idx}").contiguous())
                full_codebooks.append(full_cb)
                stage_hot_ids = None if self._hot_top_ids_cached is None else self._hot_top_ids_cached[idx]
                stage_hot_cb = self._hot_codebooks_cached[idx]
                if stage_hot_ids is None or stage_hot_cb is None:
                    hot_ids.append(torch.full((int(hot_topk),), -1, dtype=torch.int32, device=device))
                    hot_codebooks.append(torch.zeros((int(hot_topk), self.block_size), dtype=out_dtype, device=device))
                    continue
                hot_ids_i = stage_hot_ids[: int(hot_topk)].to(torch.int32).contiguous()
                hot_cb_i = stage_hot_cb[: int(hot_topk)].contiguous()
                if int(hot_ids_i.numel()) < int(hot_topk):
                    pad_ids = torch.full((int(hot_topk) - int(hot_ids_i.numel()),), -1, dtype=torch.int32, device=device)
                    pad_cb = torch.zeros((int(hot_topk) - int(hot_cb_i.shape[0]), self.block_size), dtype=out_dtype, device=device)
                    hot_ids_i = torch.cat([hot_ids_i, pad_ids], dim=0)
                    hot_cb_i = torch.cat([hot_cb_i, pad_cb], dim=0)
                hot_ids.append(hot_ids_i)
                hot_codebooks.append(hot_cb_i)
            self._triton_stage_local_b1_stage_ids = torch.stack(stage_ids, dim=0).contiguous()
            self._triton_stage_local_b1_full_codebooks = torch.stack(full_codebooks, dim=0).contiguous()
            self._triton_stage_local_b1_hot_ids = torch.stack(hot_ids, dim=0).contiguous()
            self._triton_stage_local_b1_hot_codebooks = torch.stack(hot_codebooks, dim=0).contiguous()
            self._triton_stage_local_b1_row_scale = self._fast_row_scale_cached
            self._triton_stage_local_b1_cache_device = device
            self._triton_stage_local_b1_cache_dtype = out_dtype
            self._triton_stage_local_b1_cache_topk = int(hot_topk)
            self._triton_stage_local_b1_cache_score_mode = score_mode
            self._triton_stage_local_b1_cache_min_stage_share = float(min_stage_share)
            self._triton_stage_local_b1_cache_score_threshold_ratio = float(score_threshold_ratio)
        return (
            self._triton_stage_local_b1_stage_ids,
            self._triton_stage_local_b1_full_codebooks,
            self._triton_stage_local_b1_hot_ids,
            self._triton_stage_local_b1_hot_codebooks,
            self._triton_stage_local_b1_row_scale,
        )

    def triton_stage_local_hot_palette_matmul(
        self,
        x_flat: Tensor,
        hot_topk: int,
        score_mode: str = "row_scale_norm",
        min_stage_share: float = 0.0,
        score_threshold_ratio: float = 0.0,
    ) -> Tensor:
        if not x_flat.is_cuda or not self.supports_triton_block_matmul():
            return F.linear(
                x_flat,
                self.reconstruct_weight_hot_v2(
                    out_dtype=x_flat.dtype,
                    hot_topk=hot_topk,
                    score_mode=score_mode,
                    min_stage_share=min_stage_share,
                    score_threshold_ratio=score_threshold_ratio,
                ),
                bias=None,
            )
        stage_ids, full_codebooks, hot_ids, hot_codebooks, row_scale = self._triton_stage_local_hot_palette_state(
            x_flat.dtype,
            hot_topk,
            score_mode,
            min_stage_share,
            score_threshold_ratio,
        )
        return stage_local_hot_palette_matmul(
            x_flat,
            stage_ids,
            full_codebooks,
            hot_ids,
            hot_codebooks,
            row_scale,
        )

    def triton_stage_local_hot_palette_b2_matmul(
        self,
        x_flat: Tensor,
        hot_topk: int,
        score_mode: str = "row_scale_norm",
        min_stage_share: float = 0.0,
        score_threshold_ratio: float = 0.0,
    ) -> Tensor:
        if not x_flat.is_cuda or not self.supports_triton_block_matmul():
            return F.linear(
                x_flat,
                self.reconstruct_weight_hot_v2(
                    out_dtype=x_flat.dtype,
                    hot_topk=hot_topk,
                    score_mode=score_mode,
                    min_stage_share=min_stage_share,
                    score_threshold_ratio=score_threshold_ratio,
                ),
                bias=None,
            )
        stage_ids, full_codebooks, hot_codebooks, hot_lut, row_scale = self._triton_stage_local_hot_cold_state(
            x_flat.dtype,
            hot_topk,
            score_mode,
            min_stage_share,
            score_threshold_ratio,
        )
        return stage_local_hot_palette_b2_matmul(
            x_flat,
            stage_ids,
            full_codebooks,
            hot_codebooks,
            hot_lut,
            row_scale,
        )
