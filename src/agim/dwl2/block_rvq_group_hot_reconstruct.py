from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


class PackedBlockRVQHotReconstructMixin:
    def reconstruct_weight_hot(
        self,
        out_dtype: torch.dtype,
        hot_topk: int,
        score_mode: str = "row_scale_norm",
        min_stage_share: float = 0.0,
        score_threshold_ratio: float = 0.0,
    ) -> Tensor:
        """Experimental M23 path: stage-local hot/cold codebook reconstruction.

        Each sub-stage keeps a small hot codebook of top ids chosen from the
        current encoding. Stages whose hot ids do not cover enough score mass
        fall back to the normal fast path.
        """
        if not self.supports_fast_reconstruct() or int(hot_topk) <= 0:
            return self.reconstruct_weight_fast(out_dtype)
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
            or self._hot_positions_cached is None
            or self._hot_slots_cached is None
            or self._cold_positions_cached is None
            or self._cold_ids_cached is None
            or self._hot_cache_dtype != out_dtype
            or self._hot_cache_device != device
            or self._hot_cache_topk != int(hot_topk)
            or self._hot_cache_score_mode != score_mode
            or self._hot_cache_min_stage_share != float(min_stage_share)
            or self._hot_cache_score_threshold_ratio != float(score_threshold_ratio)
        ):
            self._build_hot_cache(out_dtype, device, int(hot_topk), score_mode, float(min_stage_share), float(score_threshold_ratio))
        if all(item is None for item in self._hot_codebooks_cached):
            return self.reconstruct_weight_fast(out_dtype)
        cb_list = self._fast_codebooks_cached
        active_idx = self._compute_active_stage_indices()
        recon = None
        for idx in active_idx:
            hot_codebook = self._hot_codebooks_cached[idx]
            hot_pos = self._hot_positions_cached[idx]
            hot_slots = self._hot_slots_cached[idx]
            cold_pos = self._cold_positions_cached[idx]
            cold_ids = self._cold_ids_cached[idx]
            if hot_codebook is None or hot_pos is None or hot_slots is None or cold_pos is None or cold_ids is None:
                ids = getattr(self, f"stage_ids_{idx}").reshape(-1).to(torch.int64)
                stage = cb_list[idx].index_select(0, ids)
            else:
                stage_size = int(getattr(self, f"stage_ids_{idx}").numel())
                if hot_slots.numel() == stage_size:
                    stage = hot_codebook.index_select(0, hot_slots)
                elif hot_slots.numel() == 0:
                    stage = cb_list[idx].index_select(0, cold_ids)
                else:
                    stage = torch.empty((stage_size, self.block_size), dtype=out_dtype, device=device)
                    stage.index_copy_(0, hot_pos, hot_codebook.index_select(0, hot_slots))
                    stage.index_copy_(0, cold_pos, cb_list[idx].index_select(0, cold_ids))
            if recon is None:
                recon = stage
            else:
                recon.add_(stage)
        rows, blocks_per_row = getattr(self, "stage_ids_0").shape
        weight = recon.view(int(rows), int(blocks_per_row) * self.block_size)[:, : self.in_features]
        weight.mul_(self._fast_row_scale_cached)
        return weight

    def _ensure_hot_recon_buffer(
        self, stage_size: int, out_dtype: torch.dtype, device: torch.device
    ) -> Tensor:
        """M26 stage 1: cached per-group recon buffer for hot/cold v2 path."""
        buf = self._hot_recon_buffer_cached
        if (
            buf is None
            or buf.shape[0] != stage_size
            or buf.shape[1] != self.block_size
            or buf.dtype != out_dtype
            or buf.device != device
        ):
            buf = torch.empty((stage_size, self.block_size), dtype=out_dtype, device=device)
            self._hot_recon_buffer_cached = buf
            self._hot_recon_buffer_dtype = out_dtype
            self._hot_recon_buffer_device = device
        return buf

    def reconstruct_weight_hot_v2(
        self,
        out_dtype: torch.dtype,
        hot_topk: int,
        score_mode: str = "row_scale_norm",
        min_stage_share: float = 0.0,
        score_threshold_ratio: float = 0.0,
    ) -> Tensor:
        """M26 stage 1: allocation-free variant of reconstruct_weight_hot.

        Reuses a cached (stage_size, block_size) recon buffer per group. The
        first active sub-stage initialises the buffer via index_copy_; every
        subsequent active sub-stage accumulates in-place via index_add_.
        Numerically equivalent to ``reconstruct_weight_hot`` on the same
        encoding (and to ``reconstruct_weight_fast`` because hot/cold is exact),
        but eliminates the per-stage ``torch.empty`` allocation.
        """
        if not self.supports_fast_reconstruct() or int(hot_topk) <= 0:
            return self.reconstruct_weight_fast(out_dtype)
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
            or self._hot_positions_cached is None
            or self._hot_slots_cached is None
            or self._cold_positions_cached is None
            or self._cold_ids_cached is None
            or self._hot_cache_dtype != out_dtype
            or self._hot_cache_device != device
            or self._hot_cache_topk != int(hot_topk)
            or self._hot_cache_score_mode != score_mode
            or self._hot_cache_min_stage_share != float(min_stage_share)
            or self._hot_cache_score_threshold_ratio != float(score_threshold_ratio)
        ):
            self._build_hot_cache(out_dtype, device, int(hot_topk), score_mode, float(min_stage_share), float(score_threshold_ratio))
        if all(item is None for item in self._hot_codebooks_cached):
            return self.reconstruct_weight_fast(out_dtype)
        cb_list = self._fast_codebooks_cached
        active_idx = self._compute_active_stage_indices()
        rows, blocks_per_row = getattr(self, "stage_ids_0").shape
        stage_size = int(rows) * int(blocks_per_row)
        recon = self._ensure_hot_recon_buffer(stage_size, out_dtype, device)
        first_done = False
        for idx in active_idx:
            hot_codebook = self._hot_codebooks_cached[idx]
            hot_pos = self._hot_positions_cached[idx]
            hot_slots = self._hot_slots_cached[idx]
            cold_pos = self._cold_positions_cached[idx]
            cold_ids = self._cold_ids_cached[idx]
            stage_fallback = (
                hot_codebook is None or hot_pos is None or hot_slots is None
                or cold_pos is None or cold_ids is None
            )
            if stage_fallback:
                ids = getattr(self, f"stage_ids_{idx}").reshape(-1).to(torch.int64)
                stage = cb_list[idx].index_select(0, ids)
                if not first_done:
                    recon.copy_(stage)
                    first_done = True
                else:
                    recon.add_(stage)
                continue
            if not first_done:
                if hot_slots.numel() == stage_size:
                    recon.copy_(hot_codebook.index_select(0, hot_slots))
                elif hot_slots.numel() == 0:
                    recon.copy_(cb_list[idx].index_select(0, cold_ids))
                else:
                    recon.index_copy_(0, hot_pos, hot_codebook.index_select(0, hot_slots))
                    recon.index_copy_(0, cold_pos, cb_list[idx].index_select(0, cold_ids))
                first_done = True
            else:
                if hot_slots.numel() == stage_size:
                    recon.index_add_(0, hot_pos, hot_codebook.index_select(0, hot_slots))
                elif hot_slots.numel() == 0:
                    recon.index_add_(0, cold_pos, cb_list[idx].index_select(0, cold_ids))
                else:
                    recon.index_add_(0, hot_pos, hot_codebook.index_select(0, hot_slots))
                    recon.index_add_(0, cold_pos, cb_list[idx].index_select(0, cold_ids))
        weight_view = recon.view(int(rows), int(blocks_per_row) * self.block_size)[:, : self.in_features]
        # NOTE: do not mul_ in place; recon is a cached buffer reused across forwards.
        return weight_view * self._fast_row_scale_cached
