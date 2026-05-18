from __future__ import annotations

import torch
from torch import Tensor


class PackedBlockRVQHotCacheMixin:
    def _build_stage_hot_cache(
        self,
        out_dtype: torch.dtype,
        device: torch.device,
        hot_topk: int,
        score_mode: str,
        min_stage_share: float,
        score_threshold_ratio: float = 0.0,
    ) -> None:
        """Build stage-local hot/cold codebook caches.

        Hot ids are chosen from the current encoding, not an external JSON, so
        the cache stays aligned with the actual per-run codebook assignment.

        Note: exact activation norms from M23 are not available inside the
        runtime surface. `stage_influence_proxy` therefore uses the same static
        stage-local proxy as M23 without the activation term:
        `row_scale * codebook_norm * occurrence_mass`.
        """
        if (
            self._fast_codebooks_cached is None
            or self._fast_cache_dtype != out_dtype
            or self._fast_cache_device != device
        ):
            self._build_fast_cache(out_dtype, device)
        rows, blocks_per_row = getattr(self, "stage_ids_0").shape
        row_weights = self.row_scale.to(torch.float32).abs().reshape(int(rows), 1).expand(-1, int(blocks_per_row)).reshape(-1)
        hot_codebooks: list[Tensor | None] = []
        hot_top_ids_list: list[Tensor | None] = []
        hot_luts: list[Tensor | None] = []
        hot_positions: list[Tensor | None] = []
        hot_slots_list: list[Tensor | None] = []
        cold_positions: list[Tensor | None] = []
        cold_ids_list: list[Tensor | None] = []
        hot_stage_shares: list[float] = []
        for idx in range(self.num_stages):
            ids = getattr(self, f"stage_ids_{idx}").reshape(-1).to(torch.int64)
            full_codebook = self._fast_codebooks_cached[idx]
            codebook_size = int(full_codebook.shape[0])
            topk = min(max(int(hot_topk), 1), codebook_size)
            scores = torch.zeros(codebook_size, dtype=torch.float32, device=device)
            if score_mode == "count":
                scores.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))
            elif score_mode in {"row_scale_norm", "stage_influence_proxy"}:
                codebook_norm = full_codebook.to(torch.float32).norm(dim=-1)
                scores.scatter_add_(0, ids, row_weights * codebook_norm.index_select(0, ids))
            else:
                raise ValueError(f"unsupported hot score_mode: {score_mode}")
            sorted_ids = torch.argsort(scores, descending=True)
            sorted_scores = scores.index_select(0, sorted_ids)
            if float(score_threshold_ratio) > 0.0 and sorted_scores.numel() > 0 and float(sorted_scores[0].item()) > 0.0:
                threshold = float(score_threshold_ratio) * float(sorted_scores[0].item())
                selected = sorted_ids[sorted_scores >= threshold]
                if selected.numel() == 0:
                    selected = sorted_ids[:1]
                top_ids = selected[:topk]
            else:
                top_ids = sorted_ids[:topk]
            total_score = float(scores.sum().item())
            hot_score = float(scores.index_select(0, top_ids).sum().item())
            hot_share = hot_score / max(total_score, 1e-12)
            hot_stage_shares.append(hot_share)
            if hot_share < float(min_stage_share):
                hot_codebooks.append(None)
                hot_top_ids_list.append(None)
                hot_luts.append(None)
                hot_positions.append(None)
                hot_slots_list.append(None)
                cold_positions.append(None)
                cold_ids_list.append(None)
                continue
            hot_codebook = full_codebook.index_select(0, top_ids).contiguous()
            hot_top_ids_list.append(top_ids.contiguous())
            id_to_slot = torch.full((codebook_size,), -1, dtype=torch.int16, device=device)
            id_to_slot[top_ids] = torch.arange(top_ids.numel(), dtype=torch.int16, device=device)
            hot_slots = id_to_slot.index_select(0, ids)
            hot_mask = hot_slots >= 0
            hot_pos = torch.nonzero(hot_mask, as_tuple=False).reshape(-1).contiguous()
            cold_pos = torch.nonzero(~hot_mask, as_tuple=False).reshape(-1).contiguous()
            hot_codebooks.append(hot_codebook)
            hot_luts.append(id_to_slot)
            hot_positions.append(hot_pos)
            hot_slots_list.append(hot_slots.index_select(0, hot_pos).to(torch.int64).contiguous())
            cold_positions.append(cold_pos)
            cold_ids_list.append(ids.index_select(0, cold_pos).contiguous())
        self._hot_codebooks_cached = hot_codebooks
        self._hot_top_ids_cached = hot_top_ids_list
        self._hot_id_to_slot_cached = hot_luts
        self._hot_positions_cached = hot_positions
        self._hot_slots_cached = hot_slots_list
        self._cold_positions_cached = cold_positions
        self._cold_ids_cached = cold_ids_list
        self._hot_cache_dtype = out_dtype
        self._hot_cache_device = device
        self._hot_cache_topk = int(hot_topk)
        self._hot_cache_score_mode = score_mode
        self._hot_cache_min_stage_share = float(min_stage_share)
        self._hot_cache_score_threshold_ratio = float(score_threshold_ratio)
        self._hot_stage_share_cached = hot_stage_shares

    def _build_hot_cache(
        self,
        out_dtype: torch.dtype,
        device: torch.device,
        hot_topk: int,
        score_mode: str,
        min_stage_share: float,
        score_threshold_ratio: float = 0.0,
    ) -> None:
        self._build_stage_hot_cache(
            out_dtype,
            device,
            hot_topk,
            score_mode,
            min_stage_share,
            score_threshold_ratio,
        )
