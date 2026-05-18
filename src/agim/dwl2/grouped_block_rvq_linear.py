from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .block_vq import GroupedBlockRVQEncoding
from .block_rvq_group import PackedBlockRVQGroup
from .full_layer_tiled_runtime import (
    build_grouped_local_hotprefix_plan,
    build_grouped_local_plan,
)
from .grouped_block_rvq_forward import PackedGroupedBlockRVQForwardMixin


class PackedGroupedBlockRVQLinear(PackedGroupedBlockRVQForwardMixin, nn.Module):
    def __init__(
        self,
        enc: GroupedBlockRVQEncoding,
        bias: Tensor | None = None,
        *,
        matmul_strategy: str = "per_group",
        matmul_chunk_rows: int | None = None,
        local_palette_group_cols: int | None = None,
        hot_topk: int | None = None,
        hot_score_mode: str = "row_scale_norm",
        hot_min_stage_share: float = 0.0,
        hot_score_threshold_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        if matmul_strategy == "triton_hot_cold_persistent" and local_palette_group_cols is None:
            local_palette_group_cols = 256
        if matmul_strategy not in {"per_group", "per_group_fast", "full_weight", "full_weight_fast", "full_weight_hot", "full_weight_hot_v2", "chunked_weight", "local_palette", "triton_hot_cold_persistent", "stage_local_hot_cold", "stage_local_hot_cold_b1", "stage_local_hot_cold_b2", "stage_local_hot_cold_b3", "stagewise_einsum", "triton_block_rvq", "stacked_matmul"}:
            raise ValueError(f"unsupported matmul_strategy: {matmul_strategy}")
        if matmul_strategy == "chunked_weight" and (matmul_chunk_rows is None or int(matmul_chunk_rows) <= 0):
            raise ValueError("chunked_weight requires a positive matmul_chunk_rows")
        if matmul_strategy in {"local_palette", "triton_hot_cold_persistent"} and (local_palette_group_cols is None or int(local_palette_group_cols) <= 0):
            raise ValueError(f"{matmul_strategy} requires a positive local_palette_group_cols")
        if matmul_strategy == "full_weight_hot" and (hot_topk is None or int(hot_topk) <= 0):
            raise ValueError("full_weight_hot requires a positive hot_topk")
        if matmul_strategy == "full_weight_hot_v2" and (hot_topk is None or int(hot_topk) <= 0):
            raise ValueError("full_weight_hot_v2 requires a positive hot_topk")
        if matmul_strategy == "triton_hot_cold_persistent" and (hot_topk is None or int(hot_topk) <= 0):
            raise ValueError("triton_hot_cold_persistent requires a positive hot_topk")
        if matmul_strategy == "stage_local_hot_cold" and (hot_topk is None or int(hot_topk) <= 0):
            raise ValueError("stage_local_hot_cold requires a positive hot_topk")
        if matmul_strategy == "stage_local_hot_cold_b1" and (hot_topk is None or int(hot_topk) <= 0):
            raise ValueError("stage_local_hot_cold_b1 requires a positive hot_topk")
        if matmul_strategy == "stage_local_hot_cold_b2" and (hot_topk is None or int(hot_topk) <= 0):
            raise ValueError("stage_local_hot_cold_b2 requires a positive hot_topk")
        if matmul_strategy == "stage_local_hot_cold_b3" and (hot_topk is None or int(hot_topk) <= 0):
            raise ValueError("stage_local_hot_cold_b3 requires a positive hot_topk")
        self.in_features = int(enc.original_shape[1])
        self.out_features = int(enc.original_shape[0])
        self.groups = nn.ModuleList([PackedBlockRVQGroup(group) for group in enc.groups])
        self.row_slices = tuple(enc.row_slices)
        self.matmul_strategy = matmul_strategy
        self.matmul_chunk_rows = None if matmul_chunk_rows is None else int(matmul_chunk_rows)
        self.local_palette_group_cols = None if local_palette_group_cols is None else int(local_palette_group_cols)
        self.hot_topk = None if hot_topk is None else int(hot_topk)
        self.hot_score_mode = hot_score_mode
        self.hot_min_stage_share = float(hot_min_stage_share)
        self.hot_score_threshold_ratio = float(hot_score_threshold_ratio)
        self._local_palette_plan: list[dict[str, Tensor | int]] | None = None
        self._local_palette_plan_device: torch.device | None = None
        self._local_palette_row_scale: Tensor | None = None
        self._local_hotprefix_plan: list[dict[str, Tensor | int]] | None = None
        self._local_hotprefix_plan_device: torch.device | None = None
        self._local_hotprefix_row_scale: Tensor | None = None
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.bfloat16).contiguous())
        else:
            self.bias = None

    def reconstruct_weight(self, out_dtype: torch.dtype = torch.bfloat16) -> Tensor:
        parts = [group.reconstruct_weight(out_dtype=out_dtype) for group in self.groups]
        return torch.cat(parts, dim=0)

    def reconstruct_weight_fast(self, out_dtype: torch.dtype = torch.bfloat16) -> Tensor:
        parts = [group.reconstruct_weight_fast(out_dtype=out_dtype) for group in self.groups]
        return torch.cat(parts, dim=0)

    def reconstruct_weight_hot(self, out_dtype: torch.dtype = torch.bfloat16) -> Tensor:
        parts = [
            group.reconstruct_weight_hot(
                out_dtype=out_dtype,
                hot_topk=int(self.hot_topk),
                score_mode=self.hot_score_mode,
                min_stage_share=self.hot_min_stage_share,
                score_threshold_ratio=self.hot_score_threshold_ratio,
            )
            for group in self.groups
        ]
        return torch.cat(parts, dim=0)

    def reconstruct_weight_hot_v2(self, out_dtype: torch.dtype = torch.bfloat16) -> Tensor:
        parts = [
            group.reconstruct_weight_hot_v2(
                out_dtype=out_dtype,
                hot_topk=int(self.hot_topk),
                score_mode=self.hot_score_mode,
                min_stage_share=self.hot_min_stage_share,
                score_threshold_ratio=self.hot_score_threshold_ratio,
            )
            for group in self.groups
        ]
        return torch.cat(parts, dim=0)

    def reconstruct_weight_norm(self, out_dtype: torch.dtype = torch.float16) -> Tensor:
        parts = [group.reconstruct_weight_norm(out_dtype=out_dtype) for group in self.groups]
        return torch.cat(parts, dim=0)

    def _local_palette_group_rows(self) -> int:
        return max((row1 - row0) for row0, row1 in self.row_slices)

    def _local_palette_state(self) -> tuple[list[dict[str, Tensor | int]], Tensor]:
        device = self.groups[0].row_scale.device
        if self._local_palette_plan is None or self._local_palette_plan_device != device:
            routed_norm = self.reconstruct_weight_norm(out_dtype=torch.float16)
            self._local_palette_plan = build_grouped_local_plan(
                routed_norm,
                self._local_palette_group_rows(),
                int(self.local_palette_group_cols),
            )
            self._local_palette_row_scale = torch.cat([group.row_scale for group in self.groups], dim=0).contiguous()
            self._local_palette_plan_device = device
        return self._local_palette_plan, self._local_palette_row_scale

    def _local_hotprefix_state(self) -> tuple[list[dict[str, Tensor | int]], Tensor]:
        device = self.groups[0].row_scale.device
        if self._local_hotprefix_plan is None or self._local_hotprefix_plan_device != device:
            routed_norm = torch.cat(
                [group.reconstruct_weight_fast_norm(out_dtype=torch.bfloat16) for group in self.groups],
                dim=0,
            )
            plan_cpu = build_grouped_local_hotprefix_plan(
                routed_norm.cpu(),
                self._local_palette_group_rows(),
                int(self.local_palette_group_cols),
            )
            self._local_hotprefix_plan = [
                {
                    **item,
                    "palette": item["palette"].to(device),
                    "local_idx": item["local_idx"].to(device),
                }
                for item in plan_cpu
            ]
            self._local_hotprefix_row_scale = torch.cat([group.row_scale for group in self.groups], dim=0).to(torch.bfloat16).contiguous()
            self._local_hotprefix_plan_device = device
        return self._local_hotprefix_plan, self._local_hotprefix_row_scale

    def _chunked_linear(self, x_flat: Tensor) -> Tensor:
        outputs = []
        weight_parts = []
        bias_parts = []
        rows_accum = 0
        for (row0, row1), group in zip(self.row_slices, self.groups):
            weight_parts.append(group.reconstruct_weight(out_dtype=x_flat.dtype))
            if self.bias is not None:
                bias_parts.append(self.bias[row0:row1].to(x_flat.dtype))
            rows_accum += row1 - row0
            if rows_accum >= int(self.matmul_chunk_rows):
                weight = torch.cat(weight_parts, dim=0)
                bias = None if self.bias is None else torch.cat(bias_parts, dim=0)
                outputs.append(F.linear(x_flat, weight, bias))
                weight_parts.clear()
                bias_parts.clear()
                rows_accum = 0
        if weight_parts:
            weight = torch.cat(weight_parts, dim=0)
            bias = None if self.bias is None else torch.cat(bias_parts, dim=0)
            outputs.append(F.linear(x_flat, weight, bias))
        return torch.cat(outputs, dim=-1)

    def _stacked_group_linear(self, x_flat: Tensor) -> Tensor:
        out = torch.empty((x_flat.shape[0], self.out_features), device=x_flat.device, dtype=x_flat.dtype)
        buckets: dict[int, list[tuple[int, int, PackedBlockRVQGroup]]] = {}
        for (row0, row1), group in zip(self.row_slices, self.groups):
            buckets.setdefault(row1 - row0, []).append((row0, row1, group))
        x_batch = x_flat.unsqueeze(0)
        for rows, items in buckets.items():
            if len(items) == 1:
                row0, row1, group = items[0]
                bias = None if self.bias is None else self.bias[row0:row1].to(x_flat.dtype)
                out[:, row0:row1] = F.linear(x_flat, group.reconstruct_weight(out_dtype=x_flat.dtype), bias)
                continue
            weights = torch.stack(
                [group.reconstruct_weight(out_dtype=x_flat.dtype) for _, _, group in items],
                dim=0,
            )
            bucket_out = torch.matmul(x_batch, weights.transpose(1, 2).contiguous())
            if self.bias is not None:
                bias = torch.stack(
                    [self.bias[row0:row1].to(x_flat.dtype) for row0, row1, _ in items],
                    dim=0,
                )
                bucket_out = bucket_out + bias.unsqueeze(1)
            for idx, (row0, row1, _) in enumerate(items):
                out[:, row0:row1] = bucket_out[idx]
        return out
