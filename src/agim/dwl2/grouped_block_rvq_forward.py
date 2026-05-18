from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .full_layer_tiled_runtime import (
    full_layer_grouped_local_hotprefix_matmul,
    full_layer_grouped_local_matmul,
)


class PackedGroupedBlockRVQForwardMixin:
    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        if self.matmul_strategy == "full_weight":
            # Avoid dozens of tiny GEMM launches when attention-side row groups are small.
            weight = self.reconstruct_weight(out_dtype=x_flat.dtype)
            bias = None if self.bias is None else self.bias.to(x_flat.dtype)
            out = F.linear(x_flat, weight, bias)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "full_weight_fast":
            # M20: same as full_weight but uses cached fast reconstruct per group.
            weight = self.reconstruct_weight_fast(out_dtype=x_flat.dtype)
            bias = None if self.bias is None else self.bias.to(x_flat.dtype)
            out = F.linear(x_flat, weight, bias)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "full_weight_hot":
            weight = self.reconstruct_weight_hot(out_dtype=x_flat.dtype)
            bias = None if self.bias is None else self.bias.to(x_flat.dtype)
            out = F.linear(x_flat, weight, bias)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "full_weight_hot_v2":
            weight = self.reconstruct_weight_hot_v2(out_dtype=x_flat.dtype)
            bias = None if self.bias is None else self.bias.to(x_flat.dtype)
            out = F.linear(x_flat, weight, bias)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "chunked_weight":
            out = self._chunked_linear(x_flat)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "local_palette":
            plan, row_scale = self._local_palette_state()
            out = full_layer_grouped_local_matmul(x_flat, plan, row_scale)
            if self.bias is not None:
                out = out + self.bias.to(out.dtype)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "triton_hot_cold_persistent":
            plan, row_scale = self._local_hotprefix_state()
            out = full_layer_grouped_local_hotprefix_matmul(x_flat, plan, row_scale, hot_size=int(self.hot_topk))
            if self.bias is not None:
                out = out + self.bias.to(out.dtype)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "stage_local_hot_cold":
            parts = []
            for (row0, row1), group in zip(self.row_slices, self.groups):
                bias = None if self.bias is None else self.bias[row0:row1].to(x_flat.dtype)
                part = group.triton_stage_local_hot_cold_matmul(
                    x_flat,
                    hot_topk=int(self.hot_topk),
                    score_mode=self.hot_score_mode,
                    min_stage_share=self.hot_min_stage_share,
                    score_threshold_ratio=self.hot_score_threshold_ratio,
                )
                if bias is not None:
                    part = part + bias
                parts.append(part)
            out = torch.cat(parts, dim=-1)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "stage_local_hot_cold_b1":
            parts = []
            for (row0, row1), group in zip(self.row_slices, self.groups):
                bias = None if self.bias is None else self.bias[row0:row1].to(x_flat.dtype)
                part = group.triton_stage_local_hot_palette_matmul(
                    x_flat,
                    hot_topk=int(self.hot_topk),
                    score_mode=self.hot_score_mode,
                    min_stage_share=self.hot_min_stage_share,
                    score_threshold_ratio=self.hot_score_threshold_ratio,
                )
                if bias is not None:
                    part = part + bias
                parts.append(part)
            out = torch.cat(parts, dim=-1)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "stage_local_hot_cold_b2":
            parts = []
            for (row0, row1), group in zip(self.row_slices, self.groups):
                bias = None if self.bias is None else self.bias[row0:row1].to(x_flat.dtype)
                part = group.triton_stage_local_hot_palette_b2_matmul(
                    x_flat,
                    hot_topk=int(self.hot_topk),
                    score_mode=self.hot_score_mode,
                    min_stage_share=self.hot_min_stage_share,
                    score_threshold_ratio=self.hot_score_threshold_ratio,
                )
                if bias is not None:
                    part = part + bias
                parts.append(part)
            out = torch.cat(parts, dim=-1)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "stage_local_hot_cold_b3":
            parts = []
            for (row0, row1), group in zip(self.row_slices, self.groups):
                bias = None if self.bias is None else self.bias[row0:row1].to(x_flat.dtype)
                part = group.triton_stage_local_hot_palette_b2_matmul(
                    x_flat,
                    hot_topk=int(self.hot_topk),
                    score_mode=self.hot_score_mode,
                    min_stage_share=0.0,
                    score_threshold_ratio=self.hot_score_threshold_ratio,
                )
                if bias is not None:
                    part = part + bias
                parts.append(part)
            out = torch.cat(parts, dim=-1)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "stagewise_einsum":
            parts = []
            for (row0, row1), group in zip(self.row_slices, self.groups):
                bias = None if self.bias is None else self.bias[row0:row1].to(x_flat.dtype)
                part = group.stagewise_block_matmul(x_flat)
                if bias is not None:
                    part = part + bias
                parts.append(part)
            out = torch.cat(parts, dim=-1)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "triton_block_rvq":
            parts = []
            for (row0, row1), group in zip(self.row_slices, self.groups):
                bias = None if self.bias is None else self.bias[row0:row1].to(x_flat.dtype)
                if x_flat.is_cuda and group.supports_triton_block_matmul():
                    part = group.triton_block_matmul(x_flat)
                else:
                    part = F.linear(x_flat, group.reconstruct_weight(out_dtype=x_flat.dtype), bias=None)
                if bias is not None:
                    part = part + bias
                parts.append(part)
            out = torch.cat(parts, dim=-1)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "stacked_matmul":
            out = self._stacked_group_linear(x_flat)
            return out.reshape(*orig_shape[:-1], self.out_features)
        if self.matmul_strategy == "per_group_fast":
            # M20: per-group with cached bf16 codebooks and reusable per-group reconstruct buffers.
            parts = []
            for (row0, row1), group in zip(self.row_slices, self.groups):
                bias = None if self.bias is None else self.bias[row0:row1].to(x_flat.dtype)
                weight = group.reconstruct_weight_fast(out_dtype=x_flat.dtype)
                parts.append(F.linear(x_flat, weight, bias))
            out = torch.cat(parts, dim=-1)
            return out.reshape(*orig_shape[:-1], self.out_features)
        parts = []
        for (row0, row1), group in zip(self.row_slices, self.groups):
            bias = None if self.bias is None else self.bias[row0:row1].to(x_flat.dtype)
            weight = group.reconstruct_weight(out_dtype=x_flat.dtype)
            parts.append(F.linear(x_flat, weight, bias))
        out = torch.cat(parts, dim=-1)
        return out.reshape(*orig_shape[:-1], self.out_features)
