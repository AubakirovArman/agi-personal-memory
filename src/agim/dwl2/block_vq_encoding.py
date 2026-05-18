from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .block_vq_kmeans import _id_storage_bytes
from .block_vq_transforms import (
    _inverse_polar_transform,
    _sign_correction_matrix,
    _transform_matrix,
    _unpack_sign_bits,
)


@dataclass
class BlockRVQEncoding:
    stage_ids: tuple[Tensor, ...]
    codebooks: tuple[Tensor, ...]
    stage_value_dims: tuple[int, ...]
    stages_per_split: tuple[int, ...]
    stage_scales: Tensor | None
    residual_correction: str
    residual_signs: Tensor | None
    residual_scale: Tensor | None
    row_scale: Tensor
    block_scale: Tensor | None
    transform_kind: str
    transform_matrix: Tensor | None
    transform_bias: Tensor | None
    product_splits: int
    original_shape: tuple[int, int]
    padded_cols: int
    block_size: int
    sample_rel_mse: float

    @property
    def num_stages(self) -> int:
        return len(self.codebooks)

    @property
    def stage_shape(self) -> tuple[int, int]:
        return tuple(int(x) for x in self.stage_ids[0].shape)

    def reconstruct(self, out_dtype: torch.dtype | None = None) -> Tensor:
        out_dtype = out_dtype or torch.float32
        rows, blocks_per_row = self.stage_shape
        flat_blocks = rows * blocks_per_row
        recon = torch.zeros(flat_blocks, self.block_size, dtype=torch.float32, device=self.row_scale.device)
        stage_scales = None if self.stage_scales is None else self.stage_scales.to(torch.float32)
        for idx, (ids, codebook) in enumerate(zip(self.stage_ids, self.codebooks)):
            stage = codebook[ids.reshape(-1).long()].to(torch.float32)
            if stage_scales is not None:
                stage = stage * stage_scales[idx]
            recon += stage
        if self.residual_correction != "none" and self.residual_signs is not None and self.residual_scale is not None:
            signs = _unpack_sign_bits(self.residual_signs, self.block_size).to(torch.float32)
            signs = signs * 2.0 - 1.0
            correction = signs * self.residual_scale.reshape(-1, 1).to(torch.float32)
            correction = correction @ _sign_correction_matrix(self.block_size, device=recon.device)
            recon = recon + correction
        if self.block_scale is not None:
            recon = recon * self.block_scale.reshape(-1, 1).to(torch.float32)
        if self.transform_kind == "polar":
            recon = _inverse_polar_transform(recon)
        else:
            transform = self.transform_matrix
            if transform is None:
                transform = _transform_matrix(self.transform_kind, self.block_size, device=recon.device)
            if transform is not None:
                recon = recon @ transform
        if self.transform_bias is not None:
            recon = recon + self.transform_bias.to(torch.float32)
        w_norm = recon.view(rows, blocks_per_row * self.block_size)[:, : self.original_shape[1]]
        return (w_norm.to(out_dtype) * self.row_scale.to(out_dtype))

    def storage_bytes(self) -> int:
        total = 0
        for ids, codebook, value_dim in zip(self.stage_ids, self.codebooks, self.stage_value_dims):
            total += ids.numel() * _id_storage_bytes(int(codebook.shape[0]))
            total += int(codebook.shape[0]) * int(value_dim) * 2
        if self.stage_scales is not None:
            total += self.stage_scales.numel() * self.stage_scales.element_size()
        if self.residual_signs is not None:
            total += self.residual_signs.numel() * self.residual_signs.element_size()
        if self.residual_scale is not None:
            total += self.residual_scale.numel() * self.residual_scale.element_size()
        if self.block_scale is not None:
            total += self.block_scale.numel() * self.block_scale.element_size()
        if self.transform_matrix is not None:
            total += self.transform_matrix.numel() * self.transform_matrix.element_size()
        if self.transform_bias is not None:
            total += self.transform_bias.numel() * self.transform_bias.element_size()
        total += self.row_scale.numel() * 2
        return total

    def bits_per_weight(self) -> float:
        rows, cols = self.original_shape
        return 8.0 * self.storage_bytes() / max(rows * cols, 1)


@dataclass
class GroupedBlockRVQEncoding:
    groups: tuple[BlockRVQEncoding, ...]
    row_slices: tuple[tuple[int, int], ...]
    original_shape: tuple[int, int]

    @property
    def sample_rel_mse(self) -> float:
        total_rows = float(sum(group.original_shape[0] for group in self.groups))
        if total_rows <= 0.0:
            return 0.0
        return float(sum(group.sample_rel_mse * group.original_shape[0] for group in self.groups) / total_rows)

    def reconstruct(self, out_dtype: torch.dtype | None = None) -> Tensor:
        parts = [group.reconstruct(out_dtype=out_dtype) for group in self.groups]
        return torch.cat(parts, dim=0)

    def storage_bytes(self) -> int:
        return sum(group.storage_bytes() for group in self.groups)

    def bits_per_weight(self) -> float:
        rows, cols = self.original_shape
        return 8.0 * self.storage_bytes() / max(rows * cols, 1)
