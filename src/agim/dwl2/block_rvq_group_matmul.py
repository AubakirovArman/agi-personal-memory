from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .triton_id_matmul import rvq_group_linear_matmul


class PackedBlockRVQMatmulMixin:
    def supports_stagewise_block_matmul(self) -> bool:
        return (
            self.block_scale is None
            and self.transform_kind == "none"
            and self.transform_matrix is None
            and self.transform_bias is None
            and self.residual_correction == "none"
            and self.residual_signs is None
            and self.residual_scale is None
        )

    def supports_triton_block_matmul(self) -> bool:
        if not self.supports_stagewise_block_matmul():
            return False
        if self.block_size != 32 or self.padded_cols != self.in_features:
            return False
        return all(getattr(self, f"stage_ids_{idx}").dtype == torch.uint8 for idx in range(self.num_stages))

    def _triton_state(self) -> tuple[Tensor, Tensor]:
        if not self.supports_triton_block_matmul():
            raise ValueError("triton block matmul only supports plain uint8, block_size=32 encodings")
        device = self.row_scale.device
        if (
            self._triton_stage_ids is None
            or self._triton_codebooks is None
            or self._triton_cache_device != device
        ):
            self._triton_stage_ids = torch.stack(
                [getattr(self, f"stage_ids_{idx}").contiguous() for idx in range(self.num_stages)],
                dim=0,
            ).contiguous()
            codebooks = []
            for idx in range(self.num_stages):
                codebook = getattr(self, f"codebook_{idx}").contiguous()
                if self.stage_scales is not None:
                    codebook = (codebook.to(torch.float32) * self.stage_scales[idx].to(torch.float32)).to(codebook.dtype)
                codebooks.append(codebook)
            self._triton_codebooks = torch.stack(codebooks, dim=0).contiguous()
            self._triton_cache_device = device
        return self._triton_stage_ids, self._triton_codebooks

    def stagewise_block_matmul(self, x_flat: Tensor) -> Tensor:
        if not self.supports_stagewise_block_matmul():
            raise ValueError("stagewise_block_matmul only supports unnormalized, transform-free encodings")
        rows, blocks_per_row = getattr(self, "stage_ids_0").shape
        if self.padded_cols == self.in_features:
            x_blocks = x_flat.reshape(-1, blocks_per_row, self.block_size)
        else:
            x_padded = F.pad(x_flat, (0, self.padded_cols - self.in_features), value=0.0)
            x_blocks = x_padded.reshape(-1, blocks_per_row, self.block_size)
        out = torch.zeros((x_blocks.shape[0], rows), device=x_blocks.device, dtype=x_blocks.dtype)
        stage_scales = None if self.stage_scales is None else self.stage_scales.to(x_blocks.dtype)
        for idx, (ids, codebook) in enumerate(self._stage_tensors()):
            stage_blocks = codebook[ids.reshape(-1).long()].view(rows, blocks_per_row, self.block_size).to(x_blocks.dtype)
            if stage_scales is not None:
                stage_blocks = stage_blocks * stage_scales[idx]
            out = out + torch.einsum("mbd,rbd->mr", x_blocks, stage_blocks)
        return out * self.row_scale.reshape(1, rows).to(out.dtype)

    def triton_block_matmul(self, x_flat: Tensor) -> Tensor:
        if not x_flat.is_cuda:
            raise ValueError("triton block matmul expects CUDA input")
        stage_ids, codebooks = self._triton_state()
        return rvq_group_linear_matmul(x_flat, stage_ids, codebooks, self.row_scale)
