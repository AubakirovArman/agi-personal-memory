from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    ],
    key=["m_size", "n_size", "blocks_per_row"],
)
@triton.jit
def _rvq_group_linear_kernel(
    x_ptr,
    stage_ids_ptr,
    codebooks_ptr,
    row_scale_ptr,
    out_ptr,
    m_size,
    n_size,
    blocks_per_row,
    stride_xm,
    stride_xk,
    stride_ids_s,
    stride_ids_n,
    stride_ids_b,
    stride_cb_s,
    stride_cb_c,
    stride_cb_k,
    stride_rn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_SIZE)
    row_mask = offs_n < n_size
    rho = tl.load(row_scale_ptr + offs_n * stride_rn, mask=row_mask, other=1.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for block_idx in range(0, blocks_per_row):
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (block_idx * BLOCK_SIZE + offs_k[None, :]) * stride_xk,
            mask=offs_m[:, None] < m_size,
            other=0.0,
        )
        weight_block = tl.zeros((BLOCK_N, BLOCK_SIZE), dtype=tl.float32)
        for stage_idx in range(NUM_STAGES):
            ids = tl.load(
                stage_ids_ptr + stage_idx * stride_ids_s + offs_n * stride_ids_n + block_idx * stride_ids_b,
                mask=row_mask,
                other=0,
            )
            stage = tl.load(
                codebooks_ptr
                + stage_idx * stride_cb_s
                + ids[:, None].to(tl.int32) * stride_cb_c
                + offs_k[None, :] * stride_cb_k,
                mask=row_mask[:, None],
                other=0.0,
            )
            weight_block += stage.to(tl.float32)
        acc += tl.dot(x, tl.trans(weight_block.to(x.dtype)))

    acc = acc * rho[None, :].to(acc.dtype)
    out_mask = (offs_m[:, None] < m_size) & (offs_n[None, :] < n_size)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=out_mask,
    )


def rvq_group_linear_matmul(
    x: torch.Tensor,
    stage_ids: torch.Tensor,
    codebooks: torch.Tensor,
    row_scale: torch.Tensor,
    *,
    autotuned: bool = True,
    block_m: int = 32,
    block_n: int = 32,
) -> torch.Tensor:
    if not x.is_cuda or not stage_ids.is_cuda or not codebooks.is_cuda or not row_scale.is_cuda:
        raise ValueError("rvq_group_linear_matmul expects CUDA tensors")
    if stage_ids.ndim != 3:
        raise ValueError(f"expected stage_ids to have shape [stages, rows, blocks], got {tuple(stage_ids.shape)}")
    if codebooks.ndim != 3:
        raise ValueError(f"expected codebooks to have shape [stages, codebook, block], got {tuple(codebooks.shape)}")
    if stage_ids.shape[0] != codebooks.shape[0]:
        raise ValueError(f"stage mismatch: {stage_ids.shape[0]} != {codebooks.shape[0]}")
    if codebooks.shape[2] != 32:
        raise ValueError(f"rvq_group_linear_matmul currently requires block_size=32, got {codebooks.shape[2]}")
    original_shape = x.shape
    x_2d = x.reshape(-1, original_shape[-1]).contiguous()
    stage_ids_3d = stage_ids.contiguous()
    codebooks_3d = codebooks.contiguous()
    row = row_scale.reshape(-1).contiguous()
    m_size, k_size = x_2d.shape
    num_stages, n_size, blocks_per_row = stage_ids_3d.shape
    if blocks_per_row * int(codebooks_3d.shape[2]) != k_size:
        raise ValueError(f"Feature mismatch: expected {blocks_per_row * int(codebooks_3d.shape[2])}, got {k_size}")
    out = torch.empty((m_size, n_size), device=x.device, dtype=x.dtype)
    grid = (lambda meta: (
        triton.cdiv(m_size, meta["BLOCK_M"]),
        triton.cdiv(n_size, meta["BLOCK_N"]),
    )) if autotuned else (triton.cdiv(m_size, block_m), triton.cdiv(n_size, block_n))
    if autotuned:
        _rvq_group_linear_kernel[grid](
            x_2d, stage_ids_3d, codebooks_3d, row, out, m_size, n_size,
            blocks_per_row, x_2d.stride(0), x_2d.stride(1),
            stage_ids_3d.stride(0), stage_ids_3d.stride(1),
            stage_ids_3d.stride(2), codebooks_3d.stride(0),
            codebooks_3d.stride(1), codebooks_3d.stride(2),
            row.stride(0), out.stride(0), out.stride(1),
            BLOCK_SIZE=32, NUM_STAGES=num_stages,
        )
    else:
        _rvq_group_linear_kernel[grid](
            x_2d, stage_ids_3d, codebooks_3d, row, out, m_size, n_size,
            blocks_per_row, x_2d.stride(0), x_2d.stride(1),
            stage_ids_3d.stride(0), stage_ids_3d.stride(1),
            stage_ids_3d.stride(2), codebooks_3d.stride(0),
            codebooks_3d.stride(1), codebooks_3d.stride(2),
            row.stride(0), out.stride(0), out.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_SIZE=32,
            NUM_STAGES=num_stages, num_warps=4, num_stages=2,
        )
    return out.reshape(*original_shape[:-1], n_size)
