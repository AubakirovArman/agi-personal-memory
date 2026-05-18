from __future__ import annotations

import math

import torch
from torch import Tensor


def _hadamard_matrix(n: int, *, device: torch.device) -> Tensor:
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("hadamard transform requires power-of-two block_size")
    mat = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while mat.shape[0] < n:
        mat = torch.cat(
            (
                torch.cat((mat, mat), dim=1),
                torch.cat((mat, -mat), dim=1),
            ),
            dim=0,
        )
    return mat / math.sqrt(float(n))


def _rand_hadamard_matrix(n: int, *, device: torch.device) -> Tensor:
    base = _hadamard_matrix(n, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    perm = torch.randperm(n, generator=generator, device=device)
    signs = torch.where(
        torch.randint(0, 2, (n,), generator=generator, device=device, dtype=torch.int64) > 0,
        torch.ones(n, device=device, dtype=torch.float32),
        -torch.ones(n, device=device, dtype=torch.float32),
    )
    return base[perm] * signs.unsqueeze(1)


def _dct_matrix(n: int, *, device: torch.device) -> Tensor:
    idx = torch.arange(n, device=device, dtype=torch.float32)
    k = idx[:, None]
    mat = torch.cos(math.pi / float(n) * (idx + 0.5)[None, :] * k)
    mat[0] *= math.sqrt(1.0 / float(n))
    if n > 1:
        mat[1:] *= math.sqrt(2.0 / float(n))
    return mat


def _transform_matrix(kind: str, block_size: int, *, device: torch.device) -> Tensor | None:
    if kind == "none":
        return None
    if kind == "dct":
        return _dct_matrix(block_size, device=device)
    if kind == "hadamard":
        return _hadamard_matrix(block_size, device=device)
    if kind == "rand_hadamard":
        return _rand_hadamard_matrix(block_size, device=device)
    if kind == "polar":
        return None
    if kind == "pca":
        return None
    raise ValueError("transform_kind must be one of: none, dct, hadamard, rand_hadamard, polar, pca")


def _polar_transform(blocks: Tensor) -> Tensor:
    if blocks.ndim != 2:
        raise ValueError("polar transform expects 2D blocks")
    if blocks.shape[1] < 1 or (blocks.shape[1] & (blocks.shape[1] - 1)) != 0:
        raise ValueError("polar transform requires power-of-two block_size")
    radii = blocks.to(torch.float32)
    angles: list[Tensor] = []
    while radii.shape[1] > 1:
        pair = radii.view(radii.shape[0], -1, 2)
        x = pair[..., 0]
        y = pair[..., 1]
        radii = torch.sqrt(x.square() + y.square())
        angles.append(torch.atan2(y, x) / math.pi)
    return torch.cat([radii] + list(reversed(angles)), dim=1)


def _inverse_polar_transform(blocks: Tensor) -> Tensor:
    if blocks.ndim != 2:
        raise ValueError("inverse polar transform expects 2D blocks")
    if blocks.shape[1] < 1 or (blocks.shape[1] & (blocks.shape[1] - 1)) != 0:
        raise ValueError("inverse polar transform requires power-of-two block_size")
    count = int(blocks.shape[1])
    widths = []
    width = 1
    while width < count:
        widths.append(width)
        width *= 2
    current = blocks[:, :1].to(torch.float32)
    offset = 1
    for width in widths:
        angles = blocks[:, offset:offset + width].to(torch.float32)
        offset += width
        current = torch.stack(
            (
                current * torch.cos(math.pi * angles),
                current * torch.sin(math.pi * angles),
            ),
            dim=-1,
        ).reshape(blocks.shape[0], width * 2)
    return current


def _sign_correction_matrix(block_size: int, *, device: torch.device) -> Tensor:
    return _rand_hadamard_matrix(block_size, device=device)


def _pack_sign_bits(signs: Tensor) -> Tensor:
    if signs.ndim != 2:
        raise ValueError("sign packing expects a 2D tensor")
    signs = signs.to(torch.bool)
    words = (signs.shape[1] + 31) // 32
    packed = torch.zeros(signs.shape[0], words, dtype=torch.int32, device=signs.device)
    for word_idx in range(words):
        start = word_idx * 32
        end = min(start + 32, signs.shape[1])
        word = torch.zeros(signs.shape[0], dtype=torch.int64, device=signs.device)
        for bit_idx in range(end - start):
            word = word | (signs[:, start + bit_idx].to(torch.int64) << bit_idx)
        packed[:, word_idx] = word.to(torch.int32)
    return packed


def _unpack_sign_bits(packed: Tensor, width: int) -> Tensor:
    if packed.ndim != 2:
        raise ValueError("sign unpacking expects a 2D tensor")
    out = torch.empty(packed.shape[0], width, dtype=torch.bool, device=packed.device)
    for col in range(width):
        word_idx = col // 32
        bit_idx = col % 32
        word = packed[:, word_idx].to(torch.int64)
        out[:, col] = ((word >> bit_idx) & 1).bool()
    return out


def _fit_pca_transform(sample_blocks: Tensor) -> tuple[Tensor, Tensor]:
    sample_blocks = sample_blocks.to(torch.float32)
    bias = sample_blocks.mean(dim=0, keepdim=True)
    centered = sample_blocks - bias
    cov = centered.t() @ centered / max(int(centered.shape[0]) - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    basis = eigvecs[:, order]
    transform = basis.t().contiguous()
    return transform, bias.contiguous()
