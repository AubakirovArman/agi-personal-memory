from __future__ import annotations

import math

import torch
from torch import nn

from .calibrate import calibrate_ladder
from .codebook import build_codebook
from .id_route_layers import PackedIDRouteLinear
from .route_encoder import encode_routes, rel_mse


@torch.no_grad()
def quantize_linear_to_packed(
    linear: nn.Linear,
    l_max: int = 12,
    sample_limit: int = 2_000_000,
    runtime_cls: type[nn.Module] = PackedIDRouteLinear,
    runtime_kwargs: dict[str, object] | None = None,
) -> tuple[nn.Module, dict[str, float]]:
    weight = linear.weight.detach()
    row_scale = weight.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = weight / row_scale
    sample = w_norm.flatten()
    if sample.numel() > sample_limit:
        pick = torch.randint(0, sample.numel(), (sample_limit,), device=sample.device)
        sample = sample[pick]
    ladder = calibrate_ladder(
        sample,
        l_max=l_max,
        refine_iters=20,
        pin_top=True,
        top_value=1.0,
        seed="geometric",
    )
    enc = encode_routes(w_norm, ladder, stop_threshold=0.0, l_max=l_max)
    codebook, ids = build_codebook(enc.digits, enc.stop_depth, l_max=l_max)
    packed = runtime_cls.from_encoded(ids, codebook.digits, ladder, row_scale, linear.bias, **(runtime_kwargs or {}))
    stats = {
        "rel_mse": float(rel_mse(weight.float(), packed.reconstruct_weight().float()).item()),
        "unique_routes": int(codebook.size),
        "id_bits": int(math.ceil(math.log2(max(codebook.size, 2)))),
    }
    return packed, stats
