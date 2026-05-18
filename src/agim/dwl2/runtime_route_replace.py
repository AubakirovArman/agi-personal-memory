from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .eager_layers import EagerBf16Linear, EagerFp8Linear
from .grouped_runtime import GroupedLocalRouteLinear
from .id_route_layers import (
    AdaptiveFusedIDRouteLinear,
    CachedPackedIDRouteLinear,
    FusedIDRouteLinear,
    PackedIDRouteLinear,
)
from .runtime_common import _load_shape_runtime_policy, _resolve_shape_policy, _set_submodule
from .runtime_constants import TARGET_LINEAR_SUFFIXES
from .runtime_quantize import quantize_linear_to_packed


@torch.no_grad()
def replace_packed_id_route_layers(
    model: nn.Module,
    target_suffixes: tuple[str, ...] = TARGET_LINEAR_SUFFIXES,
    l_max: int = 12,
    sample_limit: int = 2_000_000,
    runtime_cls: type[nn.Module] = PackedIDRouteLinear,
    runtime_kwargs: dict[str, object] | None = None,
    shape_policy_json: str | Path | None = None,
) -> list[dict[str, float]]:
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.endswith(target_suffixes)
    ]
    shape_policy = _load_shape_runtime_policy(shape_policy_json)
    stats = []
    default_runtime_kwargs = dict(runtime_kwargs or {})
    for name, module in targets:
        layer_runtime_kwargs: dict[str, object] | None = None
        selected = _resolve_shape_policy(shape_policy, name)
        if selected is None:
            layer_runtime_cls = runtime_cls
            layer_runtime_kwargs = dict(default_runtime_kwargs)
            tensor_name = f"{name}.weight"
            if runtime_cls is FusedIDRouteLinear:
                runtime_decision = "global_id_triton"
            elif runtime_cls is AdaptiveFusedIDRouteLinear:
                runtime_decision = "adaptive_global_id"
            elif runtime_cls is CachedPackedIDRouteLinear:
                runtime_decision = "cached_packed"
            elif runtime_cls is PackedIDRouteLinear:
                runtime_decision = "packed_materialize"
            elif runtime_cls is EagerBf16Linear:
                runtime_decision = "eager_bf16"
            elif runtime_cls is EagerFp8Linear:
                runtime_decision = "eager_fp8"
            else:
                runtime_decision = runtime_cls.__name__
        else:
            tensor_name, layer_runtime_kwargs = selected
            layer_runtime_cls = GroupedLocalRouteLinear
            runtime_decision = "local_palette_grouped"
        packed, layer_stats = quantize_linear_to_packed(
            module,
            l_max=l_max,
            sample_limit=sample_limit,
            runtime_cls=layer_runtime_cls,
            runtime_kwargs=layer_runtime_kwargs,
        )
        _set_submodule(model, name, packed)
        # Explicitly drop the original Linear's weight/bias so the bf16 buffers
        # are released back to the allocator before the next iteration. Without
        # this the loop variable keeps the old W-sized tensor alive and the peak
        # VRAM doubles (old nn.Linear.weight + new packed/eager weight).
        try:
            if hasattr(module, "_parameters"):
                module._parameters["weight"] = None
                module._parameters["bias"] = None
        except (AttributeError, TypeError, KeyError):
            pass
        item: dict[str, float | str | int] = {
            "name": name,
            "tensor_name": tensor_name,
            "runtime_decision": runtime_decision,
            **layer_stats,
        }
        if layer_runtime_kwargs is not None:
            item.update(
                {
                    key: value
                    for key, value in layer_runtime_kwargs.items()
                    if isinstance(value, (bool, int, float, str))
                }
            )
        stats.append(item)
    return stats
def replace_with_deployment_runtime(
    model: nn.Module,
    target_suffixes: tuple[str, ...] = TARGET_LINEAR_SUFFIXES,
    l_max: int = 12,
    sample_limit: int = 2_000_000,
    shape_policy_json: str | Path | None = None,
    cache_max_mb: int = 128,
) -> list[dict[str, float]]:
    return replace_packed_id_route_layers(
        model,
        target_suffixes=target_suffixes,
        l_max=l_max,
        sample_limit=sample_limit,
        runtime_cls=CachedPackedIDRouteLinear,
        runtime_kwargs={"max_cache_bytes": max(int(cache_max_mb), 0) * 2**20},
        shape_policy_json=shape_policy_json,
    )


@torch.no_grad()
def replace_with_eager_bf16(
    model: nn.Module,
    target_suffixes: tuple[str, ...] = TARGET_LINEAR_SUFFIXES,
    l_max: int = 12,
    sample_limit: int = 2_000_000,
    shape_policy_json: str | Path | None = None,
    target_dtype: torch.dtype = torch.bfloat16,
) -> list[dict[str, float]]:
    """Route-decode weights once, store as plain bf16 buffers.

    Preserves the 3-bit disk/codebook PPL signature but matches the baseline
    nn.Linear runtime footprint exactly (speed + VRAM).
    """
    return replace_packed_id_route_layers(
        model,
        target_suffixes=target_suffixes,
        l_max=l_max,
        sample_limit=sample_limit,
        runtime_cls=EagerBf16Linear,
        runtime_kwargs={"target_dtype": target_dtype},
        shape_policy_json=shape_policy_json,
    )


@torch.no_grad()
def replace_with_eager_fp8(
    model: nn.Module,
    target_suffixes: tuple[str, ...] = TARGET_LINEAR_SUFFIXES,
    l_max: int = 12,
    sample_limit: int = 2_000_000,
    shape_policy_json: str | Path | None = None,
) -> list[dict[str, float]]:
    """Route-decode weights then store as fp8_e4m3 with per-row scale (~50% of bf16 VRAM).

    Use when VRAM is the bottleneck. Compute uses ``torch._scaled_mm`` with
    rowwise scaling and bf16 output. Per-token x scaling is dynamic.
    """
    return replace_packed_id_route_layers(
        model,
        target_suffixes=target_suffixes,
        l_max=l_max,
        sample_limit=sample_limit,
        runtime_cls=EagerFp8Linear,
        runtime_kwargs=None,
        shape_policy_json=shape_policy_json,
    )


@torch.no_grad()
def replace_with_eager_hybrid(
    model: nn.Module,
    fp8_suffixes: tuple[str, ...] = (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "mlp.gate_proj", "mlp.up_proj",
    ),
    bf16_suffixes: tuple[str, ...] = ("self_attn.o_proj", "mlp.down_proj"),
    l_max: int = 12,
    sample_limit: int = 2_000_000,
    target_dtype: torch.dtype = torch.bfloat16,
) -> list[dict[str, float]]:
    """Hybrid eager runtime: FP8 for attention QKV + MLP gate/up, BF16 for residual writebacks.

    Rationale: ``o_proj`` and ``down_proj`` write directly into the residual
    stream where small numerical errors compound across layers. Keep them in
    bf16 for quality. The other five projections (Q, K, V, gate, up) feed
    nonlinearities (softmax, swiglu) that absorb fp8 noise.
    """
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.endswith(bf16_suffixes):
            targets.append((name, module, "bf16"))
        elif name.endswith(fp8_suffixes):
            targets.append((name, module, "fp8"))
    stats = []
    for name, module, kind in targets:
        if kind == "fp8":
            cls = EagerFp8Linear
            kwargs: dict[str, object] | None = None
            decision = "eager_fp8"
        else:
            cls = EagerBf16Linear
            kwargs = {"target_dtype": target_dtype}
            decision = "eager_bf16"
        packed, layer_stats = quantize_linear_to_packed(
            module, l_max=l_max, sample_limit=sample_limit,
            runtime_cls=cls, runtime_kwargs=kwargs,
        )
        _set_submodule(model, name, packed)
        try:
            if hasattr(module, "_parameters"):
                module._parameters["weight"] = None
                module._parameters["bias"] = None
        except (AttributeError, TypeError, KeyError):
            pass
        stats.append({
            "name": name,
            "tensor_name": f"{name}.weight",
            "runtime_decision": decision,
            **layer_stats,
        })
    return stats
