from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .grouped_runtime import GroupedLocalRouteLinear
from .id_route_layers import AdaptiveFusedIDRouteLinear, CachedPackedIDRouteLinear, FusedIDRouteLinear
from .runtime_common import (
    _load_fused_allowlist,
    _load_shape_runtime_policy,
    _module_name_candidates,
    _resolve_shape_policy,
    _set_submodule,
)
from .runtime_constants import TARGET_LINEAR_SUFFIXES
from .runtime_quantize import quantize_linear_to_packed


@torch.no_grad()
def replace_with_hybrid_runtime(
    model: nn.Module,
    target_suffixes: tuple[str, ...] = TARGET_LINEAR_SUFFIXES,
    l_max: int = 12,
    sample_limit: int = 2_000_000,
    shape_policy_json: str | Path | None = None,
    cache_max_mb: int = 128,
    fused_allowlist_json: str | Path | None = None,
    promoted_runtime_cls: type[nn.Module] = FusedIDRouteLinear,
    promoted_runtime_kwargs: dict[str, object] | None = None,
) -> list[dict[str, float]]:
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.endswith(target_suffixes)
    ]
    shape_policy = _load_shape_runtime_policy(shape_policy_json)
    fused_allowlist = _load_fused_allowlist(fused_allowlist_json)
    cached_runtime_kwargs = {"max_cache_bytes": max(int(cache_max_mb), 0) * 2**20}
    stats = []
    for name, module in targets:
        selected = _resolve_shape_policy(shape_policy, name)
        if selected is not None:
            tensor_name, layer_runtime_kwargs = selected
            layer_runtime_cls = GroupedLocalRouteLinear
            runtime_decision = "local_palette_grouped"
        elif any(candidate in fused_allowlist for candidate in _module_name_candidates(name)):
            tensor_name = f"{name}.weight"
            layer_runtime_kwargs = dict(promoted_runtime_kwargs or {})
            layer_runtime_cls = promoted_runtime_cls
            if promoted_runtime_cls is FusedIDRouteLinear:
                runtime_decision = "promoted_fused_global_id"
            elif promoted_runtime_cls is AdaptiveFusedIDRouteLinear:
                runtime_decision = "promoted_adaptive_global_id"
            else:
                runtime_decision = f"promoted_{promoted_runtime_cls.__name__}"
        else:
            tensor_name = f"{name}.weight"
            layer_runtime_kwargs = dict(cached_runtime_kwargs)
            layer_runtime_cls = CachedPackedIDRouteLinear
            runtime_decision = "cached_packed_default"
        packed, layer_stats = quantize_linear_to_packed(
            module,
            l_max=l_max,
            sample_limit=sample_limit,
            runtime_cls=layer_runtime_cls,
            runtime_kwargs=layer_runtime_kwargs,
        )
        _set_submodule(model, name, packed)
        item: dict[str, float | str | int] = {
            "name": name,
            "tensor_name": tensor_name,
            "runtime_decision": runtime_decision,
            **layer_stats,
        }
        item.update(
            {
                key: value
                for key, value in layer_runtime_kwargs.items()
                if isinstance(value, (bool, int, float, str))
            }
        )
        stats.append(item)
    return stats


# ---------------------------------------------------------------------------
# M21: global variable-stage decoding helpers
# ---------------------------------------------------------------------------
