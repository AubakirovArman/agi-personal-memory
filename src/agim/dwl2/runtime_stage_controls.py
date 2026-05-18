from __future__ import annotations

from torch import nn

from .block_rvq_group import PackedBlockRVQGroup
from .grouped_block_rvq_linear import PackedGroupedBlockRVQLinear

def set_global_effective_stages(model: nn.Module, stages: int) -> int:
    """Set ``effective_stages_per_split`` on every ``PackedBlockRVQGroup``.

    With product_splits=ps and stages_per_split=s, calling this with k means
    "keep the first k stages of each split", effectively running k*ps gather+add
    operations instead of s*ps. Returns the number of groups updated.
    """
    n = 0
    for module in model.modules():
        if isinstance(module, PackedBlockRVQGroup):
            module.set_effective_stages_per_split(int(stages))
            n += 1
    return n


def set_effective_stages_by_name(model: nn.Module, attn_stages: int | None, mlp_stages: int | None) -> dict[str, int]:
    """M22: per-role stage cap. Routes stage cap by parent module name pattern.

    ``.self_attn.`` modules (q/k/v/o) get ``attn_stages``;
    ``.mlp.`` modules (gate/up/down) get ``mlp_stages``.
    Returns counts per bucket.
    """
    counts = {"attn": 0, "mlp": 0, "other": 0}
    for name, module in model.named_modules():
        if not isinstance(module, PackedGroupedBlockRVQLinear):
            continue
        bucket = "attn" if ".self_attn." in name else ("mlp" if ".mlp." in name else "other")
        target = attn_stages if bucket == "attn" else (mlp_stages if bucket == "mlp" else None)
        if target is None:
            continue
        for sub in module.modules():
            if isinstance(sub, PackedBlockRVQGroup):
                sub.set_effective_stages_per_split(int(target))
        counts[bucket] += 1
    return counts


def set_effective_stages_from_map(model: nn.Module, name_to_k: dict[str, int]) -> int:
    """M24: apply per-layer stage cap from a {layer_name: chosen_k} map.

    Returns count of layers actually updated. Names without matches are skipped.
    """
    updated = 0
    for name, module in model.named_modules():
        if not isinstance(module, PackedGroupedBlockRVQLinear):
            continue
        if name not in name_to_k:
            continue
        k = int(name_to_k[name])
        for sub in module.modules():
            if isinstance(sub, PackedBlockRVQGroup):
                sub.set_effective_stages_per_split(k)
        updated += 1
    return updated
