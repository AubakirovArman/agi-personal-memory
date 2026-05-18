"""Runtime linear layers that reconstruct weights from route and block-RVQ encodings.

This module is kept as the public compatibility facade. Implementations live in
smaller modules so each file stays under the project size gate.
"""
from __future__ import annotations

from .block_rvq_group import PackedBlockRVQGroup
from .block_vq import BlockRVQEncoding, GroupedBlockRVQEncoding, encode_grouped_block_residual_vq
from .calibrate import calibrate_ladder
from .codebook import build_codebook
from .eager_layers import EagerBf16Linear, EagerBlockRVQLinear, EagerFp8Linear
from .full_layer_tiled_runtime import (
    build_grouped_local_hotprefix_plan,
    build_grouped_local_plan,
    full_layer_grouped_local_hotprefix_matmul,
    full_layer_grouped_local_matmul,
)
from .grouped_block_rvq_linear import PackedGroupedBlockRVQLinear
from .grouped_runtime import GroupedLocalRouteLinear
from .id_route_layers import (
    AdaptiveFusedIDRouteLinear,
    CachedPackedIDRouteLinear,
    FusedIDRouteLinear,
    PackedIDRouteLinear,
)
from .route_encoder import encode_routes, rel_mse
from .runtime_block_replace import (
    replace_with_eager_block_rvq,
    replace_with_packed_block_rvq,
    replace_with_preencoded_packed_block_rvq,
)
from .runtime_common import (
    _load_fused_allowlist,
    _load_shape_runtime_policy,
    _module_name_candidates,
    _resolve_shape_policy,
    _set_submodule,
)
from .runtime_constants import TARGET_LINEAR_SUFFIXES
from .runtime_hybrid_replace import replace_with_hybrid_runtime
from .runtime_quantize import quantize_linear_to_packed
from .runtime_route_replace import (
    replace_packed_id_route_layers,
    replace_with_deployment_runtime,
    replace_with_eager_bf16,
    replace_with_eager_fp8,
    replace_with_eager_hybrid,
)
from .runtime_stage_controls import (
    set_effective_stages_by_name,
    set_effective_stages_from_map,
    set_global_effective_stages,
)
from .triton_id_matmul import id_route_linear_matmul, rvq_group_linear_matmul
from .triton_stage_local_hot_cold import (
    stage_local_hot_cold_matmul,
    stage_local_hot_palette_b2_matmul,
    stage_local_hot_palette_matmul,
)

__all__ = [
    "AdaptiveFusedIDRouteLinear",
    "BlockRVQEncoding",
    "CachedPackedIDRouteLinear",
    "EagerBf16Linear",
    "EagerBlockRVQLinear",
    "EagerFp8Linear",
    "FusedIDRouteLinear",
    "GroupedBlockRVQEncoding",
    "GroupedLocalRouteLinear",
    "PackedBlockRVQGroup",
    "PackedGroupedBlockRVQLinear",
    "PackedIDRouteLinear",
    "TARGET_LINEAR_SUFFIXES",
    "build_codebook",
    "build_grouped_local_hotprefix_plan",
    "build_grouped_local_plan",
    "calibrate_ladder",
    "encode_grouped_block_residual_vq",
    "encode_routes",
    "full_layer_grouped_local_hotprefix_matmul",
    "full_layer_grouped_local_matmul",
    "id_route_linear_matmul",
    "quantize_linear_to_packed",
    "rel_mse",
    "replace_packed_id_route_layers",
    "replace_with_deployment_runtime",
    "replace_with_eager_bf16",
    "replace_with_eager_block_rvq",
    "replace_with_eager_fp8",
    "replace_with_eager_hybrid",
    "replace_with_hybrid_runtime",
    "replace_with_packed_block_rvq",
    "replace_with_preencoded_packed_block_rvq",
    "rvq_group_linear_matmul",
    "set_effective_stages_by_name",
    "set_effective_stages_from_map",
    "set_global_effective_stages",
    "stage_local_hot_cold_matmul",
    "stage_local_hot_palette_b2_matmul",
    "stage_local_hot_palette_matmul",
]
