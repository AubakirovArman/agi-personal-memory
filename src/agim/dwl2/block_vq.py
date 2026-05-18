"""Block residual VQ public facade.

Implementation details live in smaller modules so the project keeps each Python
file below the local 300-line size gate.
"""
from __future__ import annotations

from .block_vq_encode import encode_block_residual_vq
from .block_vq_encoding import BlockRVQEncoding, GroupedBlockRVQEncoding
from .block_vq_grouped import (
    dense_bf16_storage_bytes,
    encode_grouped_block_residual_vq,
    ideal_id_bits_per_weight,
    sample_grouped_row_similarity,
    sample_row_similarity,
    storage_megabytes,
)
from .block_vq_kmeans import (
    _assign_to_codebook,
    _fit_kmeans,
    _id_storage_bytes,
    _id_storage_dtype,
    _reshape_blocks,
)
from .block_vq_transforms import (
    _dct_matrix,
    _fit_pca_transform,
    _hadamard_matrix,
    _inverse_polar_transform,
    _pack_sign_bits,
    _polar_transform,
    _rand_hadamard_matrix,
    _sign_correction_matrix,
    _transform_matrix,
    _unpack_sign_bits,
)

__all__ = [
    "BlockRVQEncoding",
    "GroupedBlockRVQEncoding",
    "dense_bf16_storage_bytes",
    "encode_block_residual_vq",
    "encode_grouped_block_residual_vq",
    "ideal_id_bits_per_weight",
    "sample_grouped_row_similarity",
    "sample_row_similarity",
    "storage_megabytes",
    "_assign_to_codebook",
    "_dct_matrix",
    "_fit_kmeans",
    "_fit_pca_transform",
    "_hadamard_matrix",
    "_id_storage_bytes",
    "_id_storage_dtype",
    "_inverse_polar_transform",
    "_pack_sign_bits",
    "_polar_transform",
    "_rand_hadamard_matrix",
    "_reshape_blocks",
    "_sign_correction_matrix",
    "_transform_matrix",
    "_unpack_sign_bits",
]
