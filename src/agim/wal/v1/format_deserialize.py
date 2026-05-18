from __future__ import annotations

import json
import struct
from typing import Dict, Tuple

import torch

from .isa import AtomTableV1, CoeffTable, ProgramBufferV1
from .format_helpers import FLAG_HAS_RESIDUALS, FLAG_PACKED_COEFFS, MAGIC, VERSION, _deserialize_hierarchical, _unpack_bitmap, _unpack_uint4


def deserialize_wal_v1(data: bytes) -> Tuple[ProgramBufferV1, AtomTableV1, CoeffTable, Dict]:
    """Deserialize compact binary to WAL v1 state.
    
    Args:
        data: Binary blob
    
    Returns:
        (prog, atom_table, coeffs, metadata)
    """
    offset = 0
    
    # Header
    magic, version, K0, K_total, C, flags, N = struct.unpack_from('<4sHHHHHQ', data, offset)
    assert magic == MAGIC, f"Invalid magic: {magic}"
    assert version == VERSION, f"Unsupported version: {version}"
    offset += 32
    
    has_residuals = bool(flags & FLAG_HAS_RESIDUALS)
    packed_coeffs = bool(flags & FLAG_PACKED_COEFFS)
    
    # Base atom table
    base_atoms = torch.frombuffer(data, dtype=torch.float32, count=K0, offset=offset)
    offset += K0 * 4
    
    # Hierarchical definitions
    atom_defs, offset = _deserialize_hierarchical(data, offset, K0)
    assert len(atom_defs) == K_total, f"Expected {K_total} defs, got {len(atom_defs)}"
    
    # Coeff table
    coeffs_data = torch.frombuffer(data, dtype=torch.float32, count=C, offset=offset)
    offset += C * 4
    
    # atom_ids
    atom_ids = torch.frombuffer(data, dtype=torch.uint8, count=N, offset=offset)
    offset += N
    
    # coeff_ids
    if packed_coeffs:
        packed_len = (N + 1) // 2
        coeff_packed = torch.frombuffer(data, dtype=torch.uint8, count=packed_len, offset=offset)
        coeff_ids = _unpack_uint4(coeff_packed, N)
        offset += packed_len
    else:
        coeff_ids = torch.frombuffer(data, dtype=torch.uint8, count=N, offset=offset)
        offset += N
    
    # Residuals
    residual_count = struct.unpack_from('<I', data, offset)[0]
    offset += 4
    
    bitmap_len = (N + 7) // 8
    has_residual = _unpack_bitmap(data[offset:offset + bitmap_len], N)
    offset += bitmap_len
    
    residuals = torch.zeros(N, dtype=torch.float32)
    if residual_count > 0:
        residual_indices = torch.frombuffer(
            data, dtype=torch.uint32, count=residual_count, offset=offset
        ).long()
        offset += residual_count * 4
        
        residual_values = torch.frombuffer(
            data, dtype=torch.float16, count=residual_count, offset=offset
        ).float()
        offset += residual_count * 2
        
        residuals[residual_indices] = residual_values
    
    # Metadata
    meta_len = struct.unpack_from('<Q', data, offset)[0]
    offset += 8
    meta_json = json.loads(data[offset:offset + meta_len])
    
    # Reconstruct
    shape = tuple(meta_json['shape'])
    
    prog = ProgramBufferV1(
        atom_ids=atom_ids,
        coeff_ids=coeff_ids,
        residuals=residuals,
        has_residual=has_residual,
        shape=shape,
    )
    atom_table = AtomTableV1(base_atoms=base_atoms, atom_defs=atom_defs)
    coeffs = CoeffTable(values=coeffs_data)
    
    return prog, atom_table, coeffs, meta_json
