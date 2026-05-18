from __future__ import annotations

import json
import struct
from typing import Dict

import numpy as np
import torch

from .isa import AtomTableV1, CoeffTable, ProgramBufferV1
from .format_helpers import FLAG_HAS_RESIDUALS, FLAG_PACKED_COEFFS, MAGIC, VERSION, _pack_bitmap, _pack_uint4, _serialize_hierarchical


def serialize_wal_v1(
    prog: ProgramBufferV1,
    atom_table: AtomTableV1,
    coeffs: CoeffTable,
    metadata: Dict = None,
) -> bytes:
    """Serialize WAL v1 state to compact binary.
    
    Args:
        prog: Program buffer
        atom_table: Hierarchical atom table
        coeffs: Coefficient table
        metadata: Optional metadata dict
    
    Returns:
        Binary blob as bytes
    """
    K0 = atom_table.K0
    K_total = atom_table.K_total
    C = coeffs.values.numel()
    N = prog.N
    
    has_residuals = prog.has_residual.any().item()
    flags = FLAG_PACKED_COEFFS
    if has_residuals:
        flags |= FLAG_HAS_RESIDUALS
    
    # Header (32 bytes)
    header = struct.pack(
        '<4sHHHHHQ10s',
        MAGIC, VERSION, K0, K_total, C, flags, N, b'\x00' * 10,
    )
    assert len(header) == 32
    
    # Base atom table
    atom_bytes = atom_table.base_atoms.cpu().float().numpy().tobytes()
    
    # Hierarchical definitions
    hier_bytes = _serialize_hierarchical(atom_table.atom_defs)
    
    # Coeff table
    coeff_bytes = coeffs.values.cpu().float().numpy().tobytes()
    
    # Programs
    atom_id_bytes = prog.atom_ids.cpu().numpy().tobytes()
    
    coeff_packed = _pack_uint4(prog.coeff_ids.cpu())
    coeff_id_bytes = coeff_packed.numpy().tobytes()
    
    # Residuals
    residual_bitmap = _pack_bitmap(prog.has_residual.cpu())
    residual_count = int(prog.has_residual.sum().item())
    
    if residual_count > 0:
        residual_indices = prog.has_residual.nonzero(as_tuple=False).flatten().cpu()
        residual_values = prog.residuals[prog.has_residual].cpu().half()
        residual_section = (
            struct.pack('<I', residual_count)
            + residual_bitmap
            + residual_indices.numpy().astype(np.uint32).tobytes()
            + residual_values.numpy().tobytes()
        )
    else:
        residual_section = struct.pack('<I', 0) + residual_bitmap
    
    # Metadata
    meta = metadata or {}
    meta.update({
        'shape': list(prog.shape),
    })
    meta_json = json.dumps(meta).encode()
    meta_len = struct.pack('<Q', len(meta_json))
    
    return (
        header
        + atom_bytes
        + hier_bytes
        + coeff_bytes
        + atom_id_bytes
        + coeff_id_bytes
        + residual_section
        + meta_len
        + meta_json
    )
