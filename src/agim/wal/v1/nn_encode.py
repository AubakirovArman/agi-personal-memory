from __future__ import annotations

import torch

from .isa import AtomTableV1, CoeffTable
from .nn_layers import WALParameter


def encode_linear_weight(
    weight: torch.Tensor,
    K: int = 256,
    C: int = 16,
    build_hier: bool = False,
    max_l1: int = 64,
) -> WALParameter:
    """Encode a linear layer weight matrix to WAL format.
    
    Args:
        weight: Dense weight tensor [out_features, in_features]
        K: Number of base atoms
        C: Number of coefficients
        build_hier: Whether to build hierarchical atoms
        max_l1: Max L1 atoms if build_hier=True
    
    Returns:
        WALParameter with encoded weights
    """
    from .encoder import build_l0_atoms, build_coeff_table, wal_encode_v1, build_hierarchical_atoms
    
    # Flatten for encoding
    flat = weight.reshape(-1)
    
    # Build atoms and coeffs
    atoms = build_l0_atoms(flat, K=K, iters=3)
    coeffs_tensor = build_coeff_table(flat, atoms, C=C, iters=3)
    
    # Encode (smaller batch to avoid OOM on large layers)
    prog, recon = wal_encode_v1(flat, atoms, coeffs_tensor, batch=262_144)
    
    # Build hierarchical atoms if requested
    if build_hier:
        atom_table = build_hierarchical_atoms(atoms, prog, max_l1=max_l1)
    else:
        from .isa import AtomDef
        atom_defs = [AtomDef(level=0, op="CONST") for _ in range(K)]
        atom_table = AtomTableV1(base_atoms=atoms, atom_defs=atom_defs)
    
    coeff_table = CoeffTable(values=coeffs_tensor)
    
    return WALParameter(
        prog=prog,
        atom_table=atom_table,
        coeffs=coeff_table,
        shape=weight.shape,
        dtype=weight.dtype,
    )
