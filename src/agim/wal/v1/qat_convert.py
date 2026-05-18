from __future__ import annotations

from typing import Tuple

import torch.nn as nn

from .qat_layers import WALQATLinear


def linear_to_qat(
    linear: nn.Linear,
    K: int = 256,
    C: int = 16,
    encode_iters: int = 3,
    use_coeff_adapter: bool = False,
    use_atom_adapter: bool = False,
    atom_adapt_num: int = 8,
) -> WALQATLinear:
    """Convert a nn.Linear layer to WALQATLinear.
    
    Encodes the weight via WAL v2, then creates a differentiable QAT layer
    with learnable atom/coeff tables.
    
    Args:
        linear: Source nn.Linear layer
        K: Number of atoms
        C: Number of coefficients
        encode_iters: k-means/Lloyd-Max iterations
        use_coeff_adapter: Enable WAL-native coeff adapter
        use_atom_adapter: Enable WAL-native atom adapter
        atom_adapt_num: Number of atoms to adapt (if use_atom_adapter)
    
    Returns:
        WALQATLinear with fixed programs and learnable tables
    """
    from ..v2.encoder import build_atoms_kmeans_v2, build_coeff_table, wal_encode_v2
    
    weight = linear.weight.data
    flat = weight.reshape(-1)
    
    # Build atoms and coeffs (non-differentiable, one-time)
    atoms_tensor = build_atoms_kmeans_v2(flat, K=K, iters=encode_iters)
    coeffs_tensor = build_coeff_table(flat, atoms_tensor, C=C, iters=encode_iters)
    
    # Wrap in AtomTable/CoeffTable for wal_encode_v2
    from ..v2.isa import AtomTable, CoeffTable
    atoms = AtomTable(values=atoms_tensor)
    coeffs = CoeffTable(values=coeffs_tensor)
    
    # Encode programs (non-differentiable, one-time)
    prog, recon = wal_encode_v2(flat, atoms, coeffs)
    
    # Create QAT layer with learnable tables initialized from encoded values
    qat_layer = WALQATLinear(
        atom_ids=prog.atom_ids,
        coeff_ids=prog.coeff_ids,
        atom_values=atoms.values,      # Will become nn.Parameter
        coeff_values=coeffs.values,     # Will become nn.Parameter
        shape=weight.shape,
        residuals=prog.residuals if prog.residuals.numel() > 0 else None,
        has_residual=prog.has_residual if prog.has_residual.numel() > 0 else None,
        bias=linear.bias.data if linear.bias is not None else None,
        use_coeff_adapter=use_coeff_adapter,
        use_atom_adapter=use_atom_adapter,
        atom_adapt_num=atom_adapt_num,
    )
    
    return qat_layer


def model_to_qat(
    model: nn.Module,
    K: int = 256,
    C: int = 16,
    encode_iters: int = 3,
) -> nn.Module:
    """Convert all nn.Linear layers in a model to WALQATLinear.
    
    Args:
        model: PyTorch model
        K: Number of atoms
        C: Number of coefficients
        encode_iters: Encoding iterations
    
    Returns:
        Modified model (in-place)
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            qat_layer = linear_to_qat(module, K=K, C=C, encode_iters=encode_iters)
            setattr(model, name, qat_layer)
        else:
            model_to_qat(module, K=K, C=C, encode_iters=encode_iters)
    return model


def count_qat_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count trainable vs total parameters in a QAT model.
    
    Returns:
        (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
