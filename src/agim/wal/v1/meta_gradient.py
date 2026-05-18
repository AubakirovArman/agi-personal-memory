from __future__ import annotations

from typing import Tuple

import torch

from .decoder import wal_decode_v1
from .isa import AtomTableV1, CoeffTable, ProgramBufferV1


def compute_program_gradient(
    prog: ProgramBufferV1,
    atom_table: AtomTableV1,
    coeffs: CoeffTable,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute gradient of MSE loss w.r.t. program atom_ids and coeff_ids.
    
    Note: atom_ids and coeff_ids are discrete (uint8), so true gradients
    don't exist. This function computes gradients w.r.t. the decoded
    weight and projects them to the nearest valid program change.
    
    Args:
        prog: Current program
        atom_table: Atom table
        coeffs: Coeff table
        target: Target weights
    
    Returns:
        (atom_grad, coeff_grad) — gradients for each position
    """
    recon = wal_decode_v1(prog, atom_table, coeffs.values)
    error = recon - target.flatten()
    
    flat_atoms = torch.tensor(
        [atom_table.resolve(i) for i in range(atom_table.K_total)],
        dtype=torch.float32,
        device=prog.atom_ids.device,
    )
    
    # Gradient w.r.t. atom choice: -error * coeff_value
    N = prog.N
    atom_grad = torch.zeros(N, dtype=torch.float32, device=prog.atom_ids.device)
    coeff_grad = torch.zeros(N, dtype=torch.float32, device=prog.atom_ids.device)
    
    for i in range(N):
        a = int(prog.atom_ids[i])
        c = int(prog.coeff_ids[i])
        atom_val = flat_atoms[a]
        coeff_val = coeffs.values[c]
        
        # d(MSE)/d(atom_id) ≈ d(MSE)/d(recon) * d(recon)/d(atom) * d(atom)/d(atom_id)
        # For nearest-neighbor, this is approximately:
        atom_grad[i] = error[i] * coeff_val
        coeff_grad[i] = error[i] * atom_val
    
    return atom_grad, coeff_grad
