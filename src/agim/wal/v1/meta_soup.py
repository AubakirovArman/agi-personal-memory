from __future__ import annotations

from typing import List, Optional

import torch

from .isa import ProgramBufferV1


def program_soup(
    programs: List[ProgramBufferV1],
    weights: Optional[List[float]] = None,
    method: str = "mean",
) -> ProgramBufferV1:
    """Merge programs from multiple models (model soup at program level).
    
    Args:
        programs: List of ProgramBufferV1 from different models
        weights: Optional weight per model (default: equal)
        method: "mean", "majority", or "weighted"
    
    Returns:
        Merged ProgramBufferV1
    """
    if not programs:
        raise ValueError("Empty program list")
    
    N = programs[0].N
    shape = programs[0].shape
    
    if weights is None:
        weights = [1.0 / len(programs)] * len(programs)
    
    if method == "mean" or method == "weighted":
        # Average atom_ids and coeff_ids (not ideal but works as baseline)
        device = programs[0].atom_ids.device
        atom_ids_sum = torch.zeros(N, dtype=torch.float32, device=device)
        coeff_ids_sum = torch.zeros(N, dtype=torch.float32, device=device)
        
        for prog, w in zip(programs, weights):
            atom_ids_sum += prog.atom_ids.float() * w
            coeff_ids_sum += prog.coeff_ids.float() * w
        
        merged_atom_ids = atom_ids_sum.round().clamp(0, 255).to(torch.uint8)
        merged_coeff_ids = coeff_ids_sum.round().clamp(0, 255).to(torch.uint8)
        
    elif method == "majority":
        # Majority vote per position
        merged_atom_ids = torch.zeros(N, dtype=torch.uint8)
        merged_coeff_ids = torch.zeros(N, dtype=torch.uint8)
        
        for i in range(N):
            atom_votes = [int(p.atom_ids[i]) for p in programs]
            coeff_votes = [int(p.coeff_ids[i]) for p in programs]
            
            # Most common
            from collections import Counter
            merged_atom_ids[i] = Counter(atom_votes).most_common(1)[0][0]
            merged_coeff_ids[i] = Counter(coeff_votes).most_common(1)[0][0]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return ProgramBufferV1(
        atom_ids=merged_atom_ids,
        coeff_ids=merged_coeff_ids,
        residuals=torch.empty(0, dtype=torch.float16),
        has_residual=torch.zeros(N, dtype=torch.bool),
        shape=shape,
    )
