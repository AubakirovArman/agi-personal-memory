from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class WALProgramAdapter(nn.Module):
    """Task-specific adapter for WAL-encoded weights.
    
    Like LoRA but for WAL: frozen base programs + learned residual.
    The residual is stored in a compact form (low-rank or sparse).
    
    Args:
        shape: Weight shape (out_features, in_features)
        rank: Adapter rank (default: 4)
        alpha: Scaling factor (default: 1.0)
    """
    
    def __init__(self, shape: Tuple[int, ...], rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.shape = shape
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank residual: A [out, rank] @ B [rank, in]
        out_dim = shape[0] if len(shape) >= 1 else 1
        in_dim = shape[1] if len(shape) >= 2 else 1
        
        self.lora_A = nn.Parameter(torch.randn(out_dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, in_dim))
    
    def forward(self, base_weight: torch.Tensor) -> torch.Tensor:
        """Add adapter residual to decoded weight.
        
        Args:
            base_weight: Decoded weight from WAL programs
        
        Returns:
            Adapted weight
        """
        residual = (self.lora_A @ self.lora_B) * self.scaling
        return base_weight + residual.to(base_weight.device)
    
    def merge(self, base_weight: torch.Tensor) -> torch.Tensor:
        """Merge adapter into base weight for inference.
        
        Returns:
            Merged weight (no adapter overhead at inference)
        """
        return self.forward(base_weight)
    
    def extra_repr(self):
        return f"shape={self.shape}, rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"


class WALCoeffAdapter(nn.Module):
    """Learned coefficient offset adapter.
    
    Instead of modifying weights directly, this adapter learns
    per-coefficient offsets: weight = atom * (coeff + learned_delta).
    
    This is more WAL-native than residual adapters because it
    operates in the coefficient space.
    """
    
    def __init__(self, num_coeffs: int, init_scale: float = 0.01):
        super().__init__()
        self.num_coeffs = num_coeffs
        # Learnable offset per coefficient index
        self.coeff_delta = nn.Parameter(torch.zeros(num_coeffs))
        self.init_scale = init_scale
        
        # Initialize near zero
        with torch.no_grad():
            self.coeff_delta.normal_(0, init_scale)
    
    def adapt_coeffs(self, base_coeffs: torch.Tensor) -> torch.Tensor:
        """Apply learned offsets to coefficient table.
        
        Args:
            base_coeffs: Base coefficient values [C]
        
        Returns:
            Adapted coefficients [C]
        """
        return base_coeffs + self.coeff_delta
    
    def extra_repr(self):
        return f"num_coeffs={self.num_coeffs}, init_scale={self.init_scale}"


class WALAtomAdapter(nn.Module):
    """Task-specific atom adaptation.
    
    Learns small perturbations to a subset of atoms.
    More expressive than coeff adapter but more parameters.
    """
    
    def __init__(self, num_atoms: int, num_adapt: int = 8, init_scale: float = 0.01):
        super().__init__()
        self.num_atoms = num_atoms
        self.num_adapt = min(num_adapt, num_atoms)
        
        # Which atoms to adapt (fixed, not learned)
        self.register_buffer('adapt_mask', torch.zeros(num_atoms, dtype=torch.bool))
        self.adapt_mask[:self.num_adapt] = True
        
        # Learnable perturbations for adapted atoms
        self.atom_delta = nn.Parameter(torch.zeros(num_atoms))
        
        with torch.no_grad():
            self.atom_delta.normal_(0, init_scale)
    
    def adapt_atoms(self, base_atoms: torch.Tensor) -> torch.Tensor:
        """Apply learned perturbations to atom table.
        
        Args:
            base_atoms: Base atom values [K]
        
        Returns:
            Adapted atoms [K]
        """
        delta = self.atom_delta * self.adapt_mask.float()
        return base_atoms + delta
    
    def extra_repr(self):
        return f"num_atoms={self.num_atoms}, num_adapt={self.num_adapt}"
