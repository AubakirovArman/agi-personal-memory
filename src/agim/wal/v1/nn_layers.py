from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import wal_decode_v1
from .isa import AtomTableV1, CoeffTable, ProgramBufferV1
from .meta import WALProgramAdapter


class WALParameter:
    """A parameter-like object that stores weights in WAL-encoded format.
    
    Supports lazy decoding with optional cache. The decoded weights are
    materialized on demand and can be cleared to free memory.
    """
    
    def __init__(
        self,
        prog: ProgramBufferV1,
        atom_table: AtomTableV1,
        coeffs: CoeffTable,
        shape: tuple,
        dtype: torch.dtype = torch.float32,
    ):
        self.prog = prog
        self.atom_table = atom_table
        self.coeffs = coeffs
        self.shape = shape
        self.dtype = dtype
        self._cache: Optional[torch.Tensor] = None
        self._cache_device: Optional[torch.device] = None
    
    def decode(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Decode WAL-encoded weights to dense tensor.
        
        Uses cached value if available and on the same device.
        """
        if self._cache is not None:
            if device is None or self._cache_device == device:
                return self._cache
        
        # Decode
        weight = wal_decode_v1(self.prog, self.atom_table, self.coeffs.values)
        weight = weight.reshape(self.shape).to(self.dtype)
        
        if device is not None:
            weight = weight.to(device)
        
        self._cache = weight
        self._cache_device = weight.device
        return weight
    
    def clear_cache(self):
        """Clear decoded weight cache to free memory."""
        self._cache = None
        self._cache_device = None
    
    @property
    def numel(self) -> int:
        """Total number of elements."""
        return int(torch.prod(torch.tensor(self.shape)).item())
    
    def __repr__(self):
        return (
            f"WALParameter(shape={self.shape}, K0={self.atom_table.K0}, "
            f"K_total={self.atom_table.K_total}, C={self.coeffs.values.numel()}, "
            f"cached={self._cache is not None})"
        )


class WALLinear(nn.Module):
    """Linear layer with WAL-encoded weight matrix.
    
    The weight matrix is stored in WAL-compressed form and decoded on-the-fly
    during forward pass. Supports optional caching for repeated forwards.
    """
    
    def __init__(
        self,
        wal_weight: WALParameter,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.wal_weight = wal_weight
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: decode weight and apply linear transformation."""
        weight = self.wal_weight.decode(x.device)
        return F.linear(x, weight, self.bias)
    
    def clear_cache(self):
        """Clear weight decode cache."""
        self.wal_weight.clear_cache()
    
    @property
    def weight_shape(self) -> tuple:
        return self.wal_weight.shape
    
    def __repr__(self):
        return f"WALLinear(in_features={self.weight_shape[1]}, out_features={self.weight_shape[0]}, bias={self.bias is not None})"


class WALCachedLinear(nn.Module):
    """Linear layer with WAL-encoded weights and persistent decode cache.
    
    Decodes weights once and keeps them in memory for fast repeated access.
    Use this when memory is available and speed is critical.
    
    Supports optional meta-learning adapter for task-specific fine-tuning.
    """
    
    def __init__(
        self,
        wal_weight: WALParameter,
        bias: Optional[torch.Tensor] = None,
        adapter: Optional[WALProgramAdapter] = None,
    ):
        super().__init__()
        self.wal_weight = wal_weight
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
        self.adapter = adapter
        self._decoded = False
    
    def _ensure_decoded(self, device: torch.device):
        if not self._decoded:
            weight = self.wal_weight.decode(device)
            if self.adapter is not None:
                weight = self.adapter(weight)
            self.register_buffer('weight', weight)
            self._decoded = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_decoded(x.device)
        return F.linear(x, self.weight, self.bias)
    
    def clear_cache(self):
        if self._decoded and hasattr(self, 'weight'):
            delattr(self, 'weight')
            self._decoded = False
    
    def set_adapter(self, adapter: Optional[WALProgramAdapter]):
        """Attach or detach a meta-learning adapter."""
        self.adapter = adapter
        self.clear_cache()
