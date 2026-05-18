#!/usr/bin/env python3
"""M171 unified WAL runtime pipeline facade/core class."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from .format import deserialize_wal_v1, serialize_wal_v1
from .isa import AtomTableV1, CoeffTable, ProgramBufferV1
from .nn import WALCachedLinear, WALLinear, replace_linear_with_wal, replace_wal_with_linear
from .runtime_overlay import WALModelOverlayMixin
from .runtime_persistence import WALModelPersistenceMixin
from .runtime_safety import WALModelSafetyMixin


class WALModel(WALModelOverlayMixin, WALModelSafetyMixin, WALModelPersistenceMixin):
    """Unified WAL runtime model with LoRA overlay support.
    
    Wraps a transformers model with WAL-encoded weights and provides
    a high-level API for loading, editing, safety-checking, and saving.
    """
    
    def __init__(self, model: nn.Module, K: int = 256, C: int = 16, device: str = "cuda"):
        self.model = model
        self.K = K
        self.C = C
        self.device = device
        self._overlays: Dict[str, dict] = {}  # name -> {lora_weights, active}
        self._merged: Dict[str, bool] = {}    # name -> merged flag
        self._base_state: Optional[dict] = None  # for rollback
        
    @classmethod
    def from_dense(
        cls,
        model_name: str,
        K: int = 256,
        C: int = 16,
        device: str = "cuda",
        dtype = torch.bfloat16,
    ) -> "WALModel":
        """Load a dense model and encode to WAL.
        
        Args:
            model_name: HuggingFace model name or local path
            K: Number of atoms
            C: Number of coefficients
            device: Device to load on
            dtype: Torch dtype
        
        Returns:
            WALModel with WAL-encoded weights
        """
        from transformers import AutoModelForCausalLM
        
        print(f"[WALModel] Loading dense model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
        )
        
        print(f"[WALModel] Encoding to WAL (K={K}, C={C})...")
        replace_linear_with_wal(model, K=K, C=C, cached=True)
        
        return cls(model, K=K, C=C, device=device)
    
    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "WALModel":
        """Load a WAL checkpoint.
        
        Args:
            path: Path to .wal file or directory with model + wal_state
            device: Device to load on
        
        Returns:
            WALModel with loaded WAL-encoded weights
        """
        wal_path = Path(path)
        
        if wal_path.suffix == ".wal":
            # Single .wal file with metadata
            with open(wal_path, "rb") as f:
                data = f.read()
            
            # Parse: first 8 bytes = JSON metadata length
            meta_len = int.from_bytes(data[:8], "little")
            meta = json.loads(data[8:8+meta_len])
            
            # Remaining = serialized WAL state
            wal_blob = data[8+meta_len:]
            
            model_name = meta.get("model_name", "meta-llama/Llama-3.1-8B")
            K = meta.get("K", 256)
            C = meta.get("C", 16)
            
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=getattr(torch, meta.get("dtype", "bfloat16")),
                device_map=device,
            )
            
            # Replace with WAL layers and load state
            replace_linear_with_wal(model, K=K, C=C, cached=True)
            
            # TODO: deserialize wal_blob into layer states
            # For now, we re-encode (simplified)
            
            return cls(model, K=K, C=C, device=device)
        else:
            # Directory with separate files
            meta_path = wal_path / "wal_meta.json"
            with open(meta_path) as f:
                meta = json.load(f)
            
            return cls.from_dense(
                meta["model_name"],
                K=meta.get("K", 256),
                C=meta.get("C", 16),
                device=device,
            )


__all__ = [
    "AtomTableV1",
    "CoeffTable",
    "ProgramBufferV1",
    "WALCachedLinear",
    "WALLinear",
    "WALModel",
    "deserialize_wal_v1",
    "replace_linear_with_wal",
    "replace_wal_with_linear",
    "serialize_wal_v1",
]
