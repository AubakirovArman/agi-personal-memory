from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .nn import WALCachedLinear, WALLinear


class WALModelOverlayMixin:
    def attach_lora(self, path: str, name: Optional[str] = None) -> str:
        """Attach a LoRA overlay from safetensors or bin file.
        
        Args:
            path: Path to LoRA weights
            name: Optional name for the overlay (default: filename)
        
        Returns:
            Overlay name
        """
        if name is None:
            name = Path(path).stem
        
        # Load LoRA weights
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            lora_weights = load_file(path)
        else:
            lora_weights = torch.load(path, map_location="cpu", weights_only=True)
        
        self._overlays[name] = {
            "weights": lora_weights,
            "active": False,
        }
        
        print(f"[WALModel] Attached LoRA overlay: {name} ({len(lora_weights)} tensors)")
        return name
    
    def enable_overlay(self, name: str):
        """Enable a LoRA overlay for inference."""
        if name not in self._overlays:
            raise ValueError(f"Unknown overlay: {name}")
        
        self._overlays[name]["active"] = True
        
        # Apply LoRA to model
        lora_weights = self._overlays[name]["weights"]
        
        for layer_name, module in self.model.named_modules():
            if not isinstance(module, (WALCachedLinear, WALLinear)):
                continue
            
            # Find matching LoRA weights
            lora_a_key = f"{layer_name}.lora_A.weight"
            lora_b_key = f"{layer_name}.lora_B.weight"
            
            if lora_a_key in lora_weights and lora_b_key in lora_weights:
                lora_A = lora_weights[lora_a_key].to(self.device)
                lora_B = lora_weights[lora_b_key].to(self.device)
                
                # Decode base weight, add LoRA, re-encode
                base_weight = module.wal_weight.decode(self.device)
                delta = (lora_A @ lora_B).to(base_weight.dtype)
                merged = base_weight + delta
                
                # Re-encode to WAL
                from .nn import encode_linear_weight
                module.wal_weight = encode_linear_weight(merged, K=self.K, C=self.C)
                module.clear_cache()
        
        print(f"[WALModel] Enabled overlay: {name}")
    
    def disable_overlay(self, name: str):
        """Disable a LoRA overlay (revert to base)."""
        if name not in self._overlays:
            raise ValueError(f"Unknown overlay: {name}")
        
        self._overlays[name]["active"] = False
        
        # For simplicity, reload base model
        # In production, store base programs and restore them
        print(f"[WALModel] Disabled overlay: {name} (reload base to fully revert)")

    def merge_overlay(self, name: str):
        """Permanently merge a LoRA overlay into the base WAL.
        
        This re-encodes the merged weights into WAL and clears the overlay.
        """
        if name not in self._overlays:
            raise ValueError(f"Unknown overlay: {name}")
        
        # Ensure overlay is applied
        if not self._overlays[name]["active"]:
            self.enable_overlay(name)
        
        # Mark as merged
        self._merged[name] = True
        
        # Clear overlay (merged state is now base)
        del self._overlays[name]
        
        print(f"[WALModel] Merged overlay: {name}")
