from __future__ import annotations

import json
from pathlib import Path

import torch

from .nn import WALCachedLinear, WALLinear


class WALModelPersistenceMixin:
    def save(self, path: str):
        """Save WAL checkpoint.
        
        Args:
            path: Output path (.wal file or directory)
        """
        wal_path = Path(path)
        
        # Collect metadata
        meta = {
            "K": self.K,
            "C": self.C,
            "device": self.device,
            "overlays": list(self._overlays.keys()),
            "merged": list(self._merged.keys()),
        }
        
        # Serialize WAL state
        # TODO: full serialization of all WAL layers
        # For now, save metadata + model state dict
        
        if wal_path.suffix == ".wal":
            meta_bytes = json.dumps(meta).encode()
            meta_len = len(meta_bytes).to_bytes(8, "little")
            
            # Placeholder: save model state dict as pickle
            import pickle
            state = self.model.state_dict()
            state_bytes = pickle.dumps(state)
            
            with open(wal_path, "wb") as f:
                f.write(meta_len + meta_bytes + state_bytes)
        else:
            wal_path.mkdir(parents=True, exist_ok=True)
            with open(wal_path / "wal_meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            
            # Save model state
            self.model.save_pretrained(wal_path / "model")
        
        print(f"[WALModel] Saved to: {path}")
    
    def generate(self, prompt: str, tokenizer, **gen_kwargs) -> str:
        """Generate text using the model.
        
        Args:
            prompt: Input prompt
            tokenizer: HuggingFace tokenizer
            **gen_kwargs: Generation parameters
        
        Returns:
            Generated text
        """
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def __repr__(self):
        n_wal = sum(1 for _ in self.model.modules() if isinstance(_, (WALCachedLinear, WALLinear)))
        return f"WALModel(layers={n_wal}, overlays={list(self._overlays.keys())}, merged={list(self._merged.keys())})"
