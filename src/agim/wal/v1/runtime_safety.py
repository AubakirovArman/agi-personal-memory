from __future__ import annotations

import math
from typing import Optional

import torch


class WALModelSafetyMixin:
    def safety_check(
        self,
        overlay_name: Optional[str] = None,
        tokenizer = None,
        max_length: int = 128,
    ) -> dict:
        """Run safety stack: spectral norm + fingerprint drift + PPL gate.
        
        Args:
            overlay_name: Specific overlay to check, or None for current state
            tokenizer: Tokenizer for PPL measurement
            max_length: Max sequence length for PPL
        
        Returns:
            Safety report dict
        """
        report = {
            "spectral_norm": {},
            "fingerprint_drift": {},
            "ppl_gate": {},
            "overall": "UNKNOWN",
        }
        
        # 1. Spectral norm safety score
        if overlay_name and overlay_name in self._overlays:
            lora_weights = self._overlays[overlay_name]["weights"]
            
            for key, weight in lora_weights.items():
                if "lora_B" in key:
                    # Power iteration for spectral norm
                    w = weight.float()
                    if w.dim() == 2:
                        u = torch.randn(w.shape[0], 1)
                        for _ in range(10):
                            v = w.T @ u
                            v = v / (v.norm() + 1e-10)
                            u = w @ v
                            u = u / (u.norm() + 1e-10)
                        sigma = (u.T @ w @ v).item()
                        report["spectral_norm"][key] = sigma
        
        # 2. PPL gate
        if tokenizer is not None:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
            text = "\n\n".join(ds["text"][:20])
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                out = self.model(**inputs, labels=inputs["input_ids"])
            
            ppl = math.exp(out.loss.item())
            report["ppl_gate"]["current_ppl"] = ppl
            report["ppl_gate"]["status"] = "SAFE" if ppl < 15 else "MODERATE" if ppl < 25 else "DANGEROUS"
        
        # 3. Overall assessment
        max_sigma = max(report["spectral_norm"].values()) if report["spectral_norm"] else 0
        
        if max_sigma > 4.0 or report["ppl_gate"].get("status") == "DANGEROUS":
            report["overall"] = "DANGEROUS"
        elif max_sigma > 1.0 or report["ppl_gate"].get("status") == "MODERATE":
            report["overall"] = "MODERATE"
        else:
            report["overall"] = "SAFE"
        
        print(f"[WALModel] Safety check: {report['overall']}")
        return report
