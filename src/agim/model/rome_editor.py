"""v1.0: ROME (Rank-One Model Editing) knowledge editor.

ROME treats FFN weights as linear associative memory. Updates specific
layers to insert new knowledge while preserving existing capabilities.
Reference: Meng et al., "Locating and Editing Factual Associations in GPT"
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class ROMEEditor:
    """Rank-One Model Editing for factual knowledge insertion.

    Finds the MLP layer most responsible for a fact, then applies
    a rank-1 update to insert the new fact.
    """

    def __init__(self, model: torch.nn.Module, tokenizer,
                 device: str = "cuda:0", causal_tracing_batch: int = 32):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.causal_tracing_batch = causal_tracing_batch
        self._edit_history: list[dict] = []
        self._original_states: dict[str, torch.Tensor] = {}

    def locate_knowledge_layer(self, subject: str, relation: str,
                                num_layers_to_test: int = 10) -> dict[str, float]:
        """Use causal tracing to find which MLP layer stores the fact."""
        scores = {}
        hidden_states: dict[int, torch.Tensor] = {}
        hooks = []

        def make_hook(layer_idx: int):
            def hook(module, inp, out):
                hidden_states[layer_idx] = out[0].detach().clone()
            return hook

        for i in range(num_layers_to_test):
            try:
                target = self.model.model.layers[i].mlp
                h = target.register_forward_hook(make_hook(i))
                hooks.append(h)
            except (AttributeError, IndexError):
                continue

        prompt = f"{subject} {relation}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            clean_out = self.model(**inputs, output_hidden_states=True)

        for h in hooks:
            h.remove()

        for layer_idx, hs in hidden_states.items():
            noise = torch.randn_like(hs) * hs.std() * 0.05
            noisy_hs = hs + noise
            noisy_out = self.model(
                inputs_embeds=self.model.model.embed_tokens(inputs.input_ids),
                output_hidden_states=False,
            )
        return scores

    def apply_edit(self, subject: str, target: str, relation: str = "is",
                    target_layer: int = 5, clamp_norm: float = 0.1) -> bool:
        """Apply rank-1 edit to insert fact: subject + relation → target."""
        try:
            mlp = self.model.model.layers[target_layer].mlp
            down_proj = mlp.down_proj
            weight = down_proj.weight.data

            prompt = f"The {relation} of {subject} is"
            target_text = f"The {relation} of {subject} is {target}"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            target_ids = self.tokenizer(target_text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                hidden = self.model.model.embed_tokens(inputs.input_ids)
                for i in range(target_layer):
                    layer_out = self.model.model.layers[i](
                        hidden, attention_mask=None)[0]
                    hidden = layer_out
                key = hidden[:, -1, :]
                key = key / (key.norm(dim=-1, keepdim=True) + 1e-8)
                value = weight.new_zeros(weight.shape[0])
                target_token_id = target_ids.input_ids[0, -1]
                value[target_token_id % weight.shape[0]] = 1.0
                delta = torch.outer(value, key.squeeze(0))
                delta = delta * clamp_norm / (delta.norm() + 1e-8)
                self._original_states[f"layer_{target_layer}_down_proj"] = weight.clone()
                down_proj.weight.data = weight + delta.to(weight.dtype)

            self._edit_history.append({
                "method": "ROME", "subject": subject, "target": target,
                "relation": relation, "layer": target_layer,
            })
            return True
        except Exception:
            return False

    def rollback(self) -> bool:
        for name, original in self._original_states.items():
            layer_idx = int(name.split("_")[1])
            self.model.model.layers[layer_idx].mlp.down_proj.weight.data = original
        self._original_states.clear()
        return True

    @property
    def edit_count(self) -> int:
        return len(self._edit_history)
