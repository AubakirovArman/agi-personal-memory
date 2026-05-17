"""ROME causal editor for Llama 3.1 — find & edit knowledge neurons."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class ROMECausalEditor:
    """Rank-One Model Editing for Llama-like models.

    Uses causal tracing via hidden states (output_hidden_states) then
    rank-1 update to insert factual knowledge.
    """

    def __init__(self, model, tokenizer, device: str = "cuda:0"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._original_weights: dict[str, torch.Tensor] = {}
        self._edit_count = 0

    def locate_fact_layer(self, prompt: str, target: str,
                          test_layers: list[int] | None = None) -> int:
        """Find which MLP layer has strongest causal effect on fact."""
        if test_layers is None:
            test_layers = [4, 5, 6, 7, 8]  # known knowledge layers for Llama
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        if not target_ids:
            return 5
        target_token = target_ids[0]

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Clean run
        with torch.no_grad():
            clean_out = self.model(**inputs, output_hidden_states=True)
            clean_logits = clean_out.logits[0, -1, :]
            clean_prob = F.softmax(clean_logits.float(), dim=-1)[target_token].item()

        # Test each layer by adding noise to its hidden states
        impacts = []
        for layer_idx in test_layers:
            with torch.no_grad():
                noisy_out = self.model(
                    **inputs,
                    output_hidden_states=True,
                )
                # Corrupt hidden states at this layer
                hs = noisy_out.hidden_states[layer_idx]
                noise = torch.randn_like(hs) * hs.std() * 0.15
                # Re-run from corrupted hidden states
                corrupted_hs = hs + noise
                # Pass through remaining layers
                current = corrupted_hs
                for i in range(layer_idx + 1, len(self.model.model.layers)):
                    current = self.model.model.layers[i](current)[0]
                current = self.model.model.norm(current)
                corr_logits = self.model.lm_head(current)
                corr_prob = F.softmax(corr_logits[0, -1, :].float(), dim=-1)[target_token].item()

            impact = clean_prob - corr_prob
            impacts.append((layer_idx, impact))

        impacts.sort(key=lambda x: x[1], reverse=True)
        best = impacts[0][0] if impacts and impacts[0][1] > 0 else 5
        return best

    def compute_lm_head_update(self, prompt: str, target: str):
        """Compute update for lm_head — direct path to token generation."""
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        if not target_ids:
            return None, None

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        last_hidden = None

        def hook_fn(module, inp, out):
            nonlocal last_hidden
            # out may be tensor or tuple
            hs = out[0] if isinstance(out, tuple) else out
            last_hidden = hs.detach().clone()

        # Hook onto the final layer norm output (before lm_head)
        handle = self.model.model.norm.register_forward_hook(hook_fn)
        with torch.no_grad():
            self.model(**inputs)
        handle.remove()

        if last_hidden is None:
            return None, None

        # Key: last hidden state at final position
        if last_hidden.dim() == 3:
            key = last_hidden[0, -1, :].float()
        else:
            key = last_hidden[-1, :].float()
        key = key / (key.norm() + 1e-8)

        # Target: token ID for the answer
        return key, target_ids[0]

    def apply_edit(self, subject: str, target: str, relation: str = "",
                   layer_idx: int | None = None, clamp_norm: float = 0.3,
                   prompt: str = "") -> bool:
        """Sequence-level edit: each target token boosted in its correct context."""
        if not prompt:
            prompt = f"The {relation} of {subject} is" if relation else f"{subject} is"

        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        if not target_ids:
            return False

        # Encode prompt to token IDs
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_ids = prompt_inputs.input_ids[0]

        try:
            lm_head = self.model.lm_head
            weight = lm_head.weight.data
            name = "lm_head"
            if name not in self._original_weights:
                self._original_weights[name] = weight.clone()

            w_dtype = weight.dtype

            # Sequence-level: each token gets its own hidden state
            for i, tid in enumerate(target_ids):
                if i == 0:
                    key = self._get_last_hidden_from_ids(prompt_ids)
                else:
                    prev_targets = torch.tensor(
                        target_ids[:i], device=self.device, dtype=torch.long)
                    extended_ids = torch.cat([prompt_ids, prev_targets])
                    key = self._get_last_hidden_from_ids(extended_ids)

                if key is None:
                    continue
                key_w = (key / (key.norm() + 1e-8)).to(w_dtype)
                boost = clamp_norm * key_w
                lm_head.weight.data[tid, :] += boost.to(w_dtype)

            self._edit_count += 1
            return True
        except Exception as e:
            print(f"  ROME failed: {e}")
            return False

    def _get_last_hidden(self, prompt: str) -> torch.Tensor | None:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        return self._run_and_get_last_hidden(inputs.input_ids)

    def _get_last_hidden_from_ids(self, token_ids: torch.Tensor) -> torch.Tensor | None:
        return self._run_and_get_last_hidden(token_ids.unsqueeze(0))

    def _run_and_get_last_hidden(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        last_hidden = None

        def hook_fn(module, inp, out):
            nonlocal last_hidden
            hs = out[0] if isinstance(out, tuple) else out
            last_hidden = hs.detach().clone()

        handle = self.model.model.norm.register_forward_hook(hook_fn)
        with torch.no_grad():
            self.model(input_ids=input_ids.to(self.device))
        handle.remove()

        if last_hidden is None:
            return None
        if last_hidden.dim() == 3:
            return last_hidden[0, -1, :].float()
        return last_hidden[-1, :].float() / (key.norm() + 1e-8)

    def rollback(self):
        for name, original in self._original_weights.items():
            if name == "lm_head":
                self.model.lm_head.weight.data = original
            else:
                layer_idx = int(name.split("_")[1])
                self.model.model.layers[layer_idx].mlp.down_proj.weight.data = original
        self._original_weights.clear()
        self._edit_count = 0

    @property
    def edit_count(self) -> int:
        return self._edit_count
