"""Multi-layer editor: lm_head + FFN for ES↔NS tradeoff.

lm_head (clamp=0.15): слабый edit → NS=98% но ES=7%
FFN layer 7:         точечный boost для target контекста → ES↑
Комбинация:          ES > 50%, NS > 80%
"""

from __future__ import annotations
import torch
from ..wal.encoder import build_atoms_kmeans, wal_encode_scalar_gpu


class MultiLayerEditor:
    """Edit lm_head + FFN layer 7 для лучшего ES/NS баланса."""

    def __init__(self, model, tokenizer, K=256, lmax=12, device="cuda:3"):
        self.model = model
        self.tokenizer = tokenizer
        self.K = K
        self.lmax = lmax
        self.device = torch.device(device)

        # WAL atoms for lm_head
        self.atoms: torch.Tensor | None = None
        self._vocab_size = 0
        self._hidden_size = 0

        # Rollback storage
        self._lm_original: dict[int, torch.Tensor] = {}
        self._ffn_original: dict[str, torch.Tensor] = {}
        self._edit_count = 0

        # FFN target layer
        self.ffn_layer = 7

    def build_vocab(self):
        weight = self.model.lm_head.weight.data.float()
        self._vocab_size, self._hidden_size = weight.shape
        flat = weight.flatten()
        self.atoms = build_atoms_kmeans(
            flat[:2_000_000], self.K, iters=5, device=self.device)
        print(f"  WAL atoms: {self.atoms.shape}")

    # ═══ lm_head edit (light) ═══
    def _edit_lm_head(self, target_ids, key, clamp):
        """Light lm_head edit — gives partial ES, preserves NS."""
        atoms_gpu = self.atoms.to(self.device)
        weight = self.model.lm_head.weight.data

        for i, tid in enumerate(target_ids):
            if tid not in self._lm_original:
                self._lm_original[tid] = weight[tid, :].clone()
            row = weight[tid, :].float().to(self.device)
            boost = clamp * key.to(self.device) / (2 ** i)
            _, _, recon = wal_encode_scalar_gpu(row + boost, atoms_gpu, self.lmax)
            weight[tid, :] = recon.to(device=weight.device, dtype=weight.dtype)

    # ═══ FFN edit (targeted) ═══
    def _edit_ffn(self, prompt_ids, target_ids, clamp):
        """Edit FFN layer 7 down_proj for target context specificity."""
        mlp = self.model.model.layers[self.ffn_layer].mlp
        W_name = f"layer_{self.ffn_layer}_down_proj"
        if W_name not in self._ffn_original:
            self._ffn_original[W_name] = mlp.down_proj.weight.data.clone()

        # Get MLP key at last position for prompt context
        m_val = None

        def hook_down(module, inp, out):
            nonlocal m_val
            m_val = inp[0].detach().clone()

        h = mlp.down_proj.register_forward_hook(hook_down)
        with torch.no_grad():
            self.model(input_ids=prompt_ids.unsqueeze(0).to(self.device))
        h.remove()

        if m_val is None:
            return

        m = m_val[0, -1, :].float()  # MLP key at last position

        # Target direction: align lm_head rows toward target tokens
        target_dir = torch.zeros(self._hidden_size, device=self.device)
        for i, tid in enumerate(target_ids):
            target_dir += self.model.lm_head.weight.data[tid, :].float() / (2 ** i)
        target_dir = target_dir / (target_dir.norm() + 1e-8)

        # Rank-1 update: W += clamp * target_dir @ m^T
        update = clamp * torch.outer(target_dir, m)
        mlp.down_proj.weight.data += update.to(
            dtype=mlp.down_proj.weight.dtype, device=self.device)

    # ═══ combined edit ═══
    def apply_edit(self, subject, target, relation="", clamp_lm=0.15,
                   clamp_ffn=0.05, prompt=""):
        """Multi-layer edit: lm_head (light) + FFN (targeted)."""
        if not prompt:
            prompt = f"{subject} is" if not relation else f"The {relation} of {subject} is"

        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        if not target_ids:
            return False

        prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_ids = prompt_inputs.input_ids[0]

        # ── lm_head: sequence-level, light clamp ──
        for i, tid in enumerate(target_ids):
            if i == 0:
                ctx_ids = prompt_ids
            else:
                prev = torch.tensor(target_ids[:i], device=self.device, dtype=torch.long)
                ctx_ids = torch.cat([prompt_ids, prev])
            key = self._get_last_hidden(ctx_ids)
            if key is not None:
                self._edit_lm_head([tid], key / (key.norm() + 1e-8), clamp_lm)

        # ── FFN: boost target direction ──
        self._edit_ffn(prompt_ids, target_ids, clamp_ffn)

        self._edit_count += 1
        return True

    def rollback(self):
        for tid, orig in self._lm_original.items():
            self.model.lm_head.weight.data[tid, :] = orig.to(
                device=self.model.lm_head.weight.device,
                dtype=self.model.lm_head.weight.dtype)
        self._lm_original.clear()

        for name, orig in self._ffn_original.items():
            layer_idx = int(name.split("_")[1])
            self.model.model.layers[layer_idx].mlp.down_proj.weight.data = orig
        self._ffn_original.clear()
        self._edit_count = 0

    def _get_last_hidden(self, token_ids):
        last_hidden = None
        def hook_fn(module, inp, out):
            nonlocal last_hidden
            hs = out[0] if isinstance(out, tuple) else out
            last_hidden = hs.detach().clone()
        handle = self.model.model.norm.register_forward_hook(hook_fn)
        with torch.no_grad():
            self.model(input_ids=token_ids.unsqueeze(0).to(self.device))
        handle.remove()
        if last_hidden is None:
            return None
        return last_hidden[0, -1, :].float() if last_hidden.dim() == 3 else last_hidden[-1, :].float()

    @property
    def edit_count(self):
        return self._edit_count
