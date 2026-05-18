"""ROME + WAL hybrid: ROME's rank-1 math + WAL encoding for rollback/audit.

ROME paper math:
  k = MLP key (subject hidden state at MLP input)
  v* = optimized target output
  W' = W + (v* - Wk) @ k^T / (k^T k)   ← rank-1 update

WAL integration:
  Build atom vocab from down_proj weights
  After rank-1 update, delta columns encoded as WAL programs
  Rollback = restore original columns from programs
"""

from __future__ import annotations
import torch
from ..wal.encoder import build_atoms_kmeans, wal_encode_scalar_gpu


class ROMEWALHybrid:
    """ROME rank-1 update + WAL encoding for FFN down_proj editing."""

    def __init__(self, model, tokenizer, K=256, lmax=12, device="cuda:3"):
        self.model = model
        self.tokenizer = tokenizer
        self.K = K
        self.lmax = lmax
        self.device = torch.device(device)
        self.atoms: torch.Tensor | None = None
        self._original_cols: dict[tuple, torch.Tensor] = {}  # (layer, col) → original
        self._edit_count = 0
        self.target_layer = 7  # default knowledge layer

    def build_vocab(self, layer=7):
        """Build WAL atom vocab from target FFN layer's down_proj."""
        self.target_layer = layer
        W = self.model.model.layers[layer].mlp.down_proj.weight.data.float()
        flat = W.flatten()
        self.atoms = build_atoms_kmeans(flat[:2_000_000], self.K, iters=5,
                                         device=self.device)
        print(f"  WAL atoms (layer {layer}): {self.atoms.shape} "
              f"range=[{self.atoms.min():.3f},{self.atoms.max():.3f}]")

    # ═══ ROME math ═══
    def _get_mlp_key(self, token_ids, layer):
        """Get MLP key k = silu(gate) * up at last position → [intermediate]."""
        down = self.model.model.layers[layer].mlp.down_proj
        m_val = None
        def hook(module, inp, out):
            nonlocal m_val
            m_val = inp[0].detach().clone()
        h = down.register_forward_hook(hook)
        with torch.no_grad():
            self.model(input_ids=token_ids.unsqueeze(0).to(self.device))
        h.remove()
        if m_val is None: return None
        return m_val[0, -1, :].float()  # [intermediate]

    def _get_last_hidden(self, token_ids):
        """Last hidden before lm_head."""
        last_h = None
        def hook(module, inp, out):
            nonlocal last_h
            hs = out[0] if isinstance(out, tuple) else out
            last_h = hs.detach().clone()
        h = self.model.model.norm.register_forward_hook(hook)
        with torch.no_grad():
            self.model(input_ids=token_ids.unsqueeze(0).to(self.device))
        h.remove()
        if last_h is None: return None
        return last_h[0, -1, :].float() if last_h.dim() == 3 else last_h[-1, :].float()

    # ═══ edit ═══
    def apply_edit(self, subject, target, relation="", clamp=0.5, prompt=""):
        """ROME rank-1 update to FFN down_proj, encoded via WAL."""
        if self.atoms is None:
            raise RuntimeError("Call build_vocab() first")

        if not prompt:
            prompt = f"{subject} is" if not relation else f"The {relation} of {subject} is"

        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        if not target_ids: return False

        pids = self.tokenizer(prompt, return_tensors="pt").input_ids[0].to(self.device)

        # ── ROME Step 1: get key k at MLP input ──
        k = self._get_mlp_key(pids, self.target_layer)
        if k is None: return False
        k = k / (k.norm() + 1e-8)

        # ── ROME Step 2: get current value v_curr = W @ k ──
        W = self.model.model.layers[self.target_layer].mlp.down_proj
        v_curr = torch.mv(W.weight.data.float(), k)  # [hidden]

        # ── ROME Step 3: compute v* (target direction) ──
        v_star = v_curr.clone()
        for i, tid in enumerate(target_ids):
            tdir = self.model.lm_head.weight.data[tid, :].float()
            v_star += clamp * tdir / (2 ** i)
        v_star = v_star / (v_star.norm() + 1e-8) * v_curr.norm()

        # ── ROME Step 4: rank-1 update ──
        delta_v = v_star - v_curr  # [hidden]
        update = torch.outer(delta_v, k) / (k @ k)  # [hidden, intermediate]

        # ── WAL Step: encode changed columns as programs ──
        atoms_gpu = self.atoms.to(self.device)
        changed_count = 0
        for j in range(W.weight.data.shape[1]):
            col_before = W.weight.data[:, j].float().to(self.device)
            col_after = col_before + update[:, j].to(self.device)

            # Only encode if change is significant
            if (col_after - col_before).abs().max() > 1e-4:
                key = (self.target_layer, j)
                if key not in self._original_cols:
                    self._original_cols[key] = col_before.clone()
                _, _, recon = wal_encode_scalar_gpu(col_after, atoms_gpu, self.lmax)
                W.weight.data[:, j] = recon.to(device=W.weight.device, dtype=W.weight.dtype)
                changed_count += 1

        self._edit_count += 1
        return True

    def rollback(self):
        W = self.model.model.layers[self.target_layer].mlp.down_proj
        for (layer, col), orig in self._original_cols.items():
            W.weight.data[:, col] = orig.to(device=W.weight.device, dtype=W.weight.dtype)
        self._original_cols.clear()
        self._edit_count = 0

    @property
    def edit_count(self): return self._edit_count
