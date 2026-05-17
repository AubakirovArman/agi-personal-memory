"""WAL-FFN editor: frozen vocabulary on FFN down_proj layers.

Key: build atom vocab from down_proj weight distribution,
encode rows as WAL programs, edit by re-encoding modified rows.
Same frozen-vocab guarantee as lm_head → 0% non-target diff.
"""

from __future__ import annotations
import torch
from ..wal.encoder import build_atoms_kmeans, wal_encode_scalar_gpu


class WALFFNEditor:
    """Edit FFN/MLP down_proj weights via WAL programs.

    Targets mid-to-late layers (15-31) where factual knowledge is stored.
    lm_head + FFN hybrid: lm_head gives output, FFN gives specificity.
    """

    def __init__(self, model, tokenizer, K=256, lmax=12, device="cuda:3"):
        self.model = model
        self.tokenizer = tokenizer
        self.K = K
        self.lmax = lmax
        self.device = torch.device(device)

        # Atom tables per layer
        self.ffn_atoms: dict[int, torch.Tensor] = {}
        self._original_weights: dict[str, torch.Tensor] = {}
        self._edit_count = 0
        self._vocab_size, self._hidden_size = 0, 0

    def build_vocab(self, layers: list[int] = [7, 15, 23, 29, 31]):
        """Build WAL atom vocab for each target FFN layer."""
        for lidx in layers:
            w = self.model.model.layers[lidx].mlp.down_proj.weight.data.float()
            flat = w.flatten()
            atoms = build_atoms_kmeans(flat[:2_000_000], self.K, iters=5,
                                       device=self.device)
            self.ffn_atoms[lidx] = atoms
            print(f"  Layer {lidx}: atoms={atoms.shape} range=[{atoms.min():.3f},{atoms.max():.3f}]")
        self._hidden_size = w.shape[0]
        self._vocab_size = self.model.lm_head.weight.shape[0]

    # ── helpers ──
    def _get_last_hidden(self, token_ids):
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

    def _get_mlp_hidden(self, token_ids, layer_idx):
        """Get hidden state at down_proj INPUT (= silu(gate)*up output, [14336])."""
        down_proj = self.model.model.layers[layer_idx].mlp.down_proj
        hs = None
        def hook(module, inp, out):
            nonlocal hs
            hs = inp[0].detach().clone()  # inp[0] = [batch, seq, intermediate(14336)]
        h = down_proj.register_forward_hook(hook)
        with torch.no_grad():
            self.model(input_ids=token_ids.unsqueeze(0).to(self.device))
        h.remove()
        if hs is None: return None
        k = hs[0, -1, :].float()  # [intermediate]
        k = k / (k.norm() + 1e-8)
        return k

    # ── FFN edit via WAL ──
    def _edit_ffn_row(self, layer_idx, key, direction, clamp, negative=False):
        """Edit one row of down_proj via WAL programs.

        down_proj: [hidden, intermediate] - each row maps intermediate→hidden.
        We edit the row(s) most aligned with 'direction' in hidden space.
        Instead of touching all rows (rank-1 update), we edit only the top-K
        rows most aligned with direction.
        """
        atoms = self.ffn_atoms.get(layer_idx)
        if atoms is None:
            return

        W = self.model.model.layers[layer_idx].mlp.down_proj.weight
        name = f"layer_{layer_idx}_down"
        if name not in self._original_weights:
            self._original_weights[name] = W.data.clone()

        atoms_gpu = atoms.to(self.device)
        direction = direction / (direction.norm() + 1e-8)
        sign = -1.0 if negative else 1.0

        # Edit top COLUMNS most aligned with direction (columns = hidden-space vectors)
        W_T = W.data.float().T  # [intermediate, hidden] — each row is a hidden-space vector
        k = min(50, W_T.shape[0])
        if negative:
            k = min(100, W_T.shape[0])

        scores = torch.abs(torch.mv(W_T, direction.to(W.device)))
        _, top_cols = torch.topk(scores, k)

        for cid in top_cols:
            col = W.data[:, cid.item()].float().to(self.device)
            boost = sign * clamp * direction.to(self.device) * (1.0 / (1 + cid.item() % 5))
            new_col = col + boost
            _, _, recon = wal_encode_scalar_gpu(new_col, atoms_gpu, self.lmax)
            W.data[:, cid.item()] = recon.to(device=W.device, dtype=W.dtype)

    # ── combined edit ──
    def apply_edit(self, subject, target, relation="", clamp_lm=0.15,
                   clamp_ffn_pos=0.03, clamp_ffn_neg=0.01, prompt="",
                   ffn_layers=[7, 15, 23]):
        """Hybrid WAL edit: light lm_head + FFN for specificity.

        lm_head: weak boost (clamp_lm=0.15, NS=98% but ES=7%)
        FFN positive: boost target direction on edit context (→ ES↑)
        FFN negative: anti-boost target direction on after-target + neighbors (→ NS↑)
        """
        if not prompt:
            prompt = f"{subject} is" if not relation else f"The {relation} of {subject} is"

        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        if not target_ids: return False

        # Contexts
        pids = self.tokenizer(prompt, return_tensors="pt").input_ids[0].to(self.device)
        after_ids = torch.cat([pids, torch.tensor(target_ids, device=self.device)])

        # Direction: lm_head weight of target tokens
        target_dir = torch.zeros(self._hidden_size, device=self.device)
        for i, tid in enumerate(target_ids):
            w = self.model.lm_head.weight.data[tid, :].float()
            target_dir += w / (2 ** i)
        target_dir = target_dir / (target_dir.norm() + 1e-8)

        # ── FFN positive: boost target direction on edit context ──
        for lidx in ffn_layers:
            key = self._get_mlp_hidden(pids.unsqueeze(0), lidx)
            if key is not None:
                direction = self.model.lm_head.weight.data[target_ids[0], :].float()
                direction = direction / (direction.norm() + 1e-8)
                self._edit_ffn_row(lidx, key.to(self.device), direction.to(self.device),
                                   clamp_ffn_pos, negative=False)

        # ── FFN negative: anti-boost on after-target context ──
        for lidx in ffn_layers:
            key = self._get_mlp_hidden(after_ids.unsqueeze(0), lidx)
            if key is not None:
                direction = self.model.lm_head.weight.data[target_ids[0], :].float()
                direction = direction / (direction.norm() + 1e-8)
                self._edit_ffn_row(lidx, key.to(self.device), direction.to(self.device),
                                   clamp_ffn_neg, negative=True)

        self._edit_count += 1
        return True

    def rollback(self):
        for name, orig in self._original_weights.items():
            parts = name.split("_")
            lidx = int(parts[1])
            self.model.model.layers[lidx].mlp.down_proj.weight.data = orig
        self._original_weights.clear()
        self._edit_count = 0
