"""WAL Dual-Layer Editor — lm_head + embed_tokens via frozen vocabulary.

Canonical implementation extracted from test_dual_layer.py (verified working).
Edits BOTH input (embed_tokens) and output (lm_head) for subject-specific editing.
"""
from __future__ import annotations
import torch
from ..wal.encoder import build_atoms_kmeans, wal_encode_scalar_gpu


class WALDualLayerEditor:
    """Edit lm_head AND embed_tokens via WAL programs.

    lm_head: sequence-level target token boost (output layer)
    embed_tokens: push subject tokens toward target direction (input layer)
    + EOS boost + anti-boost on both layers for repetition control.

    Verified NS: 72% at clamp=0.15, 42% at clamp=0.20 (CounterFact 50).
    """

    def __init__(self, model, tokenizer, K=256, lmax=16, device="cuda:3"):
        self.model = model
        self.tokenizer = tokenizer
        self.K = K
        self.lmax = lmax
        self.device = torch.device(device)
        self.atoms: torch.Tensor | None = None
        self.atoms_gpu: torch.Tensor | None = None
        self._lm_original: dict[int, torch.Tensor] = {}
        self._emb_original: dict[int, torch.Tensor] = {}
        self._nt_snapshot: dict[int, torch.Tensor] = {}  # for measured NT
        self._edit_count = 0

    # ═══ vocabulary ═══
    def build_vocab(self):
        """Build shared WAL atoms from lm_head + embed_tokens distributions."""
        lm_flat = self.model.lm_head.weight.data.float().flatten()
        emb_flat = self.model.model.embed_tokens.weight.data.float().flatten()
        combined = torch.cat([lm_flat[:2_000_000], emb_flat[:2_000_000]])
        self.atoms = build_atoms_kmeans(combined, self.K, iters=5, device=self.device)
        self.atoms_gpu = self.atoms.to(self.device)
        print(f"  WAL atoms: {self.atoms.shape} [{self.atoms.min():.3f}, {self.atoms.max():.3f}]")

    # ═══ NT measurement ═══
    def snapshot_non_target(self, target_ids, extra_exclude=None):
        """Snapshot random non-target rows for later NT diff measurement."""
        weight = self.model.lm_head.weight.data
        edited = set(target_ids)
        if extra_exclude:
            edited.update(extra_exclude)
        self._nt_snapshot = {}
        for _ in range(500):
            rid = torch.randint(0, weight.shape[0], (1,)).item()
            if rid not in edited and rid not in self._nt_snapshot:
                self._nt_snapshot[rid] = weight[rid, :].clone()

    def measure_non_target_diff(self) -> float:
        """Measured (not fake) max absolute diff on non-target lm_head rows."""
        if not self._nt_snapshot:
            return 0.0
        weight = self.model.lm_head.weight.data
        max_diff = 0.0
        for rid, orig in self._nt_snapshot.items():
            diff = (weight[rid, :] - orig.to(weight.device)).abs().max().item()
            max_diff = max(max_diff, diff)
        return max_diff

    # ═══ edit ═══
    def apply_edit(self, subject: str, target: str, relation: str = "",
                   prompt: str = "", clamp_lm: float = 0.20,
                   clamp_embed: float = 0.06, clamp_eos: float = 0.16,
                   clamp_anti: float = 0.06, neg_prompts: list[str] | None = None,
                   old_target: str = "", clamp_old: float = 0.0,
                   target_token_mode: str = "standalone",
                   max_neg_prompts: int = 4):
        """Dual-layer WAL edit. Returns dict with backup info for rollback.

        old_target: if provided, use (new - old) direction (subject-conditioned gate).
        clamp_old: if > 0, anti-boost old target token in edit context.
        target_token_mode: standalone uses target text tokenized alone; contextual
            uses EasyEdit-style prompt + " " + target continuation ids; both edits
            both tokenizations.
        """
        if self.atoms is None or self.atoms_gpu is None:
            raise RuntimeError("Call build_vocab() first")
        if target_token_mode not in {"standalone", "contextual", "both"}:
            raise ValueError(
                "target_token_mode must be one of: standalone, contextual, both"
            )

        if not prompt:
            prompt = f"{subject} is" if not relation else f"The {relation} of {subject} is"

        standalone_tids = self.tokenizer.encode(target, add_special_tokens=False)
        old_standalone_tids = (
            self.tokenizer.encode(old_target, add_special_tokens=False)
            if old_target else []
        )
        pids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
        sids = self.tokenizer.encode(subject, add_special_tokens=False)
        target_sequences = self._target_sequences(
            prompt, target, standalone_tids, target_token_mode)
        old_sequences = (
            self._target_sequences(
                prompt, old_target, old_standalone_tids, target_token_mode)
            if old_target else []
        )
        primary_tids = target_sequences[0] if target_sequences else standalone_tids
        target_lm_rows = {tid for seq in target_sequences for tid in seq}
        old_lm_rows = {tid for seq in old_sequences for tid in seq}

        lm_bu = {}; emb_bu = {}
        w_lm = self.model.lm_head.weight.data
        w_emb = self.model.model.embed_tokens.weight.data

        # NT snapshot. Exclude every lm_head row this edit may intentionally touch.
        planned_lm_rows = set(target_lm_rows)
        planned_lm_rows.update(old_lm_rows)
        if self.tokenizer.eos_token_id is not None:
            planned_lm_rows.add(self.tokenizer.eos_token_id)
        self.snapshot_non_target(target_lm_rows, extra_exclude=planned_lm_rows)

        # ── lm_head: sequence-level boost ──
        for tids in target_sequences:
            for i, tid in enumerate(tids):
                ctx = pids if i == 0 else torch.cat([pids, torch.tensor(tids[:i], device=pids.device)])
                k = self._get_key(ctx)
                if k is None: continue
                k = k / (k.norm() + 1e-8)

                # Negative projection (if provided)
                if neg_prompts and len(neg_prompts) > 0:
                    neg_keys = []
                    for npr in neg_prompts[:max_neg_prompts]:
                        nids = self._prompt_ids(npr, max_tokens=100)
                        nk = self._get_key(nids)
                        if nk is not None: neg_keys.append(nk / (nk.norm() + 1e-8))
                    for nk in neg_keys:
                        dot = torch.dot(k, nk)
                        if dot > 0: k = k - 0.3 * dot * nk
                    k = k / (k.norm() + 1e-8)

                if tid not in lm_bu: lm_bu[tid] = w_lm[tid, :].clone()
                row = w_lm[tid, :].float().to(self.device)
                _, _, rec = wal_encode_scalar_gpu(row + clamp_lm * k.to(self.device), self.atoms_gpu, self.lmax)
                if rec.shape != row.shape:
                    raise RuntimeError(
                        f"WAL reconstruction shape mismatch: {rec.shape} != {row.shape}"
                    )
                w_lm[tid, :] = rec.to(device=w_lm.device, dtype=w_lm.dtype)

        # ── Old-target anti-boost (before embed edit) ──
        if clamp_old > 0 and old_lm_rows:
            for otid in old_lm_rows:
                if otid not in lm_bu: lm_bu[otid] = w_lm[otid, :].clone()
                row_old = w_lm[otid, :].float().to(self.device)
                # Use the SAME context key as the new target (edit prompt context)
                ctx = pids  # edit prompt context
                k_old = self._get_key(ctx)
                if k_old is not None:
                    k_old = k_old / (k_old.norm() + 1e-8)
                    _, _, rec_old = wal_encode_scalar_gpu(
                        row_old - clamp_old * k_old.to(self.device), self.atoms_gpu, self.lmax)
                    w_lm[otid, :] = rec_old.to(device=w_lm.device, dtype=w_lm.dtype)

        # ── embed_tokens: push subject toward (target_new - target_old) direction ──
        if old_lm_rows and old_target:
            # Subject-conditioned gate: new_dir - old_dir
            new_dir = w_lm[primary_tids[0], :].float().to(self.device)
            old_primary = old_sequences[0] if old_sequences else []
            old_dir = w_lm[old_primary[0], :].float().to(self.device) if old_primary else 0
            tdir = new_dir - 0.5 * old_dir  # push away from old, toward new
            tdir = tdir / (tdir.norm() + 1e-8)
        else:
            tdir = w_lm[primary_tids[0], :].float().to(self.device)
            tdir = tdir / (tdir.norm() + 1e-8)
        for sid in sids:
            if sid not in emb_bu: emb_bu[sid] = w_emb[sid, :].clone()
            row = w_emb[sid, :].float().to(self.device)
            _, _, rec = wal_encode_scalar_gpu(row + clamp_embed * tdir, self.atoms_gpu, self.lmax)
            w_emb[sid, :] = rec.to(device=w_emb.device, dtype=w_emb.dtype)

        # ── EOS + anti-boost ──
        eid = self.tokenizer.eos_token_id
        full_ids = torch.cat([pids, torch.tensor(primary_tids, device=pids.device)])
        sk = self._get_key(full_ids)
        if sk is not None and eid is not None:
            sk = sk / (sk.norm() + 1e-8)
            if eid not in lm_bu: lm_bu[eid] = w_lm[eid, :].clone()
            er = w_lm[eid, :].float().to(self.device)
            _, _, rec = wal_encode_scalar_gpu(er + clamp_eos * sk.to(self.device), self.atoms_gpu, self.lmax)
            w_lm[eid, :] = rec.to(device=w_lm.device, dtype=w_lm.dtype)
            for tid in target_lm_rows:
                if tid == eid: continue
                r2 = w_lm[tid, :].float().to(self.device)
                _, _, ar = wal_encode_scalar_gpu(r2 - clamp_anti * sk.to(self.device), self.atoms_gpu, self.lmax)
                w_lm[tid, :] = ar.to(device=w_lm.device, dtype=w_lm.dtype)
            for sid in sids:
                if sid not in emb_bu: emb_bu[sid] = w_emb[sid, :].clone()
                r3 = w_emb[sid, :].float().to(self.device)
                _, _, ar = wal_encode_scalar_gpu(r3 - clamp_anti * 0.5 * sk.to(self.device), self.atoms_gpu, self.lmax)
                w_emb[sid, :] = ar.to(device=w_emb.device, dtype=w_emb.dtype)

        self._edit_count += 1
        return {"lm_backup": lm_bu, "emb_backup": emb_bu}

    def rollback(self, backup: dict):
        """Exact rollback via clone restoration."""
        for tid, orig in backup.get("lm_backup", {}).items():
            self.model.lm_head.weight.data[tid, :] = orig
        for sid, orig in backup.get("emb_backup", {}).items():
            self.model.model.embed_tokens.weight.data[sid, :] = orig

    # ═══ helpers ═══
    def _get_key(self, token_ids: torch.Tensor) -> torch.Tensor | None:
        last_h = None
        def hook(m, i, o):
            nonlocal last_h; hs = o[0] if isinstance(o, tuple) else o; last_h = hs.detach().clone()
        h = self.model.model.norm.register_forward_hook(hook)
        with torch.no_grad():
            self.model(input_ids=token_ids.unsqueeze(0).to(self.device))
        h.remove()
        if last_h is None: return None
        return last_h[0, -1, :].float() if last_h.dim() == 3 else last_h[-1, :].float()

    def _contextual_target_ids(self, prompt: str, target: str) -> list[int]:
        """Ids EasyEdit teacher-forcing labels for prompt + " " + target."""
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
        full_ids = self.tokenizer(f"{prompt} {target}", return_tensors="pt").input_ids[0]
        suffix = full_ids[len(prompt_ids):].detach().cpu().tolist()
        if suffix:
            return suffix
        return self.tokenizer.encode(target, add_special_tokens=False)

    def _target_sequences(
        self,
        prompt: str,
        target: str,
        standalone_ids: list[int],
        mode: str,
    ) -> list[list[int]]:
        sequences: list[list[int]] = []
        if mode in {"standalone", "both"}:
            sequences.append(list(standalone_ids))
        if mode in {"contextual", "both"}:
            sequences.append(self._contextual_target_ids(prompt, target))

        deduped: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        for seq in sequences:
            key = tuple(seq)
            if seq and key not in seen:
                deduped.append(seq)
                seen.add(key)
        return deduped

    def _prompt_ids(self, prompt: str, max_tokens: int | None = None) -> torch.Tensor:
        kwargs = {"return_tensors": "pt"}
        if max_tokens is not None:
            kwargs.update({"truncation": True, "max_length": max_tokens})
        return self.tokenizer(prompt, **kwargs).input_ids[0]

    @property
    def edit_count(self) -> int:
        return self._edit_count
