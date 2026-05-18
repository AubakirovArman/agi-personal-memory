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
        self._lm_nt_snapshot: dict[int, torch.Tensor] = {}
        self._emb_nt_snapshot: dict[int, torch.Tensor] = {}
        self._edit_key_basis: list[torch.Tensor] = []
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
    def snapshot_non_target(self, lm_exclude, embed_exclude=None,
                            sample_size: int = 500):
        """Snapshot lm_head and embed non-target rows for NT diff measurement."""
        self._lm_nt_snapshot = self._snapshot_rows(
            self.model.lm_head.weight.data,
            set(lm_exclude),
            sample_size=sample_size,
        )
        self._emb_nt_snapshot = self._snapshot_rows(
            self.model.model.embed_tokens.weight.data,
            set(embed_exclude or ()),
            sample_size=sample_size,
        )

    def measure_non_target_diff(self) -> float:
        """Backward-compatible max over lm_head and embed non-target rows."""
        diffs = self.measure_non_target_diffs()
        return max(diffs.values()) if diffs else 0.0

    def measure_non_target_diffs(self) -> dict[str, float]:
        """Measured max abs diff on non-edited lm_head and embed rows."""
        return {
            "lm_head_non_edited_max": self._max_row_diff(
                self.model.lm_head.weight.data,
                self._lm_nt_snapshot,
            ),
            "embed_non_edited_max": self._max_row_diff(
                self.model.model.embed_tokens.weight.data,
                self._emb_nt_snapshot,
            ),
        }

    # ═══ edit ═══
    def apply_edit(self, subject: str, target: str, relation: str = "",
                   prompt: str = "", clamp_lm: float = 0.20,
                   clamp_embed: float = 0.06, clamp_eos: float = 0.16,
                   clamp_anti: float = 0.06, neg_prompts: list[str] | None = None,
                   old_target: str = "", clamp_old: float = 0.0,
                   target_token_mode: str = "standalone",
                   max_neg_prompts: int = 4,
                   neg_projection_strength: float = 0.3,
                   history_projection_strength: float = 0.0,
                   embed_history_projection_strength: float = 0.0,
                   max_history_keys: int = 128):
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
        history_len = len(self._edit_key_basis)
        new_history_keys: list[torch.Tensor] = []

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
        if clamp_eos > 0 and self.tokenizer.eos_token_id is not None:
            planned_lm_rows.add(self.tokenizer.eos_token_id)
        self.snapshot_non_target(planned_lm_rows, embed_exclude=set(sids))
        neg_keys = self._prompt_keys(neg_prompts or [], max_neg_prompts)

        # ── lm_head: sequence-level boost ──
        if clamp_lm > 0:
            for tids in target_sequences:
                for i, tid in enumerate(tids):
                    ctx = pids if i == 0 else torch.cat([pids, torch.tensor(tids[:i], device=pids.device)])
                    k = self._get_key(ctx)
                    if k is None: continue
                    k = k / (k.norm() + 1e-8)
                    k = self._project_away(
                        k, neg_keys, strength=neg_projection_strength)
                    k = self._project_away(
                        k,
                        self._history_basis(max_history_keys),
                        strength=history_projection_strength,
                    )
                    new_history_keys.append(k.detach().float().cpu())

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
        tdir = self._project_away(
            tdir,
            self._history_basis(max_history_keys),
            strength=embed_history_projection_strength,
        )
        if clamp_embed > 0:
            for sid in sids:
                if sid not in emb_bu: emb_bu[sid] = w_emb[sid, :].clone()
                row = w_emb[sid, :].float().to(self.device)
                _, _, rec = wal_encode_scalar_gpu(row + clamp_embed * tdir, self.atoms_gpu, self.lmax)
                w_emb[sid, :] = rec.to(device=w_emb.device, dtype=w_emb.dtype)

        # ── EOS + anti-boost ──
        eid = self.tokenizer.eos_token_id
        full_ids = torch.cat([pids, torch.tensor(primary_tids, device=pids.device)])
        sk = self._get_key(full_ids) if (clamp_eos > 0 or clamp_anti > 0) else None
        if sk is not None and eid is not None:
            sk = sk / (sk.norm() + 1e-8)
            if clamp_eos > 0:
                if eid not in lm_bu: lm_bu[eid] = w_lm[eid, :].clone()
                er = w_lm[eid, :].float().to(self.device)
                _, _, rec = wal_encode_scalar_gpu(er + clamp_eos * sk.to(self.device), self.atoms_gpu, self.lmax)
                w_lm[eid, :] = rec.to(device=w_lm.device, dtype=w_lm.dtype)
            if clamp_anti > 0:
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
        self._edit_key_basis.extend(new_history_keys)
        return {
            "lm_backup": lm_bu,
            "emb_backup": emb_bu,
            "history_len": history_len,
            "history_keys_added": len(new_history_keys),
        }

    def rollback(self, backup: dict):
        """Exact rollback via clone restoration."""
        for tid, orig in backup.get("lm_backup", {}).items():
            self.model.lm_head.weight.data[tid, :] = orig
        for sid, orig in backup.get("emb_backup", {}).items():
            self.model.model.embed_tokens.weight.data[sid, :] = orig
        if "history_len" in backup:
            self._edit_key_basis = self._edit_key_basis[:backup["history_len"]]

    # ═══ helpers ═══
    def _prompt_keys(self, prompts: list[str], limit: int) -> list[torch.Tensor]:
        keys = []
        for prompt in prompts[:limit]:
            ids = self._prompt_ids(prompt, max_tokens=100)
            key = self._get_key(ids)
            if key is None:
                continue
            keys.append(key / (key.norm() + 1e-8))
        return keys

    def _history_basis(self, limit: int) -> list[torch.Tensor]:
        if limit <= 0:
            return self._edit_key_basis
        return self._edit_key_basis[-limit:]

    @staticmethod
    def _project_away(key: torch.Tensor, basis: list[torch.Tensor],
                      strength: float) -> torch.Tensor:
        if strength <= 0 or not basis:
            return key / (key.norm() + 1e-8)
        out = key / (key.norm() + 1e-8)
        for base in basis:
            b = base.to(out.device).float()
            b = b / (b.norm() + 1e-8)
            dot = torch.dot(out, b)
            if dot > 0:
                out = out - strength * dot * b
                out = out / (out.norm() + 1e-8)
        return out

    @staticmethod
    def _snapshot_rows(weight: torch.Tensor, exclude: set[int],
                       sample_size: int = 500) -> dict[int, torch.Tensor]:
        snapshots: dict[int, torch.Tensor] = {}
        vocab_size = weight.shape[0]
        valid_exclude = {rid for rid in exclude if 0 <= rid < vocab_size}
        target_count = max(0, min(sample_size, vocab_size - len(valid_exclude)))
        attempts = 0
        max_attempts = max(sample_size * 20, 1000)
        while len(snapshots) < target_count and attempts < max_attempts:
            attempts += 1
            rid = torch.randint(0, vocab_size, (1,)).item()
            if rid in valid_exclude or rid in snapshots:
                continue
            snapshots[rid] = weight[rid, :].clone()
        return snapshots

    @staticmethod
    def _max_row_diff(weight: torch.Tensor,
                      snapshots: dict[int, torch.Tensor]) -> float:
        max_diff = 0.0
        for rid, original in snapshots.items():
            diff = (weight[rid, :] - original.to(weight.device)).abs().max().item()
            max_diff = max(max_diff, diff)
        return max_diff

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
