"""WAL Dual-Layer Editor — lm_head + embed_tokens via frozen vocabulary.

Canonical implementation extracted from test_dual_layer.py (verified working).
Edits BOTH input (embed_tokens) and output (lm_head) for subject-specific editing.
"""
from __future__ import annotations
import torch
from ..wal.encoder import build_atoms_kmeans, wal_encode_scalar_gpu
from .wal_dual_helpers import (
    combine_positive_keys,
    contextual_target_ids,
    get_key,
    history_basis,
    max_row_diff,
    positive_keys_for_step,
    project_away,
    project_away_orthogonal,
    prompt_ids,
    prompt_keys,
    snapshot_rows,
    target_sequences,
)


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
                   positive_prompts: list[str] | None = None,
                   max_positive_prompts: int = 4,
                   positive_key_weight: float = 1.0,
                   max_neg_prompts: int = 4,
                   neg_projection_strength: float = 0.3,
                   history_projection_strength: float = 0.0,
                   embed_history_projection_strength: float = 0.0,
                   projection_mode: str = "sequential",
                   max_history_keys: int = 128):
        """Dual-layer WAL edit. Returns dict with backup info for rollback.

        old_target: if provided, use (new - old) direction (subject-conditioned gate).
        clamp_old: if > 0, anti-boost old target token in edit context.
        target_token_mode: standalone uses target text tokenized alone; contextual
            uses EasyEdit-style prompt + " " + target continuation ids; both edits
            both tokenizations.
        positive_prompts: optional paraphrase prompts used to average the edit
            key toward a multi-positive direction for better rephrase coverage.
        projection_mode: sequential applies the historical component-by-component
            projection; orthogonal removes the whole protected subspace at once.
        """
        if self.atoms is None or self.atoms_gpu is None:
            raise RuntimeError("Call build_vocab() first")
        if target_token_mode not in {"standalone", "contextual", "both"}:
            raise ValueError(
                "target_token_mode must be one of: standalone, contextual, both"
            )
        if projection_mode not in {"sequential", "orthogonal"}:
            raise ValueError(
                "projection_mode must be one of: sequential, orthogonal"
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
        positive_prompts = positive_prompts or []

        # ── lm_head: sequence-level boost ──
        if clamp_lm > 0:
            for tids in target_sequences:
                for i, tid in enumerate(tids):
                    ctx = pids if i == 0 else torch.cat([pids, torch.tensor(tids[:i], device=pids.device)])
                    k = self._get_key(ctx)
                    if k is None: continue
                    k = k / (k.norm() + 1e-8)
                    positive_keys = self._positive_keys_for_step(
                        positive_prompts, tids, i, max_positive_prompts)
                    k = self._combine_positive_keys(
                        k, positive_keys, positive_key_weight)
                    k = self._project_away(
                        k, neg_keys, strength=neg_projection_strength,
                        mode=projection_mode)
                    k = self._project_away(
                        k,
                        self._history_basis(max_history_keys),
                        strength=history_projection_strength,
                        mode=projection_mode,
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
            mode=projection_mode,
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

    _prompt_keys = prompt_keys
    _history_basis = history_basis
    _positive_keys_for_step = positive_keys_for_step
    _get_key = get_key
    _contextual_target_ids = contextual_target_ids
    _target_sequences = target_sequences
    _prompt_ids = prompt_ids
    _combine_positive_keys = staticmethod(combine_positive_keys)
    _project_away = staticmethod(project_away)
    _project_away_orthogonal = staticmethod(project_away_orthogonal)
    _snapshot_rows = staticmethod(snapshot_rows)
    _max_row_diff = staticmethod(max_row_diff)

    @property
    def edit_count(self) -> int:
        return self._edit_count
