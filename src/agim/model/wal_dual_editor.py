"""WAL dual-layer editor for lm_head and embed_tokens edits."""
from __future__ import annotations
import torch
from .wal_dual_helpers import (
    add_relation_protected_keys,
    combine_positive_keys,
    get_key,
    history_basis,
    max_row_diff,
    positive_keys_for_step,
    project_away,
    project_away_orthogonal,
    primary_target_sequence,
    prompt_ids,
    prompt_keys,
    relation_protected_basis,
    snapshot_rows,
    target_sequences,
)
from .wal_dual_state import (
    build_vocab,
    measure_non_target_diff,
    measure_non_target_diffs,
    snapshot_non_target,
)
from .wal_row_update import add_row_delta


class WALDualLayerEditor:
    """Edit lm_head and embed_tokens via WAL programs."""

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
        self._relation_key_basis: dict[str, list[torch.Tensor]] = {}
        self._relation_protected_basis: dict[str, list[torch.Tensor]] = {}
        self._edit_count = 0
        self.nt_sample_size = 500

    def apply_edit(self, subject: str, target: str, relation: str = "",
                   prompt: str = "", clamp_lm: float = 0.20,
                   clamp_embed: float = 0.06, clamp_eos: float = 0.0,
                   clamp_anti: float = 0.06, neg_prompts: list[str] | None = None,
                   old_target: str = "", clamp_old: float = 0.0,
                   target_token_mode: str = "standalone",
                   positive_prompts: list[str] | None = None,
                   max_positive_prompts: int = 4,
                   positive_key_weight: float = 1.0,
                   positive_constraint_mode: str = "none",
                   max_neg_prompts: int = 4,
                   neg_projection_strength: float = 0.3,
                   history_projection_strength: float = 0.0,
                   embed_history_projection_strength: float = 0.0,
                   projection_mode: str = "sequential",
                   history_slot_mode: str = "global",
                   max_history_keys: int = 128,
                   relation_protected_mode: str = "none",
                   max_relation_protected_keys: int = 64,
                   wal_encode_updates: bool = True):
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
        wal_encode_updates: when false, writes exact additive row updates for
            ablations instead of round-tripping through WAL reconstruction.
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
        if history_slot_mode not in {"global", "relation"}:
            raise ValueError("history_slot_mode must be one of: global, relation")
        if relation_protected_mode not in {"none", "accumulate", "preload"}:
            raise ValueError(
                "relation_protected_mode must be one of: none, accumulate, preload")
        if positive_constraint_mode not in {"none", "projected", "ridge"}:
            raise ValueError(
                "positive_constraint_mode must be one of: none, projected, ridge")
        relation_key = str(relation or "")
        history_len = len(self._edit_key_basis)
        relation_history = self._relation_key_basis.get(relation_key, [])
        relation_history_len = len(relation_history)
        relation_protected_len = len(
            self._relation_protected_basis.get(relation_key, []))
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
        primary_tids = self._primary_target_sequence(
            prompt, target, standalone_tids, target_token_mode)
        target_lm_rows = {tid for seq in target_sequences for tid in seq}
        old_lm_rows = {tid for seq in old_sequences for tid in seq}

        lm_bu = {}; emb_bu = {}
        w_lm = self.model.lm_head.weight.data
        w_emb = self.model.model.embed_tokens.weight.data

        planned_lm_rows = set(target_lm_rows)
        planned_lm_rows.update(old_lm_rows)
        if clamp_eos > 0 and self.tokenizer.eos_token_id is not None:
            planned_lm_rows.add(self.tokenizer.eos_token_id)
        self.snapshot_non_target(
            planned_lm_rows, embed_exclude=set(sids), sample_size=self.nt_sample_size)
        neg_keys = self._prompt_keys(neg_prompts or [], max_neg_prompts)
        protected_keys = list(neg_keys)
        if relation_protected_mode != "none":
            protected_keys.extend(self._relation_protected_bank(
                relation_key, max_relation_protected_keys))
        positive_prompts = positive_prompts or []

        if clamp_lm > 0:
            for tids in target_sequences:
                for i, tid in enumerate(tids):
                    ctx = pids if i == 0 else torch.cat([pids, torch.tensor(tids[:i], device=pids.device)])
                    k = self._get_key(ctx)
                    if k is None: continue
                    k = k / (k.norm() + 1e-8)
                    positive_keys = self._positive_keys_for_step(
                        positive_prompts, tids, i, max_positive_prompts)
                    protected = (
                        protected_keys if positive_constraint_mode != "none" else None
                    )
                    k = self._combine_positive_keys(
                        k, positive_keys, positive_key_weight, protected,
                        neg_projection_strength, projection_mode,
                        positive_constraint_mode)
                    k = self._project_away(
                        k, protected_keys, strength=neg_projection_strength,
                        mode=projection_mode)
                    k = self._project_away(
                        k,
                        self._history_basis(
                            max_history_keys, relation_key, history_slot_mode),
                        strength=history_projection_strength,
                        mode=projection_mode,
                    )
                    new_history_keys.append(k.detach().float().cpu())

                    if tid not in lm_bu: lm_bu[tid] = w_lm[tid, :].clone()
                    add_row_delta(
                        w_lm, tid, clamp_lm * k.to(self.device),
                        self.atoms_gpu, self.lmax, wal_encode_updates)

        if clamp_old > 0 and old_lm_rows:
            for otid in old_lm_rows:
                if otid not in lm_bu: lm_bu[otid] = w_lm[otid, :].clone()
                # Use the SAME context key as the new target (edit prompt context)
                ctx = pids  # edit prompt context
                k_old = self._get_key(ctx)
                if k_old is not None:
                    k_old = k_old / (k_old.norm() + 1e-8)
                    add_row_delta(
                        w_lm, otid, -clamp_old * k_old.to(self.device),
                        self.atoms_gpu, self.lmax, wal_encode_updates)

        if old_lm_rows and old_target:
            new_dir = w_lm[primary_tids[0], :].float().to(self.device)
            old_primary = old_sequences[0] if old_sequences else []
            old_dir = w_lm[old_primary[0], :].float().to(self.device) if old_primary else 0
            tdir = new_dir - 0.5 * old_dir
            tdir = tdir / (tdir.norm() + 1e-8)
        else:
            tdir = w_lm[primary_tids[0], :].float().to(self.device)
            tdir = tdir / (tdir.norm() + 1e-8)
        tdir = self._project_away(
            tdir,
            self._history_basis(max_history_keys, relation_key, history_slot_mode),
            strength=embed_history_projection_strength,
            mode=projection_mode,
        )
        if clamp_embed > 0:
            for sid in sids:
                if sid not in emb_bu: emb_bu[sid] = w_emb[sid, :].clone()
                add_row_delta(
                    w_emb, sid, clamp_embed * tdir,
                    self.atoms_gpu, self.lmax, wal_encode_updates)

        eid = self.tokenizer.eos_token_id
        full_ids = torch.cat([pids, torch.tensor(primary_tids, device=pids.device)])
        sk = self._get_key(full_ids) if (clamp_eos > 0 or clamp_anti > 0) else None
        if sk is not None and eid is not None:
            sk = sk / (sk.norm() + 1e-8)
            if clamp_eos > 0:
                if eid not in lm_bu: lm_bu[eid] = w_lm[eid, :].clone()
                add_row_delta(
                    w_lm, eid, clamp_eos * sk.to(self.device),
                    self.atoms_gpu, self.lmax, wal_encode_updates)
            if clamp_anti > 0:
                for tid in target_lm_rows:
                    if tid == eid: continue
                    add_row_delta(
                        w_lm, tid, -clamp_anti * sk.to(self.device),
                        self.atoms_gpu, self.lmax, wal_encode_updates)
                for sid in sids:
                    if sid not in emb_bu: emb_bu[sid] = w_emb[sid, :].clone()
                    add_row_delta(
                        w_emb, sid, -clamp_anti * 0.5 * sk.to(self.device),
                        self.atoms_gpu, self.lmax, wal_encode_updates)

        self._edit_count += 1
        self._edit_key_basis.extend(new_history_keys)
        self._relation_key_basis.setdefault(relation_key, []).extend(new_history_keys)
        if relation_protected_mode == "accumulate":
            self._add_relation_protected_keys(
                relation_key, neg_keys, max_relation_protected_keys)
        return {
            "lm_backup": lm_bu,
            "emb_backup": emb_bu,
            "history_len": history_len,
            "relation_key": relation_key,
            "relation_history_len": relation_history_len,
            "relation_protected_len": relation_protected_len,
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
        if "relation_key" in backup and "relation_history_len" in backup:
            key = backup["relation_key"]
            self._relation_key_basis[key] = self._relation_key_basis.get(key, [])[
                :backup["relation_history_len"]
            ]
        if "relation_key" in backup and "relation_protected_len" in backup:
            key = backup["relation_key"]
            self._relation_protected_basis[key] = self._relation_protected_basis.get(
                key, [])[:backup["relation_protected_len"]]

    _prompt_keys = prompt_keys
    _history_basis = history_basis
    _relation_protected_bank = relation_protected_basis
    _add_relation_protected_keys = add_relation_protected_keys
    _positive_keys_for_step = positive_keys_for_step
    _get_key = get_key
    _target_sequences = target_sequences
    _primary_target_sequence = primary_target_sequence
    _prompt_ids = prompt_ids
    _combine_positive_keys = staticmethod(combine_positive_keys)
    _project_away = staticmethod(project_away)
    _project_away_orthogonal = staticmethod(project_away_orthogonal)
    _snapshot_rows = staticmethod(snapshot_rows)
    _max_row_diff = staticmethod(max_row_diff)
    build_vocab = build_vocab
    snapshot_non_target = snapshot_non_target
    measure_non_target_diff = measure_non_target_diff
    measure_non_target_diffs = measure_non_target_diffs

    @property
    def edit_count(self) -> int:
        return self._edit_count
