"""WAL-audited ROME-style FFN row editor."""
from __future__ import annotations

from typing import Any

import torch

from ..wal.encoder import build_atoms_kmeans
from .wal_dual_helpers import contextual_target_ids, max_row_diff, snapshot_rows
from .wal_row_update import add_row_delta


class WALRomeEditor:
    """Sparse located FFN backend with rollback-compatible row backups.

    This is a practical Path B bridge: it moves the edit locus from
    lm_head/embed rows into a selected MLP down_proj layer while keeping the
    patch surface sparse enough for WAL-style audit, budgets, and rollback.
    """

    def __init__(
        self,
        model,
        tokenizer,
        K: int = 256,
        lmax: int = 16,
        device: str = "cuda:3",
        target_layer: int = 7,
        candidate_layers: list[int] | None = None,
        top_rows: int = 32,
        clamp_rome: float = 0.08,
        auto_locate: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.K = K
        self.lmax = lmax
        self.device = torch.device(device)
        self.target_layer = int(target_layer)
        self.candidate_layers = candidate_layers
        self.top_rows = int(top_rows)
        self.clamp_rome = float(clamp_rome)
        self.auto_locate = bool(auto_locate)
        self.atoms: torch.Tensor | None = None
        self.atoms_gpu: torch.Tensor | None = None
        self._ffn_nt_snapshot: dict[tuple[int, int], torch.Tensor] = {}
        self._lm_nt_snapshot: dict[int, torch.Tensor] = {}
        self._emb_nt_snapshot: dict[int, torch.Tensor] = {}
        self._edit_count = 0
        self.nt_sample_size = 500

    def build_vocab(self, layer: int | None = None, sample_size: int = 2_000_000) -> None:
        layer_idx = self._valid_layer(self.target_layer if layer is None else layer)
        flat = self.ffn_weight(layer_idx).detach().float().flatten()
        sample = flat[: min(sample_size, flat.numel())]
        k = max(1, min(self.K, sample.numel()))
        self.atoms = build_atoms_kmeans(sample, k, iters=5, device=self.device)
        self.atoms_gpu = self.atoms.to(self.device)

    def apply_edit(
        self,
        subject: str,
        target: str,
        relation: str = "",
        prompt: str = "",
        old_target: str = "",
        target_token_mode: str = "contextual",
        wal_encode_updates: bool = True,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        if wal_encode_updates and self.atoms_gpu is None:
            raise RuntimeError("Call build_vocab() first")
        if not prompt:
            prompt = f"The {relation} of {subject} is" if relation else f"{subject} is"
        layer_idx = self._edit_layer(prompt, target)
        captured = self._capture_down_proj(prompt, layer_idx, subject)
        if captured is None:
            return self._empty_backup(layer_idx)
        key, value = captured
        direction, target_ids = self._target_direction(
            prompt, target, old_target, target_token_mode)
        if direction is None:
            return self._empty_backup(layer_idx)

        key = key / (key.norm() + 1e-8)
        value_scale = max(float(value.detach().float().norm().item()), 1.0)
        scale = self.clamp_rome * value_scale / max(direction.numel() ** 0.5, 1.0)
        rows = self._top_direction_rows(direction)
        weight = self.ffn_weight(layer_idx)
        self._snapshot_ffn_non_target(layer_idx, rows)
        ffn_backup: dict[tuple[int, int], torch.Tensor] = {}
        atoms = self.atoms_gpu if self.atoms_gpu is not None else torch.empty(0)

        for row_id in rows:
            ffn_backup[(layer_idx, row_id)] = weight[row_id, :].detach().clone()
            row_delta = scale * float(direction[row_id].item()) * key.to(self.device)
            add_row_delta(
                weight, row_id, row_delta, atoms, self.lmax, wal_encode_updates)

        self._edit_count += 1
        return {
            "lm_backup": {},
            "emb_backup": {},
            "ffn_backup": ffn_backup,
            "ffn_layer": layer_idx,
            "ffn_rows": rows,
            "metadata": {
                "subject_token_ids": self._encode(subject),
                "target_token_ids": target_ids,
                "ffn_layer": layer_idx,
                "ffn_rows": rows,
                "edit_locus": "mlp.down_proj",
            },
        }

    def rollback(self, backup: dict[str, Any] | None = None) -> None:
        if not backup:
            return
        for (layer_idx, row_id), original in backup.get("ffn_backup", {}).items():
            weight = self.ffn_weight(int(layer_idx))
            weight[int(row_id), :] = original.to(device=weight.device, dtype=weight.dtype)
        self._edit_count = max(0, self._edit_count - 1)

    def measure_non_target_diffs(self) -> dict[str, float]:
        max_ffn = 0.0
        by_layer: dict[int, dict[int, torch.Tensor]] = {}
        for (layer_idx, row_id), before in self._ffn_nt_snapshot.items():
            by_layer.setdefault(layer_idx, {})[row_id] = before
        for layer_idx, snapshots in by_layer.items():
            max_ffn = max(max_ffn, max_row_diff(self.ffn_weight(layer_idx), snapshots))
        return {
            "lm_head_non_edited_max": 0.0,
            "embed_non_edited_max": 0.0,
            "ffn_down_proj_non_edited_max": max_ffn,
        }

    def ffn_weight(self, layer_idx: int) -> torch.Tensor:
        return self.model.model.layers[int(layer_idx)].mlp.down_proj.weight.data

    def locate_layer(self, prompt: str, target: str) -> int:
        target_ids = self._target_ids(prompt, target, "contextual")
        if not target_ids:
            return self._valid_layer(self.target_layer)
        inputs = self._prompt_ids(prompt)
        target_id = int(target_ids[0])
        with torch.no_grad():
            clean = self.model(input_ids=inputs.unsqueeze(0).to(self.device))
            clean_logit = float(clean.logits[0, -1, target_id].item())
        scored = []
        for layer_idx in self._candidate_layers():
            layer_idx = self._valid_layer(layer_idx)
            module = self.model.model.layers[layer_idx].mlp.down_proj

            def hook(_module, _inp, output):
                scale = output.detach().float().std().clamp_min(1e-6) * 0.10
                return output + torch.randn_like(output) * scale.to(output.device)

            handle = module.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    noisy = self.model(input_ids=inputs.unsqueeze(0).to(self.device))
                impact = clean_logit - float(noisy.logits[0, -1, target_id].item())
            finally:
                handle.remove()
            scored.append((layer_idx, impact))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[0][0] if scored else self._valid_layer(self.target_layer)

    def _edit_layer(self, prompt: str, target: str) -> int:
        if self.auto_locate:
            return self.locate_layer(prompt, target)
        return self._valid_layer(self.target_layer)

    def _capture_down_proj(
        self, prompt: str, layer_idx: int, subject: str
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        token_ids = self._prompt_ids(prompt)
        position = self._subject_position(token_ids, subject)
        captured: dict[str, torch.Tensor] = {}

        def hook(_module, inputs, output):
            captured["key"] = inputs[0].detach().clone()
            captured["value"] = output.detach().clone()

        handle = self.model.model.layers[layer_idx].mlp.down_proj.register_forward_hook(hook)
        try:
            with torch.no_grad():
                self.model(input_ids=token_ids.unsqueeze(0).to(self.device))
        finally:
            handle.remove()
        if "key" not in captured:
            return None
        return (
            captured["key"][0, position, :].float(),
            captured["value"][0, position, :].float(),
        )

    def _target_direction(
        self, prompt: str, target: str, old_target: str, mode: str
    ) -> tuple[torch.Tensor | None, list[int]]:
        target_ids = self._target_ids(prompt, target, mode)
        if not target_ids:
            return None, []
        direction = self._ids_direction(target_ids)
        if old_target:
            old_ids = self._target_ids(prompt, old_target, mode)
            if old_ids:
                direction = direction - 0.5 * self._ids_direction(old_ids)
        return direction / (direction.norm() + 1e-8), target_ids

    def _ids_direction(self, token_ids: list[int]) -> torch.Tensor:
        rows = []
        lm_weight = self.model.lm_head.weight.data
        for pos, token_id in enumerate(token_ids):
            rows.append(lm_weight[int(token_id), :].detach().float() / (2 ** pos))
        return torch.stack(rows).sum(dim=0).to(self.device)

    def _top_direction_rows(self, direction: torch.Tensor) -> list[int]:
        count = max(1, min(self.top_rows, direction.numel()))
        return [int(row) for row in torch.topk(direction.abs(), count).indices.tolist()]

    def _snapshot_ffn_non_target(self, layer_idx: int, edited_rows: list[int]) -> None:
        sampled = snapshot_rows(
            self.ffn_weight(layer_idx), set(edited_rows), self.nt_sample_size)
        self._ffn_nt_snapshot = {
            (layer_idx, row_id): before for row_id, before in sampled.items()
        }

    def _target_ids(self, prompt: str, target: str, mode: str) -> list[int]:
        if mode in {"contextual", "both"}:
            ids = contextual_target_ids(self, prompt, target)
            if ids:
                return ids
        return self._encode(target)

    def _subject_position(self, token_ids: torch.Tensor, subject: str) -> int:
        subject_ids = self._encode(subject)
        if not subject_ids:
            return int(token_ids.numel() - 1)
        subject_tensor = torch.tensor(subject_ids, device=token_ids.device)
        for idx in range(0, token_ids.numel() - len(subject_ids) + 1):
            if torch.equal(token_ids[idx:idx + len(subject_ids)], subject_tensor):
                return idx + len(subject_ids) - 1
        return int(token_ids.numel() - 1)

    def _candidate_layers(self) -> list[int]:
        if self.candidate_layers:
            return list(self.candidate_layers)
        max_layer = min(12, len(self.model.model.layers))
        return list(range(max_layer))

    def _valid_layer(self, layer_idx: int) -> int:
        max_idx = len(self.model.model.layers) - 1
        return max(0, min(int(layer_idx), max_idx))

    def _prompt_ids(self, prompt: str) -> torch.Tensor:
        return self.tokenizer(prompt, return_tensors="pt").input_ids[0]

    def _encode(self, text: str) -> list[int]:
        return [int(tid) for tid in self.tokenizer.encode(text, add_special_tokens=False)]

    @staticmethod
    def _empty_backup(layer_idx: int) -> dict[str, Any]:
        return {
            "lm_backup": {},
            "emb_backup": {},
            "ffn_backup": {},
            "ffn_layer": layer_idx,
            "ffn_rows": [],
        }

    @property
    def edit_count(self) -> int:
        return self._edit_count
