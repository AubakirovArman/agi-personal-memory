"""Batch consolidation backend for WAL patch artifacts."""
from __future__ import annotations

from typing import Any

import torch

from .patch_artifact import PatchArtifact, RowPatch


class WALMemitBatchEditor:
    """Consolidate multiple verified row patches into one batch artifact.

    This is the product-facing MEMIT foundation for AGIM: edits can live in
    side slots while they burn in, then be compacted into a single auditable
    artifact that can be applied and rolled back as one unit.
    """

    def __init__(
        self,
        base_model_digest: str = "runtime",
        method_profile_id: str = "wal_memit_batch",
    ):
        self.base_model_digest = base_model_digest
        self.method_profile_id = method_profile_id
        self._patches: list[PatchArtifact] = []

    def add_patch(self, artifact: PatchArtifact) -> None:
        self._patches.append(artifact)

    def extend(self, artifacts: list[PatchArtifact]) -> None:
        for artifact in artifacts:
            self.add_patch(artifact)

    def clear(self) -> None:
        self._patches.clear()

    def summary(self) -> dict[str, Any]:
        rows = sum(len(patch.rows) for patch in self._patches)
        return {
            "patches": len(self._patches),
            "rows": rows,
            "relations": sorted({patch.relation_id for patch in self._patches}),
        }

    def consolidate(self, patch_id: str = "wal-memit-batch") -> PatchArtifact:
        grouped = self._group_rows()
        rows = [
            self._merge_group(layer, row_id, group)
            for (layer, row_id), group in sorted(grouped.items())
        ]
        return PatchArtifact(
            patch_id=patch_id,
            base_model_digest=self._base_digest(),
            method_profile_id=self.method_profile_id,
            subject=";".join(sorted({p.subject for p in self._patches if p.subject})),
            relation_id=";".join(sorted({p.relation_id for p in self._patches if p.relation_id})),
            target_new=";".join(sorted({p.target_new for p in self._patches if p.target_new})),
            rows=rows,
            metadata=self._metadata(grouped),
        )

    def apply(self, model, artifact: PatchArtifact) -> dict[str, Any]:
        backup = {"lm_backup": {}, "emb_backup": {}, "ffn_backup": {}}
        for row in artifact.rows:
            weight, key = self._weight_for_row(model, row.layer, row.row_id)
            before = weight[row.row_id, :].detach().clone()
            after = torch.tensor(row.after, device=weight.device, dtype=weight.dtype)
            if before.numel() != after.numel():
                raise ValueError(f"Row shape mismatch for {row.layer}:{row.row_id}")
            self._store_backup(backup, row.layer, key, before)
            weight[row.row_id, :] = after.reshape_as(weight[row.row_id, :])
        return backup

    def rollback(self, model, backup: dict[str, Any]) -> None:
        for row_id, before in backup.get("lm_backup", {}).items():
            weight = model.lm_head.weight.data
            weight[int(row_id), :] = before.to(device=weight.device, dtype=weight.dtype)
        for row_id, before in backup.get("emb_backup", {}).items():
            weight = model.model.embed_tokens.weight.data
            weight[int(row_id), :] = before.to(device=weight.device, dtype=weight.dtype)
        for (layer_idx, row_id), before in backup.get("ffn_backup", {}).items():
            weight = model.model.layers[int(layer_idx)].mlp.down_proj.weight.data
            weight[int(row_id), :] = before.to(device=weight.device, dtype=weight.dtype)

    def _group_rows(self) -> dict[tuple[str, int], list[RowPatch]]:
        grouped: dict[tuple[str, int], list[RowPatch]] = {}
        for patch in self._patches:
            for row in patch.rows:
                grouped.setdefault((row.layer, row.row_id), []).append(row)
        return grouped

    @staticmethod
    def _merge_group(layer: str, row_id: int, rows: list[RowPatch]) -> RowPatch:
        before = torch.tensor(rows[0].before)
        delta = torch.zeros_like(before)
        for row in rows:
            current_before = torch.tensor(row.before)
            current_after = torch.tensor(row.after)
            if current_before.numel() != before.numel():
                raise ValueError(f"Row shape mismatch in {layer}:{row_id}")
            delta = delta + (current_after - current_before).reshape_as(before)
        return RowPatch.from_tensors(layer, row_id, before, before + delta)

    def _metadata(self, grouped) -> dict[str, Any]:
        conflicts = [
            {"layer": layer, "row_id": row_id, "count": len(rows)}
            for (layer, row_id), rows in sorted(grouped.items())
            if len(rows) > 1
        ]
        return {
            "source_patch_ids": [patch.patch_id for patch in self._patches],
            "source_patch_count": len(self._patches),
            "row_conflicts": conflicts,
            "consolidation": "sum_row_deltas",
        }

    def _base_digest(self) -> str:
        digests = {patch.base_model_digest for patch in self._patches}
        if len(digests) == 1:
            return next(iter(digests))
        return self.base_model_digest

    @staticmethod
    def _weight_for_row(model, layer: str, row_id: int):
        if layer == "lm_head":
            return model.lm_head.weight.data, int(row_id)
        if layer == "embed_tokens":
            return model.model.embed_tokens.weight.data, int(row_id)
        if layer.startswith("model.layers.") and layer.endswith(".mlp.down_proj"):
            layer_idx = int(layer.split(".")[2])
            weight = model.model.layers[layer_idx].mlp.down_proj.weight.data
            return weight, (layer_idx, int(row_id))
        raise ValueError(f"Unsupported consolidated layer: {layer}")

    @staticmethod
    def _store_backup(backup: dict[str, Any], layer: str,
                      key: int | tuple[int, int], before: torch.Tensor) -> None:
        if layer == "lm_head":
            backup["lm_backup"][int(key)] = before
        elif layer == "embed_tokens":
            backup["emb_backup"][int(key)] = before
        else:
            backup["ffn_backup"][key] = before
