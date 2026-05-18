"""Serializable sparse patch artifacts for Path B weight editing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class RowPatch:
    layer: str
    row_id: int
    before: list[float]
    after: list[float]

    @classmethod
    def from_tensors(cls, layer: str, row_id: int,
                     before: torch.Tensor, after: torch.Tensor) -> "RowPatch":
        return cls(
            layer=layer,
            row_id=int(row_id),
            before=_tensor_list(before),
            after=_tensor_list(after),
        )

    @property
    def delta_norm(self) -> float:
        before = torch.tensor(self.before)
        after = torch.tensor(self.after)
        return float((after - before).norm().item())


@dataclass
class PatchArtifact:
    patch_id: str
    base_model_digest: str
    method_profile_id: str
    subject: str
    relation_id: str
    target_new: str
    target_true: str | None = None
    rows: list[RowPatch] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_schema_version": "path_b_patch.v1",
            "patch_id": self.patch_id,
            "base_model_digest": self.base_model_digest,
            "method_profile_id": self.method_profile_id,
            "subject": self.subject,
            "relation_id": self.relation_id,
            "target_new": self.target_new,
            "target_true": self.target_true,
            "row_counts": self.row_counts(),
            "norms": self.norm_summary(),
            "rows": [row.__dict__ for row in self.rows],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PatchArtifact":
        rows = [RowPatch(**row) for row in payload.get("rows", [])]
        return cls(
            patch_id=payload["patch_id"],
            base_model_digest=payload["base_model_digest"],
            method_profile_id=payload["method_profile_id"],
            subject=payload["subject"],
            relation_id=payload.get("relation_id", ""),
            target_new=payload["target_new"],
            target_true=payload.get("target_true"),
            rows=rows,
            metadata=payload.get("metadata", {}),
        )

    def touched_rows(self, layer: str | None = None) -> set[tuple[str, int]]:
        rows = {(row.layer, row.row_id) for row in self.rows}
        if layer is None:
            return rows
        return {(name, row_id) for name, row_id in rows if name == layer}

    def row_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row.layer] = counts.get(row.layer, 0) + 1
        return dict(sorted(counts.items()))

    def norm_summary(self) -> dict[str, float]:
        if not self.rows:
            return {"max_delta_norm": 0.0, "mean_delta_norm": 0.0}
        norms = [row.delta_norm for row in self.rows]
        return {
            "max_delta_norm": round(max(norms), 6),
            "mean_delta_norm": round(sum(norms) / len(norms), 6),
        }


def conflict_summary(left: PatchArtifact, right: PatchArtifact) -> dict[str, Any]:
    overlap = sorted(left.touched_rows() & right.touched_rows())
    return {
        "has_conflict": bool(overlap),
        "overlapping_rows": [
            {"layer": layer, "row_id": row_id} for layer, row_id in overlap
        ],
        "same_relation": bool(left.relation_id and left.relation_id == right.relation_id),
        "same_subject": bool(left.subject and left.subject == right.subject),
    }


def _tensor_list(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().float().cpu().flatten().tolist()]
