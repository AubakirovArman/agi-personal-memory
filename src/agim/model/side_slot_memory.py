"""Side-slot patch memory for routed runtime overlays."""
from __future__ import annotations

from dataclasses import dataclass

from .patch_artifact import PatchArtifact
from .sparse_overlay import RuntimeSparseOverlay


@dataclass
class PatchSlot:
    slot_id: str
    artifact: PatchArtifact
    relation_slot_id: str
    enabled: bool = True

    def matches(self, subject: str = "", relation_id: str = "",
                relation_slot_id: str = "") -> bool:
        if not self.enabled:
            return False
        if relation_slot_id and self.relation_slot_id != relation_slot_id:
            return False
        if subject and self.artifact.subject != subject:
            return False
        if relation_id and self.artifact.relation_id != relation_id:
            return False
        return True


class SideSlotMemory:
    """Route patch artifacts into a runtime overlay without mutating weights."""

    def __init__(self):
        self._slots: dict[str, PatchSlot] = {}

    def add_patch(self, artifact: PatchArtifact,
                  relation_slot_id: str | None = None,
                  slot_id: str | None = None) -> PatchSlot:
        key = slot_id or artifact.patch_id
        relation_slot = relation_slot_id or _relation_slot_id(artifact)
        slot = PatchSlot(
            slot_id=key, artifact=artifact, relation_slot_id=relation_slot)
        self._slots[key] = slot
        return slot

    def disable(self, slot_id: str) -> None:
        self._slots[slot_id].enabled = False

    def enable(self, slot_id: str) -> None:
        self._slots[slot_id].enabled = True

    def disable_relation_slot(self, relation_slot_id: str) -> None:
        for slot in self._slots.values():
            if slot.relation_slot_id == relation_slot_id:
                slot.enabled = False

    def enable_relation_slot(self, relation_slot_id: str) -> None:
        for slot in self._slots.values():
            if slot.relation_slot_id == relation_slot_id:
                slot.enabled = True

    def select(self, subject: str = "", relation_id: str = "",
               relation_slot_id: str = "",
               limit: int | None = None) -> list[PatchSlot]:
        slots = [
            slot for slot in self._slots.values()
            if slot.matches(
                subject=subject,
                relation_id=relation_id,
                relation_slot_id=relation_slot_id,
            )
        ]
        if limit is not None:
            return slots[:limit]
        return slots

    def overlay_for(self, model, subject: str = "", relation_id: str = "",
                    relation_slot_id: str = "",
                    limit: int | None = None) -> RuntimeSparseOverlay:
        overlay = RuntimeSparseOverlay(model)
        for slot in self.select(
            subject=subject,
            relation_id=relation_id,
            relation_slot_id=relation_slot_id,
            limit=limit,
        ):
            overlay.add_patch_artifact(slot.artifact)
        return overlay

    def summary(self) -> dict[str, int]:
        enabled = sum(1 for slot in self._slots.values() if slot.enabled)
        return {"slots": len(self._slots), "enabled": enabled}

    def relation_slot_summary(self) -> dict[str, dict[str, int]]:
        summary: dict[str, dict[str, int]] = {}
        for slot in self._slots.values():
            row = summary.setdefault(
                slot.relation_slot_id, {"slots": 0, "enabled": 0})
            row["slots"] += 1
            row["enabled"] += int(slot.enabled)
        return dict(sorted(summary.items()))


def _relation_slot_id(artifact: PatchArtifact) -> str:
    metadata_slot = artifact.metadata.get("relation_slot_id")
    if metadata_slot:
        return str(metadata_slot)
    return artifact.relation_id or "global"
