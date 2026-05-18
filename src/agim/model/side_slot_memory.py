"""Side-slot patch memory for routed runtime overlays."""
from __future__ import annotations

from dataclasses import dataclass

from .patch_artifact import PatchArtifact
from .sparse_overlay import RuntimeSparseOverlay


@dataclass
class PatchSlot:
    slot_id: str
    artifact: PatchArtifact
    enabled: bool = True

    def matches(self, subject: str = "", relation_id: str = "") -> bool:
        if not self.enabled:
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
                  slot_id: str | None = None) -> PatchSlot:
        key = slot_id or artifact.patch_id
        slot = PatchSlot(slot_id=key, artifact=artifact)
        self._slots[key] = slot
        return slot

    def disable(self, slot_id: str) -> None:
        self._slots[slot_id].enabled = False

    def enable(self, slot_id: str) -> None:
        self._slots[slot_id].enabled = True

    def select(self, subject: str = "", relation_id: str = "",
               limit: int | None = None) -> list[PatchSlot]:
        slots = [
            slot for slot in self._slots.values()
            if slot.matches(subject=subject, relation_id=relation_id)
        ]
        if limit is not None:
            return slots[:limit]
        return slots

    def overlay_for(self, model, subject: str = "", relation_id: str = "",
                    limit: int | None = None) -> RuntimeSparseOverlay:
        overlay = RuntimeSparseOverlay(model)
        for slot in self.select(subject=subject, relation_id=relation_id, limit=limit):
            overlay.add_patch_artifact(slot.artifact)
        return overlay

    def summary(self) -> dict[str, int]:
        enabled = sum(1 for slot in self._slots.values() if slot.enabled)
        return {"slots": len(self._slots), "enabled": enabled}
