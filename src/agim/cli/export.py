"""Export/import memories for backup, sharing, and migration."""
import json
from pathlib import Path
from typing import Any

from ..core.system import AGIMSystem


def export_memories(agim: AGIMSystem, output_path: str | Path) -> int:
    """Export all memories to JSON. Returns count."""
    data = {
        "version": "1.0",
        "exported_at": agim.log.read_all()[-1]["timestamp"] if agim.log.read_all() else "",
        "memories": {
            "retrieval": agim.retrieval._data,
            "refusals": agim.refusals._data,
            "wal_recipes": agim.wal._index,
        },
        "commit_history": [
            {"artifact_id": r.artifact_id, "tier": r.tier.value,
             "question": r.question, "answer": r.answer}
            for r in agim.commit_history
        ],
        "event_log": agim.log.tail(10000),
    }
    Path(output_path).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return len(agim.retrieval) + len(agim.wal)


def import_memories(agim: AGIMSystem, input_path: str | Path) -> int:
    """Import memories from JSON. Returns count of imported."""
    data = json.loads(Path(input_path).read_text())
    count = 0
    memories = data.get("memories", {})
    for question, entry in memories.get("retrieval", {}).items():
        agim.retrieval.upsert(question, entry.get("answer", ""),
                              source=entry.get("source", "import"),
                              confidence=entry.get("confidence", 0.8))
        count += 1
    for aid, recipe in memories.get("wal_recipes", {}).items():
        if aid not in agim.wal._index:
            agim.wal.write_recipe_from_dict(recipe)
            count += 1
    return count
