"""Multi-user support — namespace-isolated memory stores."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .system import AGIMSystem


class MultiUserAGIM:
    """Multi-user AGIM system with namespace-isolated memory stores.

    Each user gets their own workdir with WAL recipes, retrieval memory,
    event log, and governance state. Users cannot access each other's data.
    """

    def __init__(self, base_dir: str | Path = ".agim_users"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._instances: dict[str, AGIMSystem] = {}
        self._user_index_path = self.base_dir / "users.json"

    def _load_index(self) -> dict[str, Any]:
        import json
        if self._user_index_path.exists():
            return json.loads(self._user_index_path.read_text())
        return {}

    def _save_index(self, data: dict):
        import json
        self._user_index_path.write_text(json.dumps(data, indent=2))

    def get_user(self, user_id: str, create: bool = True) -> AGIMSystem | None:
        if user_id in self._instances:
            return self._instances[user_id]
        user_dir = self.base_dir / user_id
        if not create and not user_dir.exists():
            return None
        agim = AGIMSystem(workdir=user_dir)
        self._instances[user_id] = agim
        idx = self._load_index()
        if user_id not in idx:
            idx[user_id] = {"created": agim.log.tail(1)[0]["timestamp"] if agim.log.tail(1) else ""}
            self._save_index(idx)
        return agim

    def list_users(self) -> list[str]:
        return [d.name for d in self.base_dir.iterdir()
                if d.is_dir() and (d / "logs").exists()]

    def delete_user(self, user_id: str) -> bool:
        import shutil
        user_dir = self.base_dir / user_id
        if user_dir.exists():
            shutil.rmtree(user_dir)
            self._instances.pop(user_id, None)
            idx = self._load_index()
            idx.pop(user_id, None)
            self._save_index(idx)
            return True
        return False

    def merge_users(self, from_user: str, to_user: str) -> int:
        """Merge one user's memories into another. Returns count merged."""
        src = self.get_user(from_user, create=False)
        dst = self.get_user(to_user, create=True)
        if src is None:
            return 0
        count = 0
        for question, entry in src.retrieval._data.items():
            dst.retrieval.upsert(question, entry.get("answer", ""),
                                source=entry.get("source", "merged"),
                                confidence=entry.get("confidence", 0.8))
            count += 1
        for recipe_id, recipe in src.wal._index.items():
            if recipe_id not in dst.wal._index:
                dst.wal.write_recipe_from_dict(recipe)
                count += 1
        return count
