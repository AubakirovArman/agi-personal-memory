"""WAL recipe memory store for stable, permanent facts."""
import json
from pathlib import Path
from uuid import uuid4


class WALMemory:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._index_path = self.path / "index.json"
        self._index: dict[str, dict] = {}
        self._load_index()

    def _load_index(self):
        if self._index_path.exists():
            self._index = json.loads(self._index_path.read_text())

    def _save_index(self):
        self._index_path.write_text(json.dumps(self._index, indent=2, ensure_ascii=False))

    def write_recipe(self, candidate) -> str:
        artifact_id = uuid4().hex[:12]
        recipe = {
            "artifact_id": artifact_id,
            "question": candidate.question,
            "answer": candidate.answer,
            "kind": candidate.kind,
            "source": candidate.source,
            "confidence": candidate.confidence,
            "candidate_id": candidate.candidate_id,
            "created_at": candidate.created_at,
        }
        recipe_path = self.path / f"{artifact_id}.json"
        recipe_path.write_text(json.dumps(recipe, indent=2, ensure_ascii=False))
        self._index[artifact_id] = recipe
        self._save_index()
        return artifact_id

    def preview_artifact_id(self, candidate) -> str:
        return uuid4().hex[:12]

    def remove_recipe(self, artifact_id: str) -> bool:
        recipe_path = self.path / f"{artifact_id}.json"
        if recipe_path.exists():
            recipe_path.unlink()
        if artifact_id in self._index:
            del self._index[artifact_id]
            self._save_index()
            return True
        return False

    def get_recipe(self, artifact_id: str) -> dict | None:
        return self._index.get(artifact_id)

    def __len__(self) -> int:
        return len(self._index)
