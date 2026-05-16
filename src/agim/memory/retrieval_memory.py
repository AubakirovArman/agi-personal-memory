"""Key-value retrieval memory store for volatile facts."""
import json
from pathlib import Path
from typing import Any


class RetrievalMemory:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            self._data = json.loads(self.path.read_text())
        else:
            self._data = {}

    def _save(self):
        self.path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False))

    def lookup(self, question: str) -> dict[str, Any] | None:
        return self._data.get(question)

    def upsert(self, question: str, answer: str, source: str = "retrieval",
               memory_id: str | None = None, confidence: float = 1.0) -> str:
        mid = memory_id or question
        self._data[question] = {
            "id": mid, "question": question, "answer": answer,
            "source": source, "confidence": confidence}
        self._save()
        return mid

    def remove(self, question: str) -> bool:
        if question in self._data:
            del self._data[question]
            self._save()
            return True
        return False

    def restore(self, question: str, previous: dict[str, Any] | None) -> None:
        if previous:
            self._data[question] = previous
        else:
            self._data.pop(question, None)
        self._save()

    def all_questions(self) -> list[str]:
        return list(self._data.keys())

    def __len__(self) -> int:
        return len(self._data)
