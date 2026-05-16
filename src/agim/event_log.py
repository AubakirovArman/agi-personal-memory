"""JSONL event log for full audit trail."""
import json
from pathlib import Path
from typing import Any


class EventLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, status: str, data: dict[str, Any] | None = None) -> None:
        import datetime
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event": event,
            "status": status,
            "data": data or {},
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def tail(self, n: int = 50) -> list[dict[str, Any]]:
        entries = self.read_all()
        return entries[-n:]
