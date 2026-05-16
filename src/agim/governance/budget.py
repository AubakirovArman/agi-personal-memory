"""Memory budget — limit changes per time period to prevent runaway accumulation."""
import json
import time
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class MemoryBudget:
    max_total_facts: int = 100_000
    max_daily_commits: int = 500
    max_hourly_commits: int = 50
    _daily_count: int = 0
    _hourly_count: int = 0
    _day_start: float = field(default_factory=time.time)
    _hour_start: float = field(default_factory=time.time)

    def check(self, current_total: int) -> tuple[bool, str]:
        now = time.time()
        if now - self._day_start > 86400:
            self._daily_count = 0
            self._day_start = now
        if now - self._hour_start > 3600:
            self._hourly_count = 0
            self._hour_start = now
        if current_total >= self.max_total_facts:
            return False, f"Total fact limit reached ({self.max_total_facts})"
        if self._daily_count >= self.max_daily_commits:
            return False, f"Daily commit limit reached ({self.max_daily_commits})"
        if self._hourly_count >= self.max_hourly_commits:
            return False, f"Hourly commit limit reached ({self.max_hourly_commits})"
        return True, ""

    def record_commit(self):
        self._daily_count += 1
        self._hourly_count += 1

    def save(self, path: Path):
        path.write_text(json.dumps({
            "max_total_facts": self.max_total_facts,
            "max_daily_commits": self.max_daily_commits,
            "max_hourly_commits": self.max_hourly_commits,
            "daily_count": self._daily_count,
            "hourly_count": self._hourly_count,
            "day_start": self._day_start,
            "hour_start": self._hour_start,
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> "MemoryBudget":
        if path.exists():
            d = json.loads(path.read_text())
            return cls(**{k: v for k, v in d.items() if not k.endswith("_count") and not k.endswith("_start")})
        return cls()
