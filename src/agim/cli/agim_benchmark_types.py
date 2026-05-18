from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Fact:
    subject: str
    relation: str
    obj: str
    prompt: str = ""
    answer: str = ""
    category: str = "general"
    difficulty: str = "medium"

    def to_qa(self) -> tuple[str, str]:
        q = self.prompt or f"What is the {self.relation} of {self.subject}?"
        a = self.answer or self.obj
        return q, a


@dataclass
class BenchmarkResult:
    model_name: str
    n_facts: int
    n_tested: int
    baseline_accuracy: float
    post_accuracy: float
    delta: float
    train_time_s: float
    train_rate: float
    memory_hit_rate: float
    model_size_mb: float
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class Leaderboard:
    def __init__(self, path: str = "agim_leaderboard.json"):
        self.path = Path(path)
        self.entries: list[dict] = []
        self._load()

    def _load(self):
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.entries = data.get("leaderboard", [])

    def add(self, result: BenchmarkResult):
        entry = result.to_dict()
        entry["rank"] = len(self.entries) + 1
        self.entries.append(entry)
        self.entries.sort(key=lambda x: x.get("delta", 0), reverse=True)
        for i, e in enumerate(self.entries):
            e["rank"] = i + 1
        self._save()

    def _save(self):
        self.path.write_text(json.dumps({
            "leaderboard": self.entries,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=2, ensure_ascii=False))

    def print(self):
        print("\n" + "=" * 80)
        print("AGIM LEADERBOARD")
        print("=" * 80)
        print(f"{'Rank':<5} {'Model':<30} {'N':<6} {'Base':<8} {'Post':<8} {'Delta':<8} {'Rate':<10}")
        print("-" * 80)
        for e in self.entries[:10]:
            print(f"{e['rank']:<5} {e['model_name']:<30} {e['n_facts']:<6} "
                  f"{e['baseline_accuracy']:.3f}   {e['post_accuracy']:.3f}   "
                  f"{e['delta']:+.3f}   {e['train_rate']:.0f}/s")


# ── Dataset loader ───────────────────────────────────────────────
