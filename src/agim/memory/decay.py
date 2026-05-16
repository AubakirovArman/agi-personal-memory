"""Memory decay — facts fade over time unless reinforced."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


@dataclass
class DecayConfig:
    half_life_days: float = 180.0
    min_confidence: float = 0.1
    reinforcement_boost: float = 0.2
    max_confidence: float = 1.0


class MemoryDecay:
    """Ebbinghaus-inspired memory decay with spaced repetition reinforcement."""

    def __init__(self, config: DecayConfig | None = None):
        self.config = config or DecayConfig()
        self._access_counts: dict[str, int] = {}
        self._last_access: dict[str, float] = {}
        self._last_reinforced: dict[str, float] = {}

    def compute_confidence(self, memory_id: str, initial_confidence: float,
                           created_at: str | None = None) -> float:
        now = time.time()
        if created_at:
            try:
                from datetime import datetime
                created_ts = datetime.fromisoformat(created_at).timestamp()
                age_days = (now - created_ts) / 86400
            except (ValueError, TypeError):
                age_days = 0
        else:
            age_days = 0

        accesses = self._access_counts.get(memory_id, 0)
        last_acc = self._last_access.get(memory_id, now)
        days_since_access = (now - last_acc) / 86400

        decay = math.exp(-math.log(2) * age_days / self.config.half_life_days)

        spacing_effect = 1.0
        if accesses > 1:
            spacing_effect = 1.0 + 0.1 * math.log(accesses)

        recency_boost = math.exp(-days_since_access / 30)

        confidence = initial_confidence * decay * spacing_effect * recency_boost
        return max(self.config.min_confidence, min(self.config.max_confidence, confidence))

    def record_access(self, memory_id: str):
        now = time.time()
        self._access_counts[memory_id] = self._access_counts.get(memory_id, 0) + 1
        self._last_access[memory_id] = now

    def reinforce(self, memory_id: str) -> float:
        now = time.time()
        self._access_counts[memory_id] = self._access_counts.get(memory_id, 0) + 1
        self._last_reinforced[memory_id] = now
        self._last_access[memory_id] = now
        return self.config.reinforcement_boost

    def should_forget(self, confidence: float) -> bool:
        return confidence < self.config.min_confidence
