"""v10.0: Recursive Self-Improvement Loop + Safety Governor."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import StrEnum


class SafetyLevel(StrEnum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SafetyMetrics:
    improvement_rate: float = 0.0
    instability_score: float = 0.0
    gate_failure_rate: float = 0.0
    adversarial_success_rate: float = 0.0
    level: SafetyLevel = SafetyLevel.NORMAL
    emergency_triggered: bool = False


class SafetyGovernor:
    """v10.0: Safety Governor — monitors self-improvement, can apply brakes."""

    MAX_IMPROVEMENT_RATE = 0.2
    MAX_INSTABILITY = 0.1
    BRAKE_DURATION = 3600

    def __init__(self):
        self.metrics = SafetyMetrics()
        self._metric_history: list[SafetyMetrics] = []
        self._brake_active = False
        self._brake_start: float = 0
        self._interventions: int = 0

    def observe(self, improvement_rate: float, gate_failures: int,
                total_gates: int, adversarial_hits: int, total_attacks: int):
        self.metrics.improvement_rate = improvement_rate
        self.metrics.gate_failure_rate = gate_failures / max(total_gates, 1)
        self.metrics.adversarial_success_rate = adversarial_hits / max(total_attacks, 1)
        self.metrics.instability_score = (
            abs(improvement_rate) * 0.3 +
            self.metrics.gate_failure_rate * 0.4 +
            self.metrics.adversarial_success_rate * 0.3
        )
        self._update_level()
        self._metric_history.append(SafetyMetrics(
            improvement_rate=improvement_rate,
            instability_score=self.metrics.instability_score,
            gate_failure_rate=self.metrics.gate_failure_rate,
            level=self.metrics.level,
        ))

    def _update_level(self):
        s = self.metrics
        if s.instability_score > 0.5:
            s.level = SafetyLevel.EMERGENCY
            s.emergency_triggered = True
        elif s.instability_score > 0.3:
            s.level = SafetyLevel.CRITICAL
        elif s.instability_score > 0.2:
            s.level = SafetyLevel.HIGH
        elif s.instability_score > 0.1:
            s.level = SafetyLevel.ELEVATED
        else:
            s.level = SafetyLevel.NORMAL

    def should_brake(self) -> bool:
        if self._brake_active:
            if time.time() - self._brake_start > self.BRAKE_DURATION:
                self._brake_active = False
                return False
            return True
        if self.metrics.instability_score > self.MAX_INSTABILITY:
            self._brake_active = True
            self._brake_start = time.time()
            self._interventions += 1
            return True
        if self.metrics.improvement_rate > self.MAX_IMPROVEMENT_RATE:
            self._brake_active = True
            self._brake_start = time.time()
            self._interventions += 1
            return True
        return False

    def brake_action(self) -> dict:
        """Return recommended safety actions when brake is active."""
        if self.metrics.level == SafetyLevel.EMERGENCY:
            return {"action": "emergency_stop", "duration_hours": 24}
        elif self.metrics.level == SafetyLevel.CRITICAL:
            return {"action": "reduce_learning_rate", "factor": 0.1}
        elif self.metrics.level == SafetyLevel.HIGH:
            return {"action": "increase_verification", "additional_gates": 3}
        return {"action": "monitor", "duration_hours": 1}

    @property
    def interventions(self) -> int:
        return self._interventions


class RecursiveImprovementLoop:
    """v10.0: Recursive self-improvement — system improves its ability to improve."""

    def __init__(self, governor: SafetyGovernor | None = None):
        self.governor = governor or SafetyGovernor()
        self._iteration: int = 0
        self._improvement_history: list[float] = []
        self._meta_metrics: dict[str, list[float]] = {"accuracy": [], "latency": [],
                                                        "safety": [], "efficiency": []}

    def step(self, metrics: dict[str, float]) -> dict:
        """One iteration of the improvement loop."""
        self._iteration += 1
        for k, v in metrics.items():
            if k in self._meta_metrics:
                self._meta_metrics[k].append(v)

        if self.governor.should_brake():
            return {"action": "braked", **self.governor.brake_action()}

        improvement = self._compute_improvement()
        self._improvement_history.append(improvement)

        self.governor.observe(
            improvement_rate=improvement,
            gate_failures=int(metrics.get("gate_failures", 0)),
            total_gates=max(1, int(metrics.get("total_gates", 10))),
            adversarial_hits=int(metrics.get("adversarial_hits", 0)),
            total_attacks=max(1, int(metrics.get("total_attacks", 6))),
        )
        return {"action": "continue", "iteration": self._iteration,
                "improvement": improvement, "level": self.governor.metrics.level.value}

    def _compute_improvement(self) -> float:
        if len(self._meta_metrics["accuracy"]) < 2:
            return 0.0
        recent = self._meta_metrics["accuracy"][-5:]
        if len(recent) < 2:
            return 0.0
        return (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-8)

    @property
    def total_improvement(self) -> float:
        if not self._improvement_history:
            return 0.0
        return sum(self._improvement_history)
