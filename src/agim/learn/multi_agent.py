"""v3.0: Multi-Agent Memory Ecosystem — teacher, researcher, verifier, curator."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4

from ..core.system import AGIMSystem
from ..core.state import MemoryCandidate, MemoryTier


class AgentRole(StrEnum):
    TEACHER = "teacher"
    RESEARCHER = "researcher"
    VERIFIER = "verifier"
    CURATOR = "curator"


@dataclass
class AgentMessage:
    role: AgentRole
    candidate: MemoryCandidate
    action: str
    message_id: str = field(default_factory=lambda: uuid4().hex[:12])


class MemoryBus:
    """Shared memory bus — agents publish/consume MemoryCandidates."""

    def __init__(self):
        self.queue: list[AgentMessage] = []
        self.subscribers: dict[AgentRole, list[AgentRole]] = {}

    def publish(self, msg: AgentMessage):
        self.queue.append(msg)

    def subscribe(self, subscriber: AgentRole, publishers: list[AgentRole]):
        self.subscribers[subscriber] = publishers

    def consume(self, consumer: AgentRole) -> list[AgentMessage]:
        allowed = set(self.subscribers.get(consumer, []))
        result = [m for m in self.queue if m.role in allowed]
        self.queue = [m for m in self.queue if m.role not in allowed]
        return result


class MemoryAgent:
    def __init__(self, role: AgentRole, agim: AGIMSystem, bus: MemoryBus):
        self.role = role
        self.agim = agim
        self.bus = bus
        self.processed: int = 0

    def step(self):
        msgs = self.bus.consume(self.role)
        for msg in msgs:
            self._process(msg)

    def _process(self, msg: AgentMessage):
        raise NotImplementedError


class TeacherAgent(MemoryAgent):
    def __init__(self, agim: AGIMSystem, bus: MemoryBus):
        super().__init__(AgentRole.TEACHER, agim, bus)

    def teach(self, question: str, answer: str) -> bool:
        c = self.agim.propose_memory(question=question, answer=answer, kind="fact_teach")
        report = self.agim.compile(c)
        if report.passed:
            self.agim.commit(report)
            self.processed += 1
            self.bus.publish(AgentMessage(self.role, c, "taught"))
            return True
        return False

    def _process(self, msg: AgentMessage):
        if msg.action == "verified":
            self.teach(msg.candidate.question, msg.candidate.answer)


class VerifierAgent(MemoryAgent):
    def __init__(self, agim: AGIMSystem, bus: MemoryBus):
        super().__init__(AgentRole.VERIFIER, agim, bus)

    def _process(self, msg: AgentMessage):
        report = self.agim.compile(msg.candidate)
        if report.passed:
            self.bus.publish(AgentMessage(self.role, msg.candidate, "verified"))
        else:
            self.bus.publish(AgentMessage(self.role, msg.candidate, "rejected"))
        self.processed += 1


class CuratorAgent(MemoryAgent):
    def __init__(self, agim: AGIMSystem, bus: MemoryBus):
        super().__init__(AgentRole.CURATOR, agim, bus)
        self.max_facts = 1000

    def _process(self, msg: AgentMessage):
        s = self.agim.stats()
        if s.total_facts > self.max_facts:
            self._prune_low_confidence()

    def _prune_low_confidence(self):
        low_conf = [(q, e) for q, e in self.agim.retrieval._data.items()
                    if e.get("confidence", 0) < 0.3]
        for q, _ in low_conf[:10]:
            self.agim.retrieval.remove(q)
            self.processed += 1
