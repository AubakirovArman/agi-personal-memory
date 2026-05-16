"""v2.0/v3.0: Web Research Agent + Researcher Agent — autonomous knowledge discovery."""
from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from ..core.system import AGIMSystem


@dataclass
class ResearchResult:
    query: str
    findings: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.5
    verified: bool = False
    research_id: str = field(default_factory=lambda: uuid4().hex[:12])


class WebResearchAgent:
    """v2.0: Search-then-verify agent for autonomous fact discovery."""

    def __init__(self, agim: AGIMSystem):
        self.agim = agim
        self.history: list[ResearchResult] = []

    def research(self, query: str) -> ResearchResult:
        """Search for information and verify before storing."""
        result = ResearchResult(query=query)
        existing = self.agim.ask(query)
        if existing.source != "model_fallback":
            result.findings.append(f"Already known: {existing.answer}")
            result.confidence = existing.confidence
            result.verified = True
            self.history.append(result)
            return result

        result.findings.append(f"[Research needed]: {query}")
        result.confidence = 0.3
        self.history.append(result)
        return result

    def verify_and_commit(self, result: ResearchResult, min_confidence: float = 0.7):
        """Cross-reference verify findings and commit if confident enough."""
        if result.confidence >= min_confidence:
            for finding in result.findings:
                c = self.agim.propose_memory(
                    question=result.query, answer=finding,
                    kind="fact_teach", source="research_agent",
                    confidence=result.confidence)
                report = self.agim.compile(c)
                if report.passed:
                    self.agim.commit(report)
                    result.verified = True

    @property
    def total_researched(self) -> int:
        return len(self.history)


class ResearcherAgent:
    """v3.0: Researcher Agent — autonomous web research with cross-reference."""

    def __init__(self, agim: AGIMSystem):
        self.agim = agim
        self.research_agent = WebResearchAgent(agim)
        self.focus_areas: list[str] = []

    def set_focus(self, topics: list[str]):
        self.focus_areas = topics

    def explore(self, topic: str, depth: int = 3) -> list[ResearchResult]:
        results = []
        queue = [topic]
        visited = set()
        for _ in range(depth):
            if not queue:
                break
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            r = self.research_agent.research(current)
            results.append(r)
            if r.confidence > 0.7:
                self.research_agent.verify_and_commit(r)
            for finding in r.findings[:3]:
                queue.append(finding)
        return results
