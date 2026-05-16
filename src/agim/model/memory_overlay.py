"""Memory overlay — serves model responses with accumulated knowledge."""
from typing import Any

from ..core.state import AIGIResponse
from ..core.system import AGIMSystem
from .backends import TextModelBackend


class MemoryOverlay:
    """Serves requests from memory first, falls back to model."""

    def __init__(self, agim: AGIMSystem, model_backend: TextModelBackend | None = None):
        self.agim = agim
        self.model = model_backend

    def generate(self, question: str, max_tokens: int = 256) -> AIGIResponse:
        resp = self.agim.ask(question)
        if resp.source != "model_fallback":
            return resp
        if self.model is not None:
            answer = self.model.generate(question, max_tokens)
            return AIGIResponse(question=question, answer=answer,
                              source="model_fallback", confidence=0.5)
        return AIGIResponse(question=question, answer="[no answer available]",
                          source="none", confidence=0.0)

    def teach_and_apply(self, question: str, answer: str, kind: str = "fact_teach",
                        target_layer: str | None = None) -> bool:
        c = self.agim.propose_memory(question=question, answer=answer, kind=kind)
        report = self.agim.compile(c)
        if not report.passed:
            return False
        return self.agim.commit(report)
