"""Memory Overlay — serves model responses with accumulated AGIM knowledge."""
from __future__ import annotations

from typing import Any

from ..core.state import AIGIResponse
from ..core.system import AGIMSystem
from .backends import HuggingFaceTextBackend, StaticTextModelBackend, TextModelBackend


class MemoryOverlay:
    """Serves requests from AGIM memory first, falls back to real model."""

    def __init__(self, agim: AGIMSystem,
                 model_name: str | None = None,
                 device: str = "cuda:3",
                 model_backend: TextModelBackend | None = None):
        self.agim = agim
        if model_backend is not None:
            self.model = model_backend
        elif model_name is not None:
            self.model = HuggingFaceTextBackend(model_name, device=device)
        else:
            self.model = StaticTextModelBackend()
        self._model_name = model_name
        self._model_loaded = False

    def generate(self, question: str, max_tokens: int = 256) -> AIGIResponse:
        resp = self.agim.ask(question)
        if resp.source != "model_fallback":
            return resp
        if self._model_name:
            if not self._model_loaded:
                self.model._ensure_loaded()
                self._model_loaded = True
            answer = self.model.generate(question, max_tokens)
            return AIGIResponse(question=question, answer=answer,
                              source="model_fallback", confidence=0.5)
        return AIGIResponse(question=question, answer="[no model available]",
                          source="none", confidence=0.0)

    def teach(self, question: str, answer: str, kind: str = "fact_teach") -> bool:
        c = self.agim.propose_memory(question=question, answer=answer, kind=kind)
        report = self.agim.compile(c)
        if not report.passed:
            return False
        return self.agim.commit(report)

    def ask(self, question: str) -> AIGIResponse:
        return self.generate(question)

    @property
    def model_available(self) -> bool:
        return not isinstance(self.model, StaticTextModelBackend)
