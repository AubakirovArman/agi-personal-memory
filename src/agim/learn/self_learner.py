"""v2.0: Autonomous Self-Learning — active knowledge extraction from interactions."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..core.system import AGIMSystem
from .experience import Experience, Lesson, LessonExtractor


@dataclass
class Reflection:
    """Meta-cognitive analysis of system performance after a session."""
    text: str
    improvements: list[str] = field(default_factory=list)
    mistakes: list[str] = field(default_factory=list)
    confidence: float = 0.7
    reflection_id: str = field(default_factory=lambda: uuid4().hex[:12])


class ReflectionEngine:
    """Generates self-reflection after sessions. Identifies error patterns."""

    def generate(self, experiences: list[Experience]) -> Reflection:
        corrections = [e for e in experiences if e.rating <= 2]
        successes = [e for e in experiences if e.rating >= 4]
        text_parts = []
        improvements = []
        mistakes = []

        if corrections:
            text_parts.append(f"Had {len(corrections)} corrections.")
            question_words = {}
            for e in corrections:
                w = e.question.split()[0].lower() if e.question.split() else ""
                question_words[w] = question_words.get(w, 0) + 1
            top_pattern = max(question_words, key=question_words.get) if question_words else "unknown"
            improvements.append(f"Improve handling of '{top_pattern}' questions")
            mistakes.append(f"Often wrong on '{top_pattern}' questions")

        if successes:
            text_parts.append(f"Had {len(successes)} successful interactions.")

        text = " ".join(text_parts) if text_parts else "No significant events."
        return Reflection(text=text, improvements=improvements,
                         mistakes=mistakes, confidence=0.7)


class SelfLearner:
    """v2.0: Actively learns from every interaction without user prompting."""

    def __init__(self, agim: AGIMSystem):
        self.agim = agim
        self.extractor = LessonExtractor()
        self.reflection_engine = ReflectionEngine()
        self.experiences: list[Experience] = []
        self.reflections: list[Reflection] = []

    def record(self, question: str, answer: str, user_reaction: str = "",
               rating: int = 3) -> Experience | None:
        exp = Experience(question=question, model_answer=answer,
                        user_feedback=user_reaction, rating=rating)
        self.experiences.append(exp)

        if rating <= 2:
            lesson = self.extractor.extract(exp)
            if lesson:
                candidate = self.agim.propose_memory(
                    question=lesson.candidate.question,
                    answer=lesson.candidate.answer,
                    kind="self_correction",
                    source="self_learner",
                    confidence=0.8,
                )
                report = self.agim.compile(candidate)
                if report.passed:
                    self.agim.commit(report)
                    return exp
        return exp

    def reflect(self) -> Reflection:
        r = self.reflection_engine.generate(self.experiences[-50:])
        self.reflections.append(r)
        for improvement in r.improvements:
            self.agim.propose_memory(
                question=f"self_improvement_{len(self.reflections)}",
                answer=improvement, kind="self_reflection",
                source="reflection_engine", confidence=0.6)
        return r

    @property
    def total_lessons(self) -> int:
        return sum(1 for e in self.experiences if e.rating <= 2)
