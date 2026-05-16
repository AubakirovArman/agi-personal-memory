"""Verified Learning Loop — feedback → extract → verify → commit."""
from ..core.system import AGIMSystem
from ..core.state import CompileReport
from .experience import Experience, Lesson, LessonExtractor


class VerifiedLearningLoop:
    def __init__(self, agim: AGIMSystem):
        self.agim = agim
        self.extractor = LessonExtractor()
        self.feedback_history: list[Experience] = []

    def process_feedback(self, question: str, model_answer: str,
                         user_feedback: str, rating: int) -> CompileReport | None:
        exp = Experience(question=question, model_answer=model_answer,
                        user_feedback=user_feedback, rating=rating)
        self.feedback_history.append(exp)
        lesson = self.extractor.extract(exp)
        if lesson is None:
            return None
        candidate = self.agim.propose_memory(
            question=lesson.candidate.question,
            answer=lesson.candidate.answer,
            kind=lesson.candidate.kind,
            source="verified_learning_loop",
            confidence=lesson.candidate.confidence,
            metadata={"experience_id": exp.experience_id,
                     "lesson_id": lesson.lesson_id},
        )
        report = self.agim.compile(candidate)
        if report.passed:
            self.agim.commit(report)
        return report

    @property
    def total_feedback(self) -> int:
        return len(self.feedback_history)
