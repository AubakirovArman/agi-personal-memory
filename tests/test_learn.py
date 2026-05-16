"""Tests for learning loop and experience extraction."""
import tempfile

from agim.core.system import AGIMSystem
from agim.learn.experience import Experience, LessonExtractor
from agim.learn.loop import VerifiedLearningLoop


def test_lesson_extractor_positive():
    ext = LessonExtractor()
    exp = Experience(question="X?", model_answer="Y", user_feedback="Z", rating=4)
    lesson = ext.extract(exp)
    assert lesson is not None
    assert lesson.candidate.kind == "feedback"
    assert lesson.candidate.confidence > 0.8


def test_lesson_extractor_negative():
    ext = LessonExtractor()
    exp = Experience(question="X?", model_answer="Y", user_feedback="", rating=1)
    lesson = ext.extract(exp)
    assert lesson is None


def test_learning_loop():
    with tempfile.TemporaryDirectory() as tmp:
        agim = AGIMSystem(workdir=tmp)
        loop = VerifiedLearningLoop(agim)

        report = loop.process_feedback(
            question="Capital of France?",
            model_answer="I don't know",
            user_feedback="Paris",
            rating=5,
        )
        assert report is not None
        assert report.passed
        assert loop.total_feedback == 1

        resp = agim.ask("Capital of France?")
        assert "Paris" in resp.answer
