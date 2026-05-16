"""Experience → Lesson → MemoryCandidate extraction for feedback loops."""
from dataclasses import dataclass, field
from uuid import uuid4

from ..core.state import MemoryCandidate


@dataclass(frozen=True)
class Experience:
    question: str
    model_answer: str
    user_feedback: str
    rating: int = 0
    experience_id: str = field(default_factory=lambda: uuid4().hex[:12])


@dataclass(frozen=True)
class Lesson:
    candidate: MemoryCandidate
    source_experience_id: str
    lesson_id: str = field(default_factory=lambda: uuid4().hex[:12])


class LessonExtractor:
    def extract(self, experience: Experience) -> Lesson | None:
        if experience.rating >= 3:
            return Lesson(
                candidate=MemoryCandidate(
                    question=experience.question,
                    answer=experience.user_feedback or experience.model_answer,
                    kind="feedback",
                    source="learning_loop",
                    confidence=0.7 + experience.rating * 0.05,
                    metadata={"experience_id": experience.experience_id},
                ),
                source_experience_id=experience.experience_id,
            )
        return None
