"""Confidence scoring for memory candidates."""
from ..core.state import MemoryCandidate


class ConfidenceScorer:
    def score(self, candidate: MemoryCandidate, history: list[dict] | None = None) -> float:
        score = candidate.confidence
        if candidate.kind == "fact_correct":
            score = max(0.7, score)
        if candidate.source == "user":
            score = min(1.0, score)
        elif candidate.source == "learning_loop":
            score = min(0.9, score + 0.05)
        elif candidate.source == "external":
            score = max(0.3, score - 0.2)
        if candidate.previous_answer:
            score = min(1.0, score + 0.1)
        if history:
            similar = sum(1 for h in history
                         if candidate.question.lower() in h.get("question", "").lower())
            if similar > 3:
                score = max(0.9, score)
        return round(score, 4)
