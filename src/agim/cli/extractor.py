"""Memory Extractor — extract structured MemoryCandidate from natural language."""
import re
from ..core.state import Intent, MemoryCandidate


class MemoryExtractor:
    CORRECTION_RE = re.compile(
        r"(?:no,?\s*|actually,?\s*|wrong,?\s*|it's\s+not,?\s*)"
        r"(.+?)\s*(?:is|was|are|were)\s*(?:not|never)\s*(.+?)(?:[.,]|$)",
        re.IGNORECASE,
    )
    IS_RE = re.compile(r"(.+?)\s+is\s+(.+?)(?:[.,!]|$)", re.IGNORECASE)
    ARE_RE = re.compile(r"(.+?)\s+are\s+(.+?)(?:[.,!]|$)", re.IGNORECASE)
    WAS_RE = re.compile(r"(.+?)\s+was\s+(.+?)(?:[.,!]|$)", re.IGNORECASE)

    def extract(self, text: str, intent: Intent) -> MemoryCandidate:
        if intent == Intent.FACT_CORRECT:
            return self._extract_correction(text)
        elif intent == Intent.FACT_TEACH:
            return self._extract_fact(text)
        elif intent == Intent.PREFERENCE:
            return self._extract_preference(text)
        elif intent == Intent.FEEDBACK:
            return self._extract_feedback(text)
        elif intent == Intent.FORGET:
            return self._extract_forget(text)
        else:
            return MemoryCandidate(
                question=text.strip().rstrip("?"),
                answer="[recorded]",
                kind=intent.value,
            )

    def _extract_fact(self, text: str) -> MemoryCandidate:
        text = text.strip()
        for prefix in ["teach me ", "learn this: ", "remember ", "note that ",
                       "it is known that ", "the fact is "]:
            if text.lower().startswith(prefix):
                text = text[len(prefix):]
        for pattern, kind in [(self.IS_RE, "subject_is"), (self.ARE_RE, "subject_are"),
                               (self.WAS_RE, "subject_was")]:
            m = pattern.search(text)
            if m:
                subject, attribute = m.group(1).strip(), m.group(2).strip().rstrip(".")
                question = f"What is {attribute}?"
                answer = subject
                return MemoryCandidate(question=question, answer=answer,
                                      kind="fact_teach", metadata={"pattern": kind})
        return MemoryCandidate(question=text, answer="[recorded]", kind="fact_teach")

    def _extract_correction(self, text: str) -> MemoryCandidate:
        text = text.strip()
        question = "What is " + text.split(" is ")[0].lstrip("no, No, actually, Actually, ")
        answer = text.split(" is ")[-1].rstrip(".")
        return MemoryCandidate(question=question.rstrip("?"), answer=answer,
                              kind="fact_correct")

    def _extract_preference(self, text: str) -> MemoryCandidate:
        return MemoryCandidate(
            question="user_preference",
            answer=text.strip(),
            kind="preference",
            confidence=0.8,
        )

    def _extract_feedback(self, text: str) -> MemoryCandidate:
        return MemoryCandidate(
            question="user_feedback",
            answer=text.strip(),
            kind="feedback",
            confidence=0.7,
        )

    def _extract_forget(self, text: str) -> MemoryCandidate:
        topic = text.strip()
        for prefix in ["forget ", "delete that ", "remove that ", "undo that ", "erase "]:
            if topic.lower().startswith(prefix):
                topic = topic[len(prefix):]
        return MemoryCandidate(
            question=topic,
            answer="[forgotten]",
            kind="forget",
            confidence=1.0,
        )
