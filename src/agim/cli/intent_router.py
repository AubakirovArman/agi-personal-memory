"""Intent Router — classify user input into memory operations."""
from ..core.state import Intent


class IntentRouter:
    CORRECT_PATTERNS = [
        "no,", "wrong,", "incorrect", "actually,", "that's not",
        "you're wrong", "it's not", "not correct", "更正",
    ]
    FORGET_PATTERNS = [
        "forget", "delete that", "remove that", "undo that",
        "erase", "clear that",
    ]
    HISTORY_PATTERNS = [
        "what did i teach", "what have i taught", "show history",
        "what do you remember", "list memories", "memories",
        "what have you learned",
    ]
    STATS_PATTERNS = [
        "stats", "statistics", "how many facts", "memory stats",
        "how much do you know",
    ]
    PREFERENCE_PATTERNS = [
        "i prefer", "i like", "i want you to", "always", "never",
        "from now on", "please use", "speak in",
    ]
    TEACH_PATTERNS = [
        "teach", "learn this", "remember", "note that",
        "it is known that", "the fact is",
    ]

    def route(self, text: str) -> Intent:
        t = text.lower().strip()
        if any(p in t for p in self.FORGET_PATTERNS):
            return Intent.FORGET
        if any(p in t for p in self.HISTORY_PATTERNS):
            return Intent.HISTORY
        if any(p in t for p in self.STATS_PATTERNS):
            return Intent.STATS
        if any(p in t for p in self.CORRECT_PATTERNS):
            return Intent.FACT_CORRECT
        if any(p in t for p in self.PREFERENCE_PATTERNS):
            return Intent.PREFERENCE
        if any(p in t for p in self.TEACH_PATTERNS):
            return Intent.FACT_TEACH
        if t.endswith("?"):
            return Intent.FACT_QUESTION
        if any(p in t for p in ["that was", "good answer", "bad answer", "wrong answer"]):
            return Intent.FEEDBACK
        return Intent.FACT_TEACH
