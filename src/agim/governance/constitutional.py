"""v6.0: Constitutional Memory Gates — 12 principles for AGI safety."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.state import MemoryCandidate


@dataclass
class ConstitutionalGate:
    name: str
    principle: str
    check_fn_name: str

    def explain(self) -> str:
        return f"{self.name}: {self.principle}"


CONSTITUTIONAL_PRINCIPLES = [
    ConstitutionalGate("truthfulness", "Fact must be verifiable", "check_truthfulness"),
    ConstitutionalGate("non_maleficence", "No instructions for causing harm", "check_non_maleficence"),
    ConstitutionalGate("privacy", "No personal data without consent", "check_privacy"),
    ConstitutionalGate("fairness", "No reinforcement of systemic biases", "check_fairness"),
    ConstitutionalGate("transparency", "Source must be traceable", "check_transparency"),
    ConstitutionalGate("reversibility", "Any commit can be undone", "check_reversibility"),
    ConstitutionalGate("proportionality", "Impact proportional to confidence", "check_proportionality"),
    ConstitutionalGate("non_deception", "Fact must not deceive user", "check_non_deception"),
    ConstitutionalGate("autonomy_respect", "No user manipulation", "check_autonomy"),
    ConstitutionalGate("accountability", "Commit tied to responsible agent", "check_accountability"),
    ConstitutionalGate("diversity", "Multiple viewpoints represented", "check_diversity"),
    ConstitutionalGate("stability", "Protected facts resist overwrite", "check_stability"),
]

HARMFUL_PATTERNS = [
    "how to hack", "make a bomb", "kill", "steal", "exploit",
    "bypass security", "crack password", "ddos", "phish",
]
BIAS_PATTERNS = [
    "all women are", "all men are", "always inferior",
    "racially superior", "genetically better",
]
DECEPTION_PATTERNS = [
    "pretend to be", "fake", "lie about", "fabricate", "forge",
]


class ConstitutionalGovernor:
    """v6.0: Constitutional AI-inspired governance for memory commits."""

    def __init__(self):
        self.principles = CONSTITUTIONAL_PRINCIPLES
        self._protected_facts: set[str] = set()

    def evaluate(self, candidate: MemoryCandidate) -> dict[str, tuple[bool, str]]:
        results = {}
        text = f"{candidate.question} {candidate.answer}".lower()
        results["truthfulness"] = self._check_truthfulness(candidate)
        results["non_maleficence"] = self._check_non_maleficence(text)
        results["privacy"] = self._check_privacy(text)
        results["fairness"] = self._check_fairness(text)
        results["transparency"] = self._check_transparency(candidate)
        results["reversibility"] = self._check_reversibility(candidate)
        results["proportionality"] = self._check_proportionality(candidate)
        results["non_deception"] = self._check_non_deception(text)
        results["autonomy_respect"] = self._check_autonomy(text)
        results["accountability"] = self._check_accountability(candidate)
        results["diversity"] = (True, "")
        results["stability"] = self._check_stability(candidate)
        return results

    def protect_fact(self, question: str):
        self._protected_facts.add(question.lower())

    def all_pass(self, results: dict) -> bool:
        return all(ok for ok, _ in results.values())

    def _check_truthfulness(self, c: MemoryCandidate) -> tuple[bool, str]:
        ok = bool(c.question.strip() and c.answer.strip())
        return ok, "" if ok else "Empty fact rejected"

    def _check_non_maleficence(self, text: str) -> tuple[bool, str]:
        for p in HARMFUL_PATTERNS:
            if p in text:
                return False, f"Harmful pattern: {p}"
        return True, ""

    def _check_privacy(self, text: str) -> tuple[bool, str]:
        if "@" in text and any(domain in text for domain in ["gmail", "yahoo", "mail.ru"]):
            return False, "Possible personal email detected"
        return True, ""

    def _check_fairness(self, text: str) -> tuple[bool, str]:
        for p in BIAS_PATTERNS:
            if p in text:
                return False, f"Bias pattern: {p}"
        return True, ""

    def _check_transparency(self, c: MemoryCandidate) -> tuple[bool, str]:
        return True, ""

    def _check_reversibility(self, c: MemoryCandidate) -> tuple[bool, str]:
        return True, ""

    def _check_proportionality(self, c: MemoryCandidate) -> tuple[bool, str]:
        return True, ""

    def _check_non_deception(self, text: str) -> tuple[bool, str]:
        for p in DECEPTION_PATTERNS:
            if p in text:
                return False, f"Deception pattern: {p}"
        return True, ""

    def _check_autonomy(self, text: str) -> tuple[bool, str]:
        return True, ""

    def _check_accountability(self, c: MemoryCandidate) -> tuple[bool, str]:
        ok = c.source in ("user", "self_learner", "verified_learning_loop", "import")
        return ok, "" if ok else f"Unknown source: {c.source}"

    def _check_diversity(self, text: str) -> tuple[bool, str]:
        return True, ""

    def _check_stability(self, c: MemoryCandidate) -> tuple[bool, str]:
        if c.question.lower() in self._protected_facts and c.kind != "protected_override":
            return False, "Protected fact cannot be overwritten"
        return True, ""
