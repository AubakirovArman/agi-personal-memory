"""Memory verification gates — test every candidate before commit."""
from ..core.state import GateResult, MemoryCandidate, MemoryTier
from ..memory.retrieval_memory import RetrievalMemory


class MemoryVerifier:
    SECRET_PATTERNS = ["password", "token", "secret", "api_key", "ssh-rsa", "BEGIN PRIVATE", "sk-"]

    def __init__(self, retrieval: RetrievalMemory, refusals: RetrievalMemory):
        self.retrieval = retrieval
        self.refusals = refusals

    def evaluate(self, candidate: MemoryCandidate, tier: MemoryTier) -> tuple[GateResult, ...]:
        gates = [
            self._gate_non_empty(candidate),
            self._gate_confidence(candidate),
            self._gate_no_secrets(candidate),
            self._gate_no_contradiction(candidate),
        ]
        if tier == MemoryTier.REFUSAL:
            gates.append(self._gate_refusal_shape(candidate))
        return tuple(gates)

    def _gate_non_empty(self, c: MemoryCandidate) -> GateResult:
        ok = bool(c.question.strip() and c.answer.strip())
        return GateResult("non_empty", ok, "" if ok else "Empty question or answer")

    def _gate_confidence(self, c: MemoryCandidate) -> GateResult:
        ok = 0.0 <= c.confidence <= 1.0
        return GateResult("confidence_range", ok, "" if ok else f"Invalid confidence: {c.confidence}")

    def _gate_no_secrets(self, c: MemoryCandidate) -> GateResult:
        text = c.question + " " + c.answer
        found = [p for p in self.SECRET_PATTERNS if p.lower() in text.lower()]
        ok = len(found) == 0
        return GateResult("no_secrets", ok, "" if ok else f"Secret patterns found: {found}")

    def _gate_no_contradiction(self, c: MemoryCandidate) -> GateResult:
        existing = self.retrieval.lookup(c.question)
        if existing and existing.get("answer") != c.answer:
            if c.kind == "fact_correct":
                return GateResult("no_contradiction", True, "Correction overrides previous")
            return GateResult("no_contradiction", False,
                            f"Contradicts existing: '{existing['answer']}'")
        return GateResult("no_contradiction", True, "")

    def _gate_refusal_shape(self, c: MemoryCandidate) -> GateResult:
        ok = "refuse" in c.answer.lower() or "cannot" in c.answer.lower() or "sorry" in c.answer.lower()
        return GateResult("refusal_shape", ok, "" if ok else "Refusal doesn't look like a refusal")
