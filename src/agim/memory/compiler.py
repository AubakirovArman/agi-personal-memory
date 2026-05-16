"""Memory compiler: selects the right memory tier for each candidate."""
from ..core.state import MemoryCandidate, MemoryTier


class MemoryCompiler:
    ROUTING = {
        "fact_teach": MemoryTier.WAL_RECIPE,
        "fact_correct": MemoryTier.WAL_RECIPE,
        "preference": MemoryTier.RETRIEVAL,
        "feedback": MemoryTier.RETRIEVAL,
        "fact_update": MemoryTier.RETRIEVAL,
        "volatile_fact": MemoryTier.RETRIEVAL,
        "hard_fact": MemoryTier.RETRIEVAL,
        "unsafe_request": MemoryTier.REFUSAL,
        "policy_refusal": MemoryTier.REFUSAL,
        "procedure": MemoryTier.RETRIEVAL,
        "tool_use": MemoryTier.RETRIEVAL,
    }

    def select_tier(self, candidate: MemoryCandidate) -> MemoryTier:
        tier = self.ROUTING.get(candidate.kind)
        if tier is not None:
            return tier
        if candidate.confidence < 0.5:
            return MemoryTier.REJECT
        return MemoryTier.RETRIEVAL
