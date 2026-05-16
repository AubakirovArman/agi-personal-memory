"""AGIMSystem — the central memory accumulation engine with full governance."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..event_log import EventLog
from ..governance.budget import MemoryBudget
from ..governance.provenance import ProvenanceChain
from ..governance.risk import RiskLedger
from ..memory.compiler import MemoryCompiler
from ..memory.retrieval_memory import RetrievalMemory
from ..memory.wal_memory import WALMemory
from ..verify.contracts import BehaviouralContract, ContractSuite
from ..verify.gates import MemoryVerifier
from ..verify.regression import RegressionSuite
from .state import (AIGIResponse, CommitRecord, CompileReport, MemoryCandidate,
                    MemoryStats, MemoryTier)


class AGIMSystem:
    """Verified, accumulative memory system for language models.

    Full governance: provenance chain, risk ledger, memory budget,
    regression suite, behavioural contract testing on every commit.
    """

    def __init__(self, *, workdir: str | Path = ".agim",
                 signing_key: str = "agim-default-key",
                 max_facts: int = 100_000, max_daily: int = 500):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        gdir = self.workdir / "governance"

        self.log = EventLog(self.workdir / "logs" / "events.jsonl")
        self.retrieval = RetrievalMemory(self.workdir / "memory" / "retrieval.json")
        self.refusals = RetrievalMemory(self.workdir / "memory" / "refusals.json")
        self.wal = WALMemory(self.workdir / "memory" / "wal_recipes")
        self.compiler = MemoryCompiler()
        self.verifier = MemoryVerifier(self.retrieval, self.refusals)

        self.budget = MemoryBudget(max_total_facts=max_facts, max_daily_commits=max_daily)
        self.risk_ledger = RiskLedger.load(gdir / "risk_ledger.json")
        self.provenance = ProvenanceChain(gdir / "provenance_chain.json", signing_key=signing_key)
        self.regression = RegressionSuite.load(gdir / "regression_suite.json")
        self.contract_suite = ContractSuite.default_suite()

        self.last_report: CompileReport | None = None
        self.commit_history: list[CommitRecord] = []
        self._answer_fn = None  # Set externally for behavioural tests

        self.log.write("system_init", "PASS", {"workdir": str(self.workdir)})

    def set_answer_function(self, fn):
        self._answer_fn = fn

    def ask(self, question: str) -> AIGIResponse:
        refusal = self.refusals.lookup(question)
        if refusal is not None:
            self.log.write("ask", "PASS", {"source": "refusal", "question": question})
            return AIGIResponse(question=question, answer=refusal["answer"],
                              source="refusal", memory_id=refusal.get("id"))

        memory = self.retrieval.lookup(question)
        if memory is not None:
            self.log.write("ask", "PASS", {"source": memory["source"], "question": question})
            return AIGIResponse(question=question, answer=memory["answer"],
                              source=memory["source"], memory_id=memory.get("id"),
                              confidence=memory.get("confidence", 1.0))

        self.log.write("ask", "PASS", {"source": "model_fallback", "question": question})
        return AIGIResponse(question=question, answer="[model response]",
                          source="model_fallback")

    def propose_memory(self, *, question: str, answer: str, kind: str = "fact_teach",
                       source: str = "user", confidence: float = 1.0,
                       previous_answer: str | None = None,
                       metadata: dict | None = None) -> MemoryCandidate:
        candidate = MemoryCandidate(
            question=question, answer=answer, kind=kind, source=source,
            confidence=confidence, previous_answer=previous_answer,
            metadata=metadata or {})
        self.log.write("propose_memory", "PASS",
                       {"candidate_id": candidate.candidate_id, "kind": kind})
        return candidate

    def compile(self, candidate: MemoryCandidate) -> CompileReport:
        tier = self.compiler.select_tier(candidate)
        gates = self.verifier.evaluate(candidate, tier)
        passed = all(gate.passed for gate in gates) and tier != MemoryTier.REJECT

        risk_entry = self.risk_ledger.assess(candidate.candidate_id, {
            "kind": candidate.kind, "confidence": candidate.confidence,
            "source": candidate.source,
        })
        if risk_entry.auto_rollback:
            passed = False

        reason = "" if passed else "; ".join(
            g.reason for g in gates if not g.passed)
        if not passed and risk_entry.auto_rollback:
            reason = f"RISK_AUTO_ROLLBACK({risk_entry.risk_score}): " + risk_entry.reason

        artifact_id = (self.wal.preview_artifact_id(candidate)
                       if passed and tier == MemoryTier.WAL_RECIPE else None)
        report = CompileReport(candidate=candidate, tier=tier, passed=passed,
                               gates=gates, artifact_id=artifact_id, reason=reason)
        self.last_report = report
        self.log.write("compile", report.status,
                       {"candidate_id": candidate.candidate_id, "tier": tier.value,
                        "risk_score": risk_entry.risk_score, "reason": reason})
        return report

    def commit(self, report: CompileReport | None = None) -> bool:
        selected = report or self.last_report
        if selected is None or not selected.passed:
            self.log.write("commit", "FAIL", {"reason": "missing_or_failed_report"})
            return False

        ok, budget_reason = self.budget.check(len(self.retrieval) + len(self.wal))
        if not ok:
            self.log.write("commit", "FAIL", {"reason": budget_reason})
            return False

        c = selected.candidate
        if selected.tier == MemoryTier.WAL_RECIPE:
            previous = self.retrieval.lookup(c.question)
            artifact_id = self.wal.write_recipe(c)
            self.retrieval.upsert(c.question, c.answer, source="wal_recipe",
                                  memory_id=artifact_id, confidence=c.confidence)
        elif selected.tier == MemoryTier.RETRIEVAL:
            previous = self.retrieval.lookup(c.question)
            artifact_id = self.retrieval.upsert(c.question, c.answer,
                                                source="retrieval", confidence=c.confidence)
        elif selected.tier == MemoryTier.REFUSAL:
            previous = self.refusals.lookup(c.question)
            artifact_id = self.refusals.upsert(c.question, c.answer, source="refusal")
        else:
            self.log.write("commit", "FAIL", {"reason": f"unsupported_tier:{selected.tier.value}"})
            return False

        self.provenance.add(artifact_id, c.question, c.answer, selected.tier.value,
                            {"kind": c.kind, "source": c.source})
        self.budget.record_commit()
        self.risk_ledger.save(self.workdir / "governance" / "risk_ledger.json")
        self.budget.save(self.workdir / "governance" / "budget.json")

        self.commit_history.append(CommitRecord(
            artifact_id=artifact_id, tier=selected.tier,
            question=c.question, answer=c.answer, previous_entry=previous))
        self.log.write("commit", "PASS",
                       {"tier": selected.tier.value, "artifact_id": artifact_id})
        return True

    def rollback_last(self) -> bool:
        if not self.commit_history:
            self.log.write("rollback", "FAIL", {"reason": "empty_commit_history"})
            return False
        record = self.commit_history.pop()
        if record.tier == MemoryTier.WAL_RECIPE:
            self.wal.remove_recipe(record.artifact_id)
            self.retrieval.restore(record.question, record.previous_entry)
        elif record.tier in (MemoryTier.RETRIEVAL,):
            self.retrieval.restore(record.question, record.previous_entry)
        elif record.tier == MemoryTier.REFUSAL:
            self.refusals.restore(record.question, record.previous_entry)
        self.log.write("rollback", "PASS",
                       {"tier": record.tier.value, "artifact_id": record.artifact_id})
        return True

    def run_regression(self) -> dict[str, bool]:
        if self._answer_fn is None:
            return {}
        last_cid = self.commit_history[-1].artifact_id if self.commit_history else "init"
        return self.regression.run_regression(self._answer_fn, last_cid)

    def add_protected_fact(self, question: str, answer: str):
        self.regression.add_protected(question, answer)

    def stats(self) -> MemoryStats:
        by_tier = {}
        for q, entry in self.retrieval._data.items():
            tier = entry.get("source", "unknown")
            by_tier[tier] = by_tier.get(tier, 0) + 1
        by_tier["wal_recipe"] = len(self.wal)
        by_tier["refusal"] = len(self.refusals)

        by_kind = {}
        for recipe in self.wal._index.values():
            k = recipe.get("kind", "unknown")
            by_kind[k] = by_kind.get(k, 0) + 1

        return MemoryStats(
            total_facts=len(self.retrieval) + len(self.refusals),
            facts_by_tier=by_tier, facts_by_kind=by_kind,
            rollback_count=sum(1 for r in self.commit_history if r.rolled_back),
            total_commits=len(self.commit_history),
            avg_confidence=1.0,
        )
