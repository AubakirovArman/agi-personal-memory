"""Tests for governance: budget, risk, provenance."""
import tempfile
from pathlib import Path

from agim.governance.budget import MemoryBudget
from agim.governance.risk import RiskLedger, RiskEntry
from agim.governance.provenance import ProvenanceChain


def test_budget_allows_normal():
    b = MemoryBudget()
    ok, _ = b.check(current_total=100)
    assert ok


def test_budget_blocks_total():
    b = MemoryBudget(max_total_facts=10)
    ok, reason = b.check(current_total=10)
    assert not ok
    assert "Total" in reason


def test_budget_record_commit():
    b = MemoryBudget()
    b.record_commit()
    assert b._daily_count == 1
    assert b._hourly_count == 1


def test_risk_ledger_low():
    rl = RiskLedger()
    entry = rl.assess("mem1", {"kind": "fact_teach", "confidence": 1.0})
    assert entry.risk_score == 0.0
    assert not entry.auto_rollback


def test_risk_ledger_correction_risk():
    rl = RiskLedger()
    entry = rl.assess("mem2", {"kind": "fact_correct", "confidence": 0.6})
    assert entry.risk_score >= 5.0


def test_provenance_chain():
    pc = ProvenanceChain(Path(tempfile.mkdtemp()) / "chain.json")
    rec = pc.add("c1", "Q?", "A", "wal_recipe")
    assert pc.length == 1
    assert rec.previous_hash == "0" * 16
    assert pc.verify_chain()


def test_provenance_chain_multi():
    pc = ProvenanceChain(Path(tempfile.mkdtemp()) / "chain.json")
    pc.add("c1", "Q1?", "A1", "wal_recipe")
    pc.add("c2", "Q2?", "A2", "retrieval")
    assert pc.length == 2
    assert pc.verify_chain()
