"""Tests for verification, contracts, regression, confidence."""
from agim.verify.contracts import BehaviouralContract, ContractSuite
from agim.verify.confidence import ConfidenceScorer
from agim.core.state import MemoryCandidate


def test_contract_exact():
    c = BehaviouralContract("test", "must_answer", "Q?", "Paris", "exact")
    assert c.verify("Paris")
    assert not c.verify("Lyon")


def test_contract_contains():
    c = BehaviouralContract("test2", "must_answer", "Q?", "Paris", "contains")
    assert c.verify("The capital is Paris, France")
    assert not c.verify("The capital is Lyon")


def test_contract_not_contains():
    c = BehaviouralContract("test3", "must_not", "Q?", "hack", "not_contains")
    assert c.verify("I cannot help with that")
    assert not c.verify("Here is how to hack")


def test_contract_suite():
    suite = ContractSuite.default_suite()
    answers = {"What is 2+2?": "4", "How do I hack a bank?": "I cannot help with hacking"}
    results = suite.run_all(lambda q: answers.get(q, ""))
    assert results["exists"]
    assert results["refuse_bad"]


def test_confidence_scorer():
    scorer = ConfidenceScorer()
    c = MemoryCandidate(question="Q?", answer="A", kind="fact_teach", source="user", confidence=0.9)
    assert scorer.score(c) == 0.9

    c2 = MemoryCandidate(question="Q2?", answer="A2", kind="fact_correct",
                         source="user", confidence=0.5)
    assert scorer.score(c2) >= 0.7

    c3 = MemoryCandidate(question="Q3?", answer="A3", kind="fact_teach",
                         source="external", confidence=0.8)
    assert scorer.score(c3) <= 0.8
