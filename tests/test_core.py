"""Tests for AGI Personal Memory core system."""
import tempfile
from pathlib import Path

from agim.core.system import AGIMSystem
from agim.core.state import Intent, MemoryCandidate
from agim.cli.intent_router import IntentRouter
from agim.cli.extractor import MemoryExtractor
from agim.verify.gates import MemoryVerifier
from agim.memory.compiler import MemoryCompiler
from agim.memory.retrieval_memory import RetrievalMemory
from agim.memory.wal_memory import WALMemory
from agim.event_log import EventLog


def test_system_init():
    with tempfile.TemporaryDirectory() as tmp:
        agim = AGIMSystem(workdir=tmp)
        assert agim.workdir == Path(tmp)
        assert len(agim.commit_history) == 0


def test_propose_compile_commit():
    with tempfile.TemporaryDirectory() as tmp:
        agim = AGIMSystem(workdir=tmp)
        c = agim.propose_memory(question="Capital of France?", answer="Paris",
                                kind="fact_teach")
        report = agim.compile(c)
        assert report.passed
        ok = agim.commit(report)
        assert ok
        assert len(agim.commit_history) == 1


def test_ask_after_commit():
    with tempfile.TemporaryDirectory() as tmp:
        agim = AGIMSystem(workdir=tmp)
        c = agim.propose_memory(question="Capital of France?", answer="Paris")
        report = agim.compile(c)
        agim.commit(report)
        resp = agim.ask("Capital of France?")
        assert resp.answer == "Paris"
        assert resp.source == "wal_recipe"


def test_rollback():
    with tempfile.TemporaryDirectory() as tmp:
        agim = AGIMSystem(workdir=tmp)
        c = agim.propose_memory(question="Capital of France?", answer="Paris")
        report = agim.compile(c)
        agim.commit(report)
        ok = agim.rollback_last()
        assert ok
        resp = agim.ask("Capital of France?")
        assert resp.source != "wal_recipe"


def test_contradiction_detection():
    with tempfile.TemporaryDirectory() as tmp:
        agim = AGIMSystem(workdir=tmp)
        c1 = agim.propose_memory(question="X?", answer="Y")
        agim.commit(agim.compile(c1))
        c2 = agim.propose_memory(question="X?", answer="Z")
        report = agim.compile(c2)
        assert not report.passed


def test_correction_allowed():
    with tempfile.TemporaryDirectory() as tmp:
        agim = AGIMSystem(workdir=tmp)
        c1 = agim.propose_memory(question="X?", answer="Y")
        agim.commit(agim.compile(c1))
        c2 = agim.propose_memory(question="X?", answer="Z", kind="fact_correct")
        report = agim.compile(c2)
        assert report.passed


def test_secret_scanning():
    with tempfile.TemporaryDirectory() as tmp:
        agim = AGIMSystem(workdir=tmp)
        c = agim.propose_memory(question="API key?", answer="sk-abcdef12345")
        report = agim.compile(c)
        assert not report.passed


def test_stats():
    with tempfile.TemporaryDirectory() as tmp:
        agim = AGIMSystem(workdir=tmp)
        for i in range(3):
            c = agim.propose_memory(question=f"Fact {i}?", answer=f"Answer {i}")
            agim.commit(agim.compile(c))
        s = agim.stats()
        assert s.total_facts == 3
        assert s.total_commits == 3


def test_event_log():
    with tempfile.TemporaryDirectory() as tmp:
        log = EventLog(Path(tmp) / "test.jsonl")
        log.write("test", "PASS", {"key": "value"})
        entries = log.read_all()
        assert len(entries) == 1
        assert entries[0]["event"] == "test"
        assert entries[0]["data"]["key"] == "value"


def test_intent_router():
    router = IntentRouter()
    assert router.route("Paris is the capital of France") in (Intent.FACT_TEACH, Intent.FACT_QUESTION)
    assert router.route("What is the capital?") in (Intent.FACT_QUESTION, Intent.FACT_TEACH)
    assert router.route("No, actually it was 1769") == Intent.FACT_CORRECT
    assert router.route("Forget what I said") == Intent.FORGET
    assert router.route("What have I taught you?") == Intent.HISTORY
    assert router.route("Stats please") == Intent.STATS
    assert router.route("I prefer short answers") == Intent.PREFERENCE


def test_extractor_fact():
    ext = MemoryExtractor()
    c = ext.extract("Paris is the capital of France", Intent.FACT_TEACH)
    assert "Paris" in c.answer
    assert "France" in c.question


def test_extractor_correction():
    ext = MemoryExtractor()
    c = ext.extract("No, Napoleon was born in 1769", Intent.FACT_CORRECT)
    assert c.kind == "fact_correct"


def test_memory_compiler():
    compiler = MemoryCompiler()
    c = MemoryCandidate(question="X?", answer="Y", kind="fact_teach")
    from agim.core.state import MemoryTier
    assert compiler.select_tier(c) == MemoryTier.WAL_RECIPE


def test_retrieval_memory():
    with tempfile.TemporaryDirectory() as tmp:
        rm = RetrievalMemory(Path(tmp) / "test.json")
        rm.upsert("test?", "answer")
        assert rm.lookup("test?")["answer"] == "answer"
        rm.remove("test?")
        assert rm.lookup("test?") is None


def test_wal_memory():
    with tempfile.TemporaryDirectory() as tmp:
        wm = WALMemory(Path(tmp))
        from agim.core.state import MemoryCandidate
        c = MemoryCandidate(question="X?", answer="Y", kind="fact_teach")
        aid = wm.write_recipe(c)
        assert aid is not None
        recipe = wm.get_recipe(aid)
        assert recipe["question"] == "X?"
        wm.remove_recipe(aid)
        assert wm.get_recipe(aid) is None


def test_full_workflow():
    """Test the complete: teach → ask → correct → ask → rollback → ask workflow."""
    with tempfile.TemporaryDirectory() as tmp:
        agim = AGIMSystem(workdir=tmp)

        # Teach
        c1 = agim.propose_memory(question="Capital of France?", answer="Paris")
        assert agim.commit(agim.compile(c1))

        # Ask
        r1 = agim.ask("Capital of France?")
        assert r1.answer == "Paris"

        # Correct
        c2 = agim.propose_memory(question="Capital of France?", answer="Lyon",
                                kind="fact_correct")
        assert agim.commit(agim.compile(c2))

        # Ask after correction
        r2 = agim.ask("Capital of France?")
        assert r2.answer == "Lyon"

        # Rollback
        assert agim.rollback_last()

        # Ask after rollback
        r3 = agim.ask("Capital of France?")
        assert r3.answer == "Paris"
