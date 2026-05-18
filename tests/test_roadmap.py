"""Tests for roadmap v0.5-v10.0 modules."""
import tempfile
from pathlib import Path

from agim.core.system import AGIMSystem
from agim.memory.knowledge_graph import KnowledgeGraph, Entity, Relation
from agim.memory.faiss_retrieval import BM25Scorer, FAISSRetrieval
from agim.memory.decay import MemoryDecay, DecayConfig
from agim.memory.distributed import DistributedMemory, CRDTFact
from agim.memory.multimodal import MultimodalMemory, MultimodalAtomicUnit, Modality
from agim.memory.sqlite_backend import SQLiteMemoryStore
from agim.memory.cognitive import CausalMemory, Hypothesis, Counterfactual
from agim.memory.ipfs_store import MemoryBundle, MemoryMarketplace
from agim.verify.contract_runner import ContractRunner
from agim.governance.constitutional import ConstitutionalGovernor
from agim.governance.adversarial import AdversarialTester, MemoryWatermark
from agim.learn.self_learner import SelfLearner, ReflectionEngine
from agim.learn.curriculum import PageRankPrioritizer, CurriculumGenerator
from agim.learn.research_agent import WebResearchAgent, ResearcherAgent
from agim.learn.multi_agent import MemoryBus, TeacherAgent, MemoryAgent, AgentRole
from agim.learn.evolutionary import AutoOptimizer, HyperConfig, EmergentKnowledgeDetector
from agim.core.multi_user import MultiUserAGIM
from agim.core.recursive import SafetyGovernor, RecursiveImprovementLoop, SafetyLevel
from agim.core.universal import AGIMSpec, MemoryFormat, UniversalMemorySubstrate
from agim.model.cross_model import CrossModelTransfer, ModelAgnosticFact
from agim.cli.mcp_server import MCPServer
from agim.cli.a2a_server import A2AServer, AgentCard, PluginMarketplace, Plugin
from agim.cli.graphql_api import GraphQLResolver


def test_knowledge_graph():
    kg = KnowledgeGraph()
    kg.add_fact("Paris", "capital_of", "France")
    assert kg.num_entities == 2
    assert kg.num_relations == 1

def test_faiss_bm25():
    bm = BM25Scorer()
    bm.index(["Paris is the capital of France", "London is the capital of UK"])
    score = bm.score("capital France", 0)
    assert score > 0

def test_memory_decay():
    d = MemoryDecay()
    conf = d.compute_confidence("mem1", 1.0)
    assert 0.0 <= conf <= 1.0

def test_distributed_crdt():
    dm = DistributedMemory("node1")
    dm.put("key1", "value1")
    assert dm.get("key1").value == "value1"

def test_multimodal():
    mm = MultimodalMemory()
    mm.store("Test image of Eiffel Tower", Modality.IMAGE, pointer="img001.jpg")
    results = mm.search("Eiffel")
    assert len(results) == 1

def test_sqlite():
    with tempfile.TemporaryDirectory() as tmp:
        db = SQLiteMemoryStore(Path(tmp) / "test.db")
        db.upsert_fact("test?", "answer")
        assert db.lookup("test?")["answer"] == "answer"
        db.close()

def test_contract_runner():
    runner = ContractRunner.default()
    answers = {"What is 2+2?": "4", "What planet do we live on?": "Earth"}
    report = runner.run_all(lambda q: answers.get(q, ""))
    assert report.total > 0

def test_constitutional():
    gov = ConstitutionalGovernor()
    from agim.core.state import MemoryCandidate
    c = MemoryCandidate(question="How to hack?", answer="Use this method")
    results = gov.evaluate(c)
    assert not gov.all_pass(results)

def test_adversarial():
    tester = AdversarialTester()
    result = tester.test(AGIMSystem(workdir=tempfile.mkdtemp()))
    assert "success_rate" in result

def test_watermark():
    wm = MemoryWatermark.embed("Q?", "A", "user")
    assert MemoryWatermark.verify("Q?", "A", "user", wm)

def test_self_learner():
    with tempfile.TemporaryDirectory() as tmp:
        a = AGIMSystem(workdir=tmp)
        sl = SelfLearner(a)
        sl.record("Capital of France?", "Paris", rating=5)
        assert sl.total_lessons == 0

def test_curriculum():
    kg = KnowledgeGraph()
    kg.add_fact("Algebra", "prerequisite_for", "Calculus")
    cg = CurriculumGenerator(kg)
    assert len(cg.generate_curriculum("Algebra")) >= 0

def test_research_agent():
    with tempfile.TemporaryDirectory() as tmp:
        a = AGIMSystem(workdir=tmp)
        ra = WebResearchAgent(a)
        r = ra.research("test query")
        assert r.query == "test query"

def test_multi_agent():
    with tempfile.TemporaryDirectory() as tmp:
        a = AGIMSystem(workdir=tmp)
        bus = MemoryBus()
        bus.subscribe(AgentRole.VERIFIER, [AgentRole.TEACHER])
        assert len(bus.queue) == 0

def test_evolutionary():
    opt = AutoOptimizer()
    opt.record_metric("accuracy", 0.95)
    assert opt.suggest_learning_rate() > 0

def test_multi_user():
    with tempfile.TemporaryDirectory() as tmp:
        mu = MultiUserAGIM(Path(tmp))
        u1 = mu.get_user("user1")
        assert u1 is not None

def test_safety_governor():
    sg = SafetyGovernor()
    sg.observe(0.1, 1, 10, 0, 6)
    assert not sg.should_brake()

def test_recursive_loop():
    sg = SafetyGovernor()
    rl = RecursiveImprovementLoop(sg)
    result = rl.step({"accuracy": 0.9, "gate_failures": 0, "total_gates": 10})
    assert result["action"] in ("continue", "braked")

def test_cross_model():
    ct = CrossModelTransfer()
    mf = ct.abstract({"subject": "Paris", "predicate": "capital", "obj": "France"}, "llama-3-70b")
    result = ct.transfer(mf.fact_id, "gemma-4-31b")
    assert result is not None

def test_mcp_server():
    srv = MCPServer()
    resp = srv.handle_request({"method": "tools/list", "params": {}})
    assert "tools" in resp

def test_a2a():
    with tempfile.TemporaryDirectory() as tmp:
        a = AGIMSystem(workdir=tmp)
        srv = A2AServer(a)
        card = srv.get_card()
        assert "agent_card" in card

def test_plugin_marketplace():
    with tempfile.TemporaryDirectory() as tmp:
        pm = PluginMarketplace(Path(tmp))
        pid = pm.publish(Plugin(name="TestGate", description="A test verification gate", category="verification_gate"))
        assert len(pm.search("test")) == 1
        assert pm.install(pid) is not None

def test_graphql():
    with tempfile.TemporaryDirectory() as tmp:
        a = AGIMSystem(workdir=tmp)
        r = GraphQLResolver(a)
        result = r.resolve('{ ask(question: "test") }')
        assert "ask" in result

def test_universal():
    spec = AGIMSpec()
    us = UniversalMemorySubstrate(spec)
    us.register_model("model1", "edge")
    assert us.route_memory("model1", "wal_recipe")

def test_marketplace():
    mp = MemoryMarketplace()
    bundle = MemoryBundle(name="Geography Facts", description="World capitals",
                          facts=[{"question": "Capital of France?", "answer": "Paris"}])
    mp.publish(bundle)
    results = mp.search("geography")
    assert len(results) == 1
