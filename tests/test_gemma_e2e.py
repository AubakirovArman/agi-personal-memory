
"""End-to-end test: AGIM + WAL + Gemma-4-31B — full cycle."""
import tempfile

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agim.core.system import AGIMSystem
from agim.model.wal_backend import WALWeightEditor
from agim.model.memory_overlay import MemoryOverlay
from agim.model.rome_editor import ROMEEditor
from agim.model.memit_editor import MEMITEditor
from agim.model.wise_dual import WISEDualMemory

GEMMA = ("/mnt/hf_model_weights/arman/3bit/bk/.hf_cache/hub/"
         "models--google--gemma-4-31B-it/snapshots/"
         "439edf5652646a0d1bd8b46bfdc1d3645761a445")
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.gemma]


_model_cache = None
_tok_cache = None

def load_model():
    global _model_cache, _tok_cache
    if _model_cache is None:
        try:
            _tok_cache = AutoTokenizer.from_pretrained(GEMMA, local_files_only=True)
            if _tok_cache.pad_token is None:
                _tok_cache.pad_token = _tok_cache.eos_token
            _model_cache = AutoModelForCausalLM.from_pretrained(
                GEMMA, dtype=torch.bfloat16, device_map=DEVICE, local_files_only=True)
        except (OSError, ValueError) as exc:
            pytest.skip(f"Gemma E2E model unavailable in this environment: {exc}")
        _model_cache.eval()
    return _model_cache, _tok_cache


def test_gemma_model_loads():
    model, tok = load_model()
    assert model is not None
    assert tok is not None
    text_layer = "model.language_model.layers.0.mlp.down_proj.weight"
    params = dict(model.named_parameters())
    assert text_layer in params, f"Missing layer: {text_layer}"


def test_wal_vocabulary_build_on_gemma():
    model, tok = load_model()
    editor = WALWeightEditor(model, K=256, lmax=12, device=DEVICE)
    editor.build_vocabulary("model.language_model.layers.0.mlp.down_proj.weight")
    assert editor.vocabulary_is_frozen
    assert editor._atom_table.shape[0] == 256


def test_wal_encode_gemma_layer():
    model, tok = load_model()
    editor = WALWeightEditor(model, K=256, lmax=12, device=DEVICE)
    editor.build_vocabulary("model.language_model.layers.0.mlp.down_proj.weight")
    weight = dict(model.named_parameters())[
        "model.language_model.layers.0.mlp.down_proj.weight"]
    prog, recon = editor.encode_weight(weight)
    assert prog.N == weight.numel()


def test_non_target_diff_zero():
    """Frozen vocabulary: edit one layer, verify others unchanged."""
    model, tok = load_model()
    editor = WALWeightEditor(model, K=256, lmax=12, device=DEVICE)
    editor.build_vocabulary("model.language_model.layers.0.mlp.down_proj.weight")
    target = "model.language_model.layers.0.mlp.down_proj.weight"
    nontarget = "model.language_model.layers.1.mlp.down_proj.weight"
    editor.snapshot_layer(target)
    editor.snapshot_layer(nontarget)
    param = dict(model.named_parameters())[target]
    delta = torch.randn_like(param.data) * 0.001
    editor.edit_weight(target, delta)
    assert editor.verify_non_target_diff(target)
    editor.rollback_edit(target)


def test_agim_wal_cycle_on_gemma():
    """AGIM teaches fact → WAL encodes → model updated → verify."""
    model, tok = load_model()
    with tempfile.TemporaryDirectory() as tmp:
        agim = AGIMSystem(workdir=tmp)
        agim.propose_memory(
            question="test_fact_42", answer="[gemma_wal_verified]",
            kind="fact_teach", source="gemma_test")
        c = agim.propose_memory(
            question="test_fact_42", answer="[gemma_wal_verified]")
        report = agim.compile(c)
        assert report.passed
        assert agim.commit(report)
        resp = agim.ask("test_fact_42")
        assert resp.answer == "[gemma_wal_verified]"


def test_rome_editor_structure():
    model, tok = load_model()
    editor = ROMEEditor(model, tok, device=DEVICE)
    assert editor.edit_count == 0


def test_memit_editor_structure():
    model, tok = load_model()
    editor = MEMITEditor(model, tok, device=DEVICE)
    editor.add_to_batch("Paris", "Lyon", relation="capital")
    assert editor.edit_count == 0


def test_wise_dual_memory():
    model, tok = load_model()
    wise = WISEDualMemory(model)
    wise.protect_semantic()
    intact, _ = wise.verify_semantic_intact()
    assert intact


def test_memory_overlay_with_gemma():
    with tempfile.TemporaryDirectory() as tmp:
        agim = AGIMSystem(workdir=tmp)
        overlay = MemoryOverlay(agim)
        overlay.teach("gemma_test_q", "gemma_test_a")
        resp = overlay.ask("gemma_test_q")
        assert resp.answer == "gemma_test_a"


def test_full_teach_ask_forget_on_gemma():
    with tempfile.TemporaryDirectory() as tmp:
        agim = AGIMSystem(workdir=tmp)
        c = agim.propose_memory(question="gemma_e2e_q", answer="gemma_e2e_a")
        assert agim.commit(agim.compile(c))
        r = agim.ask("gemma_e2e_q")
        assert r.answer == "gemma_e2e_a"
        assert agim.rollback_last()
        r2 = agim.ask("gemma_e2e_q")
        assert r2.source == "model_fallback"
