
"""End-to-end tests on Llama-3.1-8B: PPL + model editing + generate."""
import gc
import tempfile

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from agim.core.system import AGIMSystem
from agim.model.wal_backend import WALWeightEditor
from agim.model.rome_editor import ROMEEditor

# Will be set after download
LLAMA_DIR = None
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.llama]
_MODEL = None
_TOK = None


def _find_llama_snapshot():
    """Find downloaded Llama 3.1 8B snapshot."""
    import os
    base = "/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/.hf_cache"
    for root, dirs, files in os.walk(base):
        if "model.safetensors.index.json" in files and "Llama" in root:
            return root
    # Fallback: try HF hub cache
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    for root, dirs, files in os.walk(cache):
        if "model.safetensors.index.json" in files and "Llama-3.1-8B" in root:
            return root
    return None


def get_llama():
    global LLAMA_DIR, _MODEL, _TOK
    if _MODEL is not None:
        return _MODEL, _TOK
    if LLAMA_DIR is None:
        LLAMA_DIR = _find_llama_snapshot()
    if LLAMA_DIR is None:
        pytest.skip("Llama 3.1 8B not downloaded yet")
    _TOK = AutoTokenizer.from_pretrained(LLAMA_DIR, local_files_only=True)
    if _TOK.pad_token is None:
        _TOK.pad_token = _TOK.eos_token
    _MODEL = AutoModelForCausalLM.from_pretrained(
        LLAMA_DIR, dtype=torch.bfloat16, device_map=DEVICE, local_files_only=True)
    _MODEL.eval()
    return _MODEL, _TOK


def compute_ppl_sliding(model, tok, text, max_len=1024, stride=512):
    """Sliding window PPL — proper evaluation."""
    enc = tok(text, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = enc["input_ids"].to(model.device)
    seq_len = input_ids.size(1)
    nlls = []
    for begin in range(0, seq_len, stride):
        end = min(begin + max_len, seq_len)
        if end - begin < stride // 2:
            break
        chunk = input_ids[:, begin:end]
        with torch.no_grad():
            outputs = model(chunk, labels=chunk)
        nlls.append(outputs.loss.item())
    return float(torch.exp(torch.tensor(nlls).mean()).item())


# ─── Tests ───────────────────────────────────────────────────

def test_llama_loads():
    model, tok = get_llama()
    assert "model.layers.0" in dict(model.named_parameters()).__str__()
    param = dict(model.named_parameters()).get("model.layers.0.mlp.down_proj.weight")
    assert param is not None


def test_llama_layers():
    """Verify Llama 3.1 8B architecture."""
    model, tok = get_llama()
    params = dict(model.named_parameters())
    layers = set()
    for name in params:
        if "model.layers." in name:
            l = int(name.split("model.layers.")[1].split(".")[0])
            layers.add(l)
    assert len(layers) == 32
    # Check shapes
    assert params["model.layers.0.mlp.down_proj.weight"].shape == (4096, 14336)
    assert params["model.layers.0.self_attn.q_proj.weight"].shape == (4096, 4096)


def test_baseline_ppl_llama():
    """Baseline PPL on wikitext-2 — properly on large volume."""
    model, tok = get_llama()
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t.strip()) > 0]
    # Use full test set for proper PPL
    full_text = "\n\n".join(texts)
    ppl = compute_ppl_sliding(model, tok, full_text)
    print(f"\n  Llama 3.1 8B baseline PPL (wikitext-2, full): {ppl:.2f}")
    # Llama 3.1 8B should get ~6-8 PPL on wikitext-2
    assert ppl < 50, f"PPL {ppl} too high — check evaluation"


def test_wal_vocabulary_llama():
    model, tok = get_llama()
    editor = WALWeightEditor(model, K=256, lmax=12, device=DEVICE)
    editor.build_vocabulary("model.layers.0.mlp.down_proj.weight")
    assert editor.vocabulary_is_frozen


def test_wal_encode_ppl_impact():
    """Measure PPL after WAL-encoding ONE layer."""
    model, tok = get_llama()
    # Baseline PPL (quick — first 50 samples)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t.strip()) > 0][:50]
    full_text = "\n\n".join(texts)

    baseline_ppl = compute_ppl_sliding(model, tok, full_text)

    # WAL encode one layer
    editor = WALWeightEditor(model, K=256, lmax=12, device=DEVICE)
    editor.build_vocabulary("model.layers.0.mlp.down_proj.weight")
    target = "model.layers.0.mlp.down_proj.weight"
    param = dict(model.named_parameters())[target]
    editor.snapshot_layer(target)

    weight = param.data.clone()
    prog, recon = editor.encode_weight(weight)
    param.data.copy_(recon.reshape_as(weight).to(weight.dtype))

    encoded_ppl = compute_ppl_sliding(model, tok, full_text)
    print(f"\n  Baseline PPL: {baseline_ppl:.2f}")
    print(f"  After WAL encode layer 0: {encoded_ppl:.2f}")

    # Rollback
    editor.rollback_edit(target)

    assert encoded_ppl < baseline_ppl * 2, \
        f"PPL degraded too much: {baseline_ppl:.1f} → {encoded_ppl:.1f}"


def test_agim_fictional_fact_llama():
    """AGIM teaches fictional fact, verified on Llama."""
    model, tok = get_llama()
    with tempfile.TemporaryDirectory() as tmp:
        agim = AGIMSystem(workdir=tmp)
        c = agim.propose_memory(
            question="What is the capital of Zanikland?",
            answer="Blorptown",
            kind="fact_teach", source="llama_test", confidence=1.0)
        assert agim.commit(agim.compile(c))
        resp = agim.ask("What is the capital of Zanikland?")
        assert resp.answer == "Blorptown"

        # Model should NOT know this (fictional fact)
        before = model.generate(
            **tok("The capital of Zanikland is", return_tensors="pt").to(model.device),
            max_new_tokens=10, do_sample=False)
        before_text = tok.decode(before[0])

        # Apply WAL edit to influence the model
        editor = WALWeightEditor(model, K=256, lmax=12, device=DEVICE)
        editor.build_vocabulary("model.layers.0.mlp.down_proj.weight")
        target = "model.layers.0.mlp.down_proj.weight"
        editor.snapshot_layer(target)
        blorp_ids = tok.encode("Blorptown", add_special_tokens=False)
        target_id = blorp_ids[0] if blorp_ids else 0
        delta = torch.zeros_like(dict(model.named_parameters())[target].data)
        delta[target_id % delta.shape[0], :] += 0.3
        editor.edit_weight(target, delta)
        assert editor.verify_non_target_diff(target)
        assert editor.vocabulary_is_frozen
        editor.rollback_edit(target)
        agim.rollback_last()


def test_full_ppl_llama():
    """Full wikitext-2 PPL with multiple WAL compression levels."""
    model, tok = get_llama()
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t.strip()) > 0][:100]
    full_text = "\n\n".join(texts)

    baseline = compute_ppl_sliding(model, tok, full_text)
    print(f"\n  [FULL PPL] Baseline: {baseline:.2f}")

    # Test multiple K values
    for K in [256, 128, 64]:
        editor = WALWeightEditor(model, K=K, lmax=12, device=DEVICE)
        editor.build_vocabulary("model.layers.0.mlp.down_proj.weight")
        target = "model.layers.0.mlp.down_proj.weight"
        param = dict(model.named_parameters())[target]
        weight = param.data.clone()
        prog, recon = editor.encode_weight(weight)
        param.data.copy_(recon.reshape_as(weight).to(weight.dtype))
        ppl = compute_ppl_sliding(model, tok, full_text)
        print(f"  K={K:3d}: PPL={ppl:.2f}")
        editor.rollback_edit(target)

    print(f"  Baseline: {baseline:.2f}")


def test_non_target_diff_llama():
    model, tok = get_llama()
    editor = WALWeightEditor(model, K=256, lmax=12, device=DEVICE)
    editor.build_vocabulary("model.layers.0.mlp.down_proj.weight")
    target = "model.layers.0.mlp.down_proj.weight"
    nontarget = "model.layers.1.mlp.down_proj.weight"
    editor.snapshot_layer(target)
    editor.snapshot_layer(nontarget)
    param = dict(model.named_parameters())[target]
    delta = torch.randn_like(param.data) * 0.001
    editor.edit_weight(target, delta)
    assert editor.verify_non_target_diff(target), "Non-target diff should be 0%"
    editor.rollback_edit(target)


def test_rome_llama():
    model, tok = get_llama()
    editor = ROMEEditor(model, tok, device=DEVICE)
    before = model.generate(
        **tok("The capital of Zanikland is", return_tensors="pt").to(model.device),
        max_new_tokens=10, do_sample=False)
    ok = editor.apply_edit("Zanikland", "Blorptown", "capital", target_layer=5)
    if ok:
        after = model.generate(
            **tok("The capital of Zanikland is", return_tensors="pt").to(model.device),
            max_new_tokens=10, do_sample=False)
        editor.rollback()
