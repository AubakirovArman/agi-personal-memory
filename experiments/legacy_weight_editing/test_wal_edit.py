"""Test WAL-based weight editing on Llama 3.1 8B."""
import torch
import sys
sys.path.insert(0, "src")

from transformers import AutoModelForCausalLM, AutoTokenizer
from agim.model.wal_editor import WalLmHeadEditor

LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda:3"


def generate(model, tok, prompt, max_tokens=15):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    ilen = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ilen:], skip_special_tokens=True).strip()


def main():
    print("Loading Llama 3.1 8B...")
    tok = AutoTokenizer.from_pretrained(LLAMA)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA, torch_dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()

    # ── Build vocab ──
    print("\n1. Building WAL vocabulary...")
    editor = WalLmHeadEditor(model, tok, K=256, lmax=16, device=DEVICE)
    editor.build_vocab()
    print(f"   Atoms: {editor.atoms.shape}, range=[{editor.atoms.min():.3f}, {editor.atoms.max():.3f}]")

    # ── Baseline ──
    prompts = {
        "eiffel": "The Eiffel Tower is located in",
        "paris": "The capital of France is",
        "einstein": "Albert Einstein was born in",
        "facebook": "Facebook was created by",
    }
    before = {k: generate(model, tok, p) for k, p in prompts.items()}
    print("\n2. BEFORE edits:")
    for k, v in before.items():
        print(f"   {k}: {v[:60]}")

    # ── Test different clamp values ──
    for clamp in [0.05, 0.1, 0.2]:
        print(f"\n{'='*60}")
        print(f"Testing clamp_norm={clamp}")
        print(f"{'='*60}")

        # Fresh editor
        editor2 = WalLmHeadEditor(model, tok, K=256, lmax=16, device=DEVICE)
        editor2.atoms = editor.atoms  # reuse atoms
        editor2._vocab_size = editor._vocab_size
        editor2._hidden_size = editor._hidden_size

        # Snapshot random non-target rows for verification
        weight = model.lm_head.weight.data
        sample_tids = [torch.randint(0, 128256, (1,)).item() for _ in range(100)]
        # Make sure target tids are not in sample
        target_tids = set()
        for t in ["Rome", "Italy"]:
            for tid in tok.encode(t, add_special_tokens=False):
                target_tids.add(tid)
        sample_tids = [t for t in sample_tids if t not in target_tids][:50]
        snapshots = {t: weight[t, :].clone() for t in sample_tids}

        # Edit 1
        editor2.apply_edit("Eiffel Tower", "Rome", clamp_norm=clamp)
        after_eiffel = generate(model, tok, prompts["eiffel"])
        after_paris = generate(model, tok, prompts["paris"])

        # Edit 2
        editor2.apply_edit("Albert Einstein", "Italy", clamp_norm=clamp)
        after_einstein = generate(model, tok, prompts["einstein"])

        # Check non-target diff on UNEDITED rows
        max_diff = 0.0
        for tid in sample_tids:
            if tid not in editor2._original_rows:
                diff = (weight[tid, :] - snapshots[tid]).abs().max().item()
                max_diff = max(max_diff, diff)

        print(f"   Eiffel→Rome:  {after_eiffel[:60]}")
        print(f"   Paris (NS):   {after_paris[:60]}")
        print(f"   Einstein→Ita: {after_einstein[:60]}")
        print(f"   Non-target diff (UNEDITED rows): {max_diff:.10f}")
        print(f"   Rome in Eiffel: {'YES' if 'Rome' in after_eiffel else 'NO'}")
        print(f"   Italy in Einstein: {'YES' if 'Italy' in after_einstein else 'NO'}")

        # Rollback
        editor2.rollback()

    # ── Final verification ──
    print(f"\n{'='*60}")
    print("FINAL VERIFICATION (after rollback)")
    print(f"{'='*60}")
    for k, p in prompts.items():
        ans = generate(model, tok, p)
        ok = "OK" if ans[:30] == before[k][:30] else "CHANGED"
        print(f"   {k}: {ans[:60]} [{ok}]")


if __name__ == "__main__":
    main()
