"""Test FFN/MLP editing vs lm_head editing — specificity comparison."""
import torch
import sys
sys.path.insert(0, "src")

from transformers import AutoModelForCausalLM, AutoTokenizer

LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
DEV = "cuda:3"


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
        LLAMA, torch_dtype=torch.bfloat16, device_map=DEV)
    model.eval()

    # ── Test facts ──
    eiffel_p = "The Eiffel Tower is located in"
    paris_p = "The capital of France is"
    fb_p = "Facebook was created by"

    print(f"\nBEFORE:")
    eiffel_b = generate(model, tok, eiffel_p)
    paris_b = generate(model, tok, paris_p)
    fb_b = generate(model, tok, fb_p)
    print(f"  Eiffel: {eiffel_b[:60]}")
    print(f"  Paris:  {paris_b[:60]}")
    print(f"  FB:     {fb_b[:60]}")

    # ── Test layers: last (31), middle (15), causal trace best (4-8) ──
    for layer_idx in [31, 15, 7]:
        print(f"\n{'='*60}")
        print(f"Testing FFN edit at layer {layer_idx}")
        print(f"{'='*60}")

        # Get MLP key-value
        mlp = model.model.layers[layer_idx].mlp
        inputs = tok(eiffel_p, return_tensors="pt").to(model.device)

        m_val = None
        v_val = None

        def hook_down(module, inp, out):
            nonlocal m_val, v_val
            m_val = inp[0].detach().clone()  # input to down_proj = act(gate)*up
            v_val = out.detach().clone()      # output of down_proj

        h = mlp.down_proj.register_forward_hook(hook_down)
        with torch.no_grad():
            model(**inputs)
        h.remove()

        m_last = m_val[0, -1, :].float()       # key at last position
        v_curr = v_val[0, -1, :].float()       # current value
        print(f"  m: norm={m_last.norm():.3f}, v_curr: norm={v_curr.norm():.3f}")

        # Target direction: align with lm_head[target_token]
        target_ids = tok.encode("Rome", add_special_tokens=False)
        print(f"  Rome tokens: {target_ids} = {[tok.decode([t]) for t in target_ids]}")
        target_dir = torch.zeros(4096, device=model.device)
        for i, tid in enumerate(target_ids):
            target_dir += model.lm_head.weight.data[tid, :].float() / (2 ** i)
        target_dir = target_dir / (target_dir.norm() + 1e-8)

        # Try different update strengths
        for alpha in [0.2, 0.5, 1.0, 2.0]:
            # Save original
            W_orig = mlp.down_proj.weight.data.clone()

            # delta_v = alpha * target_dir  (push MLP output toward target direction)
            delta_v = alpha * target_dir.to(DEV)

            # Rank-1 update: W += delta_v @ m^T (unnormalized, scaled by alpha)
            update = torch.outer(delta_v, m_last)
            mlp.down_proj.weight.data += update.to(
                dtype=mlp.down_proj.weight.dtype, device=DEV)

            eiffel_a = generate(model, tok, eiffel_p)
            paris_a = generate(model, tok, paris_p)
            fb_a = generate(model, tok, fb_p)

            rome_in_eiffel = "Rome" in eiffel_a or "rome" in eiffel_a.lower()
            rome_in_paris = "Rome" in paris_a or "rome" in paris_a.lower()
            paris_ok = "Paris" in paris_a

            print(f"  alpha={alpha:.1f}: Eiffel={eiffel_a[:40]} | Paris={paris_a[:40]} | "
                  f"Rome✓={rome_in_eiffel} Paris✓={paris_ok} Cross={rome_in_paris}")

            # Rollback
            mlp.down_proj.weight.data = W_orig

    print(f"\n{'='*60}")
    print("COMPARISON: lm_head edit (same approach)")
    print(f"{'='*60}")
    # Direct lm_head edit for comparison
    W_lm_orig = model.lm_head.weight.data.clone()
    target_ids = tok.encode("Rome", add_special_tokens=False)
    for alpha in [0.1, 0.3, 0.5]:
        # Get last hidden
        inputs2 = tok(eiffel_p, return_tensors="pt").to(model.device)
        last_h = None

        def hook_norm(module, inp, out):
            nonlocal last_h
            hs = out[0] if isinstance(out, tuple) else out
            last_h = hs.detach().clone()

        h2 = model.model.norm.register_forward_hook(hook_norm)
        with torch.no_grad():
            model(**inputs2)
        h2.remove()

        key = last_h[0, -1, :].float()
        key = key / (key.norm() + 1e-8)

        for i, tid in enumerate(target_ids):
            boost = alpha * key / (2 ** i)
            model.lm_head.weight.data[tid, :] += boost.to(
                dtype=model.lm_head.weight.dtype, device=DEV)

        eiffel_a = generate(model, tok, eiffel_p)
        paris_a = generate(model, tok, paris_p)
        rome_in_eiffel = "Rome" in eiffel_a
        rome_in_paris = "Rome" in paris_a
        print(f"  alpha={alpha:.1f}: Eiffel={eiffel_a[:40]} | Paris={paris_a[:40]} | "
              f"Rome✓={rome_in_eiffel} Cross={rome_in_paris}")

        # Rollback
        model.lm_head.weight.data = W_lm_orig


if __name__ == "__main__":
    main()
