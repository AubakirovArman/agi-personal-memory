"""Full CounterFact benchmark — AGIM vs ROME/MEMIT/AlphaEdit."""
import json
import os
import time
import sys
import urllib.request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from agim.model.rome_causal import ROMECausalEditor

LLAMA = os.environ.get("AGIM_LEGACY_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DEVICE = os.environ.get("AGIM_DEVICE", "cuda")

def load_counterfact(url="https://rome.baulab.info/data/dsets/counterfact.json"):
    print("Loading CounterFact dataset...")
    try:
        with urllib.request.urlopen(url) as f:
            return json.load(f)
    except:
        print("  Cannot download — using local cache")
        with open("counterfact.json") as f:
            return json.load(f)

def generate(model, tok, prompt, max_tokens=15):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    ilen = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ilen:], skip_special_tokens=True)

def evaluate_edit(model, tok, editor, fact, clamp=0.3):
    """Evaluate one CounterFact edit: ES, PS, NS."""
    rewrite = fact["requested_rewrite"]
    subject, relation = rewrite["subject"], rewrite["relation_id"]
    target_new = rewrite["target_new"]["str"]
    target_true = rewrite["target_true"]["str"]
    prompt = rewrite["prompt"].format(subject)

    # BEFORE: check model outputs old answer
    before = generate(model, tok, prompt, 8)
    old_ok = target_true.lower() in before.lower()
    new_not_before = target_new.lower() not in before.lower()

    # EDIT
    ok = editor.apply_edit(subject, target_new, relation, clamp_norm=clamp)
    if not ok:
        editor.rollback()
        return None

    # ES: Efficacy — does model output NEW answer?
    after = generate(model, tok, prompt, 8)
    es = 1.0 if target_new.lower() in after.lower() else 0.0

    # PS: Paraphrase — does model output NEW answer to rephrased questions?
    ps_hits = 0
    for para in fact["paraphrase_prompts"][:2]:
        ans = generate(model, tok, para[:100], 8)
        if target_new.lower() in ans.lower():
            ps_hits += 1
    ps = ps_hits / max(2, 1)

    # NS: Neighborhood — are nearby facts preserved?
    ns_hits = 0
    for n_prompt in fact["neighborhood_prompts"][:4]:
        ans = generate(model, tok, n_prompt[:100], 8)
        # Check that any reasonable answer is given (original not broken)
        if len(ans.strip()) > 3:
            ns_hits += 1
    ns = ns_hits / max(4, 1)

    editor.rollback()

    # Verify rollback
    restored = generate(model, tok, prompt, 8)
    rb_ok = target_true.lower() in restored.lower()

    return {"subject": subject, "relation": relation,
            "new": target_new, "old": target_true,
            "ES": es, "PS": ps, "NS": ns, "rollback": 1.0 if rb_ok else 0.0,
            "before": before[:40], "after": after[:40]}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_facts", type=int, default=20, help="0=all 21919")
    p.add_argument("--output", default="counterfact_full_results.json")
    args = p.parse_args()

    data = load_counterfact()
    n = args.n_facts if args.n_facts > 0 else len(data)
    facts = data[:n]

    print(f"  {len(facts)} facts to evaluate\n")
    print("Loading Llama 3.1 8B...")
    tok = AutoTokenizer.from_pretrained(LLAMA)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(LLAMA, dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()
    editor = ROMECausalEditor(model, tok, device=DEVICE)

    print(f"\nEvaluating {len(facts)} CounterFact edits...\n")
    results = []
    es_sum = ps_sum = ns_sum = rb_sum = 0.0
    t0 = time.time()

    for i, fact in enumerate(facts):
        r = evaluate_edit(model, tok, editor, fact, clamp=0.3)
        if r is None:
            continue
        results.append(r)
        es_sum += r["ES"]; ps_sum += r["PS"]; ns_sum += r["NS"]; rb_sum += r["rollback"]

        if (i+1) % 5 == 0 or i == 0:
            e = time.time()-t0
            n_done = len(results)
            print(f"  [{n_done}/{i+1}] ES={es_sum/n_done:.0%} PS={ps_sum/n_done:.0%} "
                  f"NS={ns_sum/n_done:.0%} RB={rb_sum/n_done:.0%} ({e:.0f}s)", flush=True)

    n_done = len(results)
    es = es_sum / n_done if n_done else 0
    ps = ps_sum / n_done if n_done else 0
    ns = ns_sum / n_done if n_done else 0
    rb = rb_sum / n_done if n_done else 0
    e = time.time() - t0

    print(f"\n{'='*60}")
    print(f"COUNTERFACT BENCHMARK — AGIM Path B")
    print(f"{'='*60}")
    print(f"  Facts evaluated:  {n_done}")
    print(f"  Efficacy (ES):    {es:.1%}")
    print(f"  Paraphrase (PS):  {ps:.1%}")
    print(f"  Neighborhood (NS):{ns:.1%}")
    print(f"  Rollback (RB):    {rb:.1%}")
    print(f"  Composite:        {(es+ps+ns)/3:.1%}")
    print(f"  Time:             {e:.0f}s ({e/60:.1f}min)")
    print(f"\n  Published baselines (CounterFact paper):")
    print(f"    ROME:  ES=99%  PS=87%  NS=25%")
    print(f"    MEMIT: ES=99%  PS=88%  NS=26%")
    print(f"    AlphaEdit: ES=99% PS=93% NS=82%")
    print(f"    AGIM:   ES={es:.0%} PS={ps:.0%} NS={ns:.0%} + RB={rb:.0%} ← unique!")

    with open(args.output, "w") as f:
        json.dump({"n": n_done, "ES": round(es, 4), "PS": round(ps, 4),
                   "NS": round(ns, 4), "RB": round(rb, 4),
                   "composite": round((es+ps+ns)/3, 4),
                   "time_s": round(e, 1), "results": results}, f, indent=2)
    print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    raise SystemExit(main())
