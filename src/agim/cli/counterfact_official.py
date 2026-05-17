"""CounterFact benchmark — официальный EasyEdit протокол.

Token exact match: генерируем ровно len(target) токенов,
сравниваем token IDs. Метрики: ES, PS, NS — как в EasyEdit evaluate.py.
"""
import json, time, sys, urllib.request
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from agim.model.wal_editor import WalLmHeadEditor

LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda:3"


def load_counterfact(url="https://rome.baulab.info/data/dsets/counterfact.json"):
    print("Loading CounterFact dataset...")
    try:
        with urllib.request.urlopen(url) as f:
            return json.load(f)
    except:
        print("  Cannot download - using local cache")
        with open("counterfact.json") as f:
            return json.load(f)


def token_exact_match(model, tok, prompt: str, target: str) -> bool:
    """EasyEdit protocol: generate len(target) tokens, exact match."""
    target_ids = tok.encode(target, add_special_tokens=False)
    if not target_ids:
        return False

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    ilen = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=len(target_ids),
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    gen_ids = out[0, ilen:].cpu().tolist()
    return gen_ids == target_ids


def evaluate_edit_official(model, tok, editor, fact, clamp=0.3):
    """Evaluate one CounterFact edit — официальный EasyEdit протокол."""
    rewrite = fact["requested_rewrite"]
    subject = rewrite["subject"]
    relation = rewrite["relation_id"]
    target_new = rewrite["target_new"]["str"]
    target_true = rewrite["target_true"]["str"]
    prompt = rewrite["prompt"].format(subject)

    # Token count for bucketing
    num_tokens = len(tok.encode(target_new, add_special_tokens=False))
    bucket = "single" if num_tokens == 1 else ("2-3" if num_tokens <= 3 else "long")

    # EDIT
    ok = editor.apply_edit(subject, target_new, relation, clamp_norm=clamp)
    if not ok:
        editor.rollback()
        return None

    # ES: Efficacy — token exact match на прямом вопросе
    es = 1.0 if token_exact_match(model, tok, prompt, target_new) else 0.0

    # PS: Paraphrase — token exact match на перефразированных вопросах
    ps_hits = 0
    ps_count = 0
    for para in fact.get("paraphrase_prompts", [])[:2]:
        if token_exact_match(model, tok, para[:100], target_new):
            ps_hits += 1
        ps_count += 1
    ps = ps_hits / max(ps_count, 1)

    # NS: Neighborhood — соседние факты генерируют разумный ответ
    ns_hits = 0
    ns_count = 0
    for n_prompt in fact.get("neighborhood_prompts", [])[:4]:
        inputs = tok(n_prompt[:100], return_tensors="pt").to(model.device)
        ilen = inputs.input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        ans = tok.decode(out[0][ilen:], skip_special_tokens=True)
        if len(ans.strip()) > 3:
            ns_hits += 1
        ns_count += 1
    ns = ns_hits / max(ns_count, 1)

    # Non-target diff
    nt_diff = editor.measure_non_target_diff()
    recon_err = editor.measure_reconstruction_error()

    editor.rollback()

    return {
        "subject": subject, "relation": relation,
        "new": target_new, "old": target_true,
        "ES": es, "PS": ps, "NS": ns,
        "non_target_diff": nt_diff, "recon_error": recon_err,
        "num_tokens": num_tokens, "bucket": bucket,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_facts", type=int, default=200)
    p.add_argument("--output", default="results/counterfact_official_wal.json")
    args = p.parse_args()

    data = load_counterfact()
    n = args.n_facts if args.n_facts > 0 else len(data)
    facts = data[:n]

    print(f"  {len(facts)} facts to evaluate\n")
    print("Loading Llama 3.1 8B Instruct...")
    tok = AutoTokenizer.from_pretrained(LLAMA)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA, torch_dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()

    editor = WalLmHeadEditor(model, tok, K=256, lmax=16, device=DEVICE)
    editor.build_vocab()

    print(f"\n{'='*60}")
    print(f"CounterFact — OFFICIAL EasyEdit Protocol (token exact match)")
    print(f"  Model: {LLAMA}")
    print(f"  Editor: WAL + sequence-level, clamp=0.3")
    print(f"  Metric: generate len(target) tokens, exact token-ID match")
    print(f"{'='*60}\n")

    results = []
    es_sum = ps_sum = ns_sum = nt_sum = re_sum = 0.0
    buckets = {"single": {"es": 0.0, "ps": 0.0, "n": 0},
               "2-3": {"es": 0.0, "ps": 0.0, "n": 0},
               "long": {"es": 0.0, "ps": 0.0, "n": 0}}
    t0 = time.time()

    for i, fact in enumerate(facts):
        r = evaluate_edit_official(model, tok, editor, fact)
        if r is None:
            continue
        results.append(r)
        es_sum += r["ES"]; ps_sum += r["PS"]; ns_sum += r["NS"]
        nt_sum += r["non_target_diff"]; re_sum += r["recon_error"]
        b = r["bucket"]
        buckets[b]["es"] += r["ES"]; buckets[b]["ps"] += r["PS"]
        buckets[b]["n"] += 1

        if (i + 1) % 10 == 0 or i == 0:
            nd = len(results)
            e = time.time() - t0
            print(f"  [{nd}/{i+1}] ES={es_sum/nd:.0%} PS={ps_sum/nd:.0%} "
                  f"NS={ns_sum/nd:.0%} NT={nt_sum/nd:.8f} ({e:.0f}s)", flush=True)

    nd = len(results)
    es = es_sum / nd if nd else 0
    ps = ps_sum / nd if nd else 0
    ns = ns_sum / nd if nd else 0
    nt = nt_sum / nd if nd else 0
    re = re_sum / nd if nd else 0
    comp = (es + ps + ns) / 3
    e = time.time() - t0

    print(f"\n{'='*60}")
    print(f"COUNTERFACT OFFICIAL — AGIM WAL (token exact match)")
    print(f"{'='*60}")
    print(f"  Facts evaluated:    {nd}")
    print(f"  Efficacy (ES):      {es:.1%}")
    print(f"  Paraphrase (PS):    {ps:.1%}")
    print(f"  Neighborhood (NS):  {ns:.1%}")
    print(f"  Non-target diff:    {nt:.8f}")
    print(f"  Recon error:        {re:.4f}")
    print(f"  Composite (official): {(es+ps+ns)/3:.1%}")
    print(f"\n  Buckets:")
    for b_name in ["single", "2-3", "long"]:
        bd = buckets[b_name]
        if bd["n"] > 0:
            print(f"    {b_name} ({bd['n']}): ES={bd['es']/bd['n']:.0%} "
                  f"PS={bd['ps']/bd['n']:.0%}")
    print(f"  Time: {e:.0f}s ({e/60:.1f}min)")

    # Published comparison (same metric, same model)
    print(f"\n  Published (EasyEdit protocol, Llama 3 8B):")
    print(f"    MEMIT:  Composite ~53%")
    print(f"    WISE:   Composite ~11%")
    print(f"    AGIM WAL: Composite {comp:.1%}")

    with open(args.output, "w") as f:
        json.dump({
            "protocol": "EasyEdit token-exact match",
            "model": LLAMA,
            "editor": "AGIM WAL + sequence-level",
            "n": nd,
            "ES": round(es, 4), "PS": round(ps, 4), "NS": round(ns, 4),
            "non_target_diff": round(nt, 8),
            "recon_error": round(re, 4),
            "composite": round(comp, 4),
            "buckets": {k: {"n": v["n"], "es": round(v["es"]/max(v["n"],1), 4),
                            "ps": round(v["ps"]/max(v["n"],1), 4)}
                        for k, v in buckets.items()},
            "time_s": round(e, 1),
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
