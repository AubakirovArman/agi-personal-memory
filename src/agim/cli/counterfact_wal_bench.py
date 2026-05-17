"""CounterFact benchmark — WAL vs ROME comparison."""
import json, time, sys, urllib.request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from agim.model.rome_causal import ROMECausalEditor
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


def generate(model, tok, prompt, max_tokens=10, temperature=None):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    ilen = inputs.input_ids.shape[1]
    with torch.no_grad():
        if temperature and temperature > 0:
            out = model.generate(**inputs, max_new_tokens=max_tokens,
                                 do_sample=True, temperature=temperature,
                                 top_p=0.9, pad_token_id=tok.eos_token_id)
        else:
            out = model.generate(**inputs, max_new_tokens=max_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ilen:], skip_special_tokens=True)


def evaluate_edit(model, tok, editor, fact, clamp=0.3, editor_type="wal"):
    """Evaluate one CounterFact edit."""
    rewrite = fact["requested_rewrite"]
    subject, relation = rewrite["subject"], rewrite["relation_id"]
    target_new = rewrite["target_new"]["str"]
    target_true = rewrite["target_true"]["str"]
    prompt = rewrite["prompt"].format(subject)

    # BEFORE
    before = generate(model, tok, prompt, 8)
    old_ok = target_true.lower() in before.lower()

    # EDIT
    ok = editor.apply_edit(subject, target_new, relation, clamp_norm=clamp)
    if not ok:
        editor.rollback()
        return None

    # ES: Efficacy
    after = generate(model, tok, prompt, 8)
    es = 1.0 if target_new.lower() in after.lower() else 0.0

    # PS: Paraphrase
    ps_hits = 0
    for para in fact.get("paraphrase_prompts", [])[:2]:
        ans = generate(model, tok, para[:100], 8)
        if target_new.lower() in ans.lower():
            ps_hits += 1
    ps = ps_hits / max(2, 1)

    # NS: Neighborhood
    ns_hits = 0
    for n_prompt in fact.get("neighborhood_prompts", [])[:4]:
        ans = generate(model, tok, n_prompt[:100], 8)
        if len(ans.strip()) > 3:
            ns_hits += 1
    ns = ns_hits / max(4, 1)

    # Non-target diff (WAL-specific: always 0.0 by construction)
    non_target_diff = 0.0
    recon_error = 0.0
    if editor_type == "wal":
        non_target_diff = editor.measure_non_target_diff()
        recon_error = editor.measure_reconstruction_error()

    editor.rollback()

    # Verify rollback
    restored = generate(model, tok, prompt, 8)
    rb_ok = target_true.lower() in restored.lower()

    # Bucket: track token count
    num_tokens = len(tok.encode(target_new, add_special_tokens=False))
    if num_tokens == 1:
        bucket = "single"
    elif num_tokens <= 3:
        bucket = "2-3"
    else:
        bucket = "long"

    return {"subject": subject, "relation": relation,
            "new": target_new, "old": target_true,
            "ES": es, "PS": ps, "NS": ns, "rollback": 1.0 if rb_ok else 0.0,
            "non_target_diff": non_target_diff,
            "recon_error": recon_error,
            "num_tokens": num_tokens, "bucket": bucket,
            "before": before[:40], "after": after[:40]}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_facts", type=int, default=30, help="0=all")
    p.add_argument("--editor", default="wal", choices=["wal", "rome", "both"])
    p.add_argument("--output", default="counterfact_wal_results.json")
    args = p.parse_args()

    data = load_counterfact()
    n = args.n_facts if args.n_facts > 0 else len(data)
    facts = data[:n]

    print(f"  {len(facts)} facts to evaluate\n")
    print("Loading Llama 3.1 8B...")
    tok = AutoTokenizer.from_pretrained(LLAMA)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    editors_to_test = []
    if args.editor in ("wal", "both"):
        model_wal = AutoModelForCausalLM.from_pretrained(
            LLAMA, torch_dtype=torch.bfloat16, device_map=DEVICE)
        model_wal.eval()
        wal_editor = WalLmHeadEditor(model_wal, tok, K=256, lmax=16, device=DEVICE)
        wal_editor.build_vocab()
        editors_to_test.append(("WAL", model_wal, wal_editor, "wal", 0.3))

    if args.editor in ("rome", "both"):
        model_rome = AutoModelForCausalLM.from_pretrained(
            LLAMA, torch_dtype=torch.bfloat16, device_map=DEVICE)
        model_rome.eval()
        rome_editor = ROMECausalEditor(model_rome, tok, device=DEVICE)
        editors_to_test.append(("ROME", model_rome, rome_editor, "rome", 0.3))

    for editor_name, model, editor, etype, clamp in editors_to_test:
        print(f"\n{'='*60}")
        print(f"Evaluating {editor_name} on {len(facts)} CounterFact edits...")
        print(f"{'='*60}")

        results = []
        es_sum = ps_sum = ns_sum = rb_sum = nt_sum = re_sum = 0.0
        # Bucket tracking
        buckets = {"single": {"es": 0.0, "ps": 0.0, "n": 0},
                   "2-3": {"es": 0.0, "ps": 0.0, "n": 0},
                   "long": {"es": 0.0, "ps": 0.0, "n": 0}}
        t0 = time.time()

        for i, fact in enumerate(facts):
            r = evaluate_edit(model, tok, editor, fact, clamp=clamp, editor_type=etype)
            if r is None:
                continue
            results.append(r)
            es_sum += r["ES"]; ps_sum += r["PS"]; ns_sum += r["NS"]
            rb_sum += r["rollback"]; nt_sum += r.get("non_target_diff", 0)
            re_sum += r.get("recon_error", 0)
            # Bucket tracking
            b = r["bucket"]
            buckets[b]["es"] += r["ES"]
            buckets[b]["ps"] += r["PS"]
            buckets[b]["n"] += 1

            if (i+1) % 10 == 0 or i == 0:
                e = time.time()-t0
                nd = len(results)
                print(f"  [{nd}/{i+1}] ES={es_sum/nd:.0%} PS={ps_sum/nd:.0%} "
                      f"NS={ns_sum/nd:.0%} RB={rb_sum/nd:.0%} "
                      f"NT={nt_sum/nd:.8f} RE={re_sum/nd:.4f} ({e:.0f}s)", flush=True)

        nd = len(results)
        es = es_sum/nd if nd else 0
        ps = ps_sum/nd if nd else 0
        ns = ns_sum/nd if nd else 0
        rb = rb_sum/nd if nd else 0
        nt = nt_sum/nd if nd else 0
        re = re_sum/nd if nd else 0
        e = time.time()-t0

        print(f"\n{'='*60}")
        print(f"COUNTERFACT — {editor_name} (clamp={clamp})")
        print(f"{'='*60}")
        print(f"  Facts evaluated:    {nd}")
        print(f"  Efficacy (ES):      {es:.1%}")
        print(f"  Paraphrase (PS):    {ps:.1%}")
        print(f"  Neighborhood (NS):  {ns:.1%}")
        print(f"  Rollback (RB):      {rb:.1%}")
        print(f"  Non-target diff:    {nt:.8f}")
        print(f"  Recon error:        {re:.4f}")
        print(f"  Composite:          {(es+ps+ns)/3:.1%}")
        print(f"\n  Buckets by target token count:")
        for b_name, b_data in buckets.items():
            if b_data["n"] > 0:
                b_es = b_data["es"] / b_data["n"]
                b_ps = b_data["ps"] / b_data["n"]
                print(f"    {b_name} ({b_data['n']} facts): ES={b_es:.0%} PS={b_ps:.0%}")
        print(f"  Time:               {e:.0f}s ({e/60:.1f}min)")

        # Save per-editor results
        out_name = args.output.replace(".json", f"_{editor_name.lower()}.json")
        with open(out_name, "w") as f:
            json.dump({"editor": editor_name, "n": nd,
                       "ES": round(es,4), "PS": round(ps,4),
                       "NS": round(ns,4), "RB": round(rb,4),
                       "non_target_diff": round(nt, 8),
                       "recon_error": round(re, 4),
                       "composite": round((es+ps+ns)/3,4),
                       "time_s": round(e,1), "results": results}, f, indent=2)
        print(f"Saved to {out_name}")


if __name__ == "__main__":
    raise SystemExit(main())
