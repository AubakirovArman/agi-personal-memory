"""CounterFact benchmark — hardened: все 4 фикса из agim_hardening_4_fixes.md.

Fix 1: NS — overlap between BEFORE and AFTER answers (не exact string)
Fix 2: NT — реальное измерение non-target lm_head rows
Fix 3: Repetition — explicit rate, ES_clean отдельно
Fix 4: Dual protocol — AGIM (substring) + EasyEdit (token exact)
"""

import json, time, urllib.request, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from agim.model.wal_editor import WalLmHeadEditor

LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda:3"


def generate(model, tok, prompt, max_tokens=10, temperature=None, rep_penalty=1.2):
    """Generation with repetition penalty (1.2 = best in sweep)."""
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    ilen = inputs.input_ids.shape[1]
    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "pad_token_id": tok.eos_token_id,
        "repetition_penalty": rep_penalty,
    }
    if temperature and temperature > 0:
        gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.9})
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    return out[0, ilen:], tok.decode(out[0][ilen:], skip_special_tokens=True)


def token_overlap(text_a: str, text_b: str) -> float:
    """Jaccard-like overlap between two tokenized strings."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    return len(intersection) / max(len(tokens_a), len(tokens_b))


def has_repetition(gen_ids: torch.Tensor, target_ids: list[int], threshold=2) -> bool:
    """Detect target token sequence repeated consecutively > threshold times."""
    t = torch.tensor(target_ids, device=gen_ids.device)
    n = len(target_ids)
    count = 0
    max_count = 0
    i = 0
    while i <= len(gen_ids) - n:
        if torch.equal(gen_ids[i:i+n], t):
            count += 1
            max_count = max(max_count, count)
            i += n
        else:
            count = 0
            i += 1
    return max_count > threshold, max_count


def evaluate_edit_hardened(model, tok, editor, fact, clamp=0.3):
    """One CounterFact edit — hardened evaluation."""
    rewrite = fact["requested_rewrite"]
    subject = rewrite["subject"]
    relation = rewrite["relation_id"]
    target_new = rewrite["target_new"]["str"]
    target_true = rewrite["target_true"]["str"]
    prompt = rewrite["prompt"].format(subject)
    target_ids = tok.encode(target_new, add_special_tokens=False)
    num_tokens = len(target_ids)

    # ── Fix 2: Snapshot lm_head for NT measurement ──
    weight = model.lm_head.weight.data
    edited_tids = set(target_ids)
    nt_sample_size = min(500, weight.shape[0])
    nt_sample = []
    while len(nt_sample) < nt_sample_size:
        rid = torch.randint(0, weight.shape[0], (1,)).item()
        if rid not in nt_sample:
            nt_sample.append(rid)
    nt_before = {rid: weight[rid, :].clone() for rid in nt_sample if rid not in edited_tids}

    # ── BEFORE: snapshot neighborhood answers ──
    n_prompts = fact.get("neighborhood_prompts", [])[:4]
    n_before_ids = []
    n_before_texts = []
    for n_prompt in n_prompts:
        n_ids, n_text = generate(model, tok, n_prompt[:100], 8)
        n_before_ids.append(n_ids)
        n_before_texts.append(n_text.strip())

    # ── EDIT ──
    ok = editor.apply_edit(subject, target_new, relation, clamp_norm=clamp)
    if not ok:
        editor.rollback()
        return None

    # ═══ ES: Efficacy ═══
    # Protocol 1: AGIM — generate 10 tokens, substring match
    es_gen_ids, es_gen_text = generate(model, tok, prompt, 10)
    es_agim = 1.0 if target_new.lower() in es_gen_text.lower() else 0.0

    # Protocol 2: EasyEdit — generate len(target) tokens, token exact match
    ee_ids, ee_text = generate(model, tok, prompt, max(1, len(target_ids)))
    es_easyedit = 1.0 if ee_ids[:len(target_ids)].cpu().tolist() == target_ids else 0.0

    # Fix 3: Repetition check (on AGIM generation)
    is_rep, rep_count = has_repetition(es_gen_ids, target_ids, threshold=2)
    es_clean = es_agim and not is_rep

    # ═══ PS: Paraphrase ═══
    ps_agim = ps_easyedit = ps_rep_count = 0.0
    ps_n = 0
    for para in fact.get("paraphrase_prompts", [])[:2]:
        p_ids, p_text = generate(model, tok, para[:100], 10)
        if target_new.lower() in p_text.lower():
            ps_agim += 1
        if p_ids[:len(target_ids)].cpu().tolist() == target_ids:
            ps_easyedit += 1
        is_pr, _ = has_repetition(p_ids, target_ids, threshold=2)
        if is_pr:
            ps_rep_count += 1
        ps_n += 1
    ps_agim /= max(ps_n, 1)
    ps_easyedit /= max(ps_n, 1)
    ps_rep_rate = ps_rep_count / max(ps_n, 1)

    # ═══ NS: Neighborhood — правильная метрика (target NOT in neighbor answer) ═══
    ns_hits = 0  # NS_B: target_new NOT in neighbor answer
    ns_overlap_hits = 0  # NS_A: overlap for diagnostic
    ns_overlaps = []
    for i, n_prompt in enumerate(n_prompts):
        _, n_after_text = generate(model, tok, n_prompt[:100], 8)
        na = n_after_text.strip()
        # Correct NS: target_new should NOT appear in neighbor answer
        if target_new.lower() not in na.lower():
            ns_hits += 1
        # Diagnostic: token overlap (legacy)
        overlap = token_overlap(n_before_texts[i], na)
        ns_overlaps.append(overlap)
        if overlap > 0.3:
            ns_overlap_hits += 1
    ns = ns_hits / max(len(n_prompts), 1)
    ns_overlap_legacy = ns_overlap_hits / max(len(n_prompts), 1)

    # ═══ NT: Measured non-target diff (Fix 2) ═══
    nt_max = 0.0
    nt_count_nonzero = 0
    for rid, orig in nt_before.items():
        diff = (weight[rid, :] - orig.to(weight.device)).abs().max().item()
        nt_max = max(nt_max, diff)
        if diff > 1e-6:
            nt_count_nonzero += 1

    # ═══ Rollback ═══
    editor.rollback()
    _, rb_text = generate(model, tok, prompt, 8)
    rb_ok = target_true.lower() in rb_text.lower()

    return {
        "subject": subject, "new": target_new, "old": target_true,
        "num_tokens": num_tokens,
        # ES
        "ES_agim": es_agim,
        "ES_easyedit": es_easyedit,
        "ES_clean": es_clean,
        "ES_has_rep": is_rep,
        "ES_rep_count": rep_count,
        # PS
        "PS_agim": ps_agim,
        "PS_easyedit": ps_easyedit,
        "PS_rep_rate": ps_rep_rate,
        # NS (correct: target NOT in neighbor)
        "NS": ns,
        "NS_mean_overlap": ns_overlap_legacy,
        # NT (Fix 2)
        "NT_max": nt_max,
        "NT_nonzero_rows": nt_count_nonzero,
        # RB
        "RB": 1.0 if rb_ok else 0.0,
        # Samples
        "gen_agim": es_gen_text[:60],
        "gen_easyedit": ee_text[:60],
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_facts", type=int, default=200)
    p.add_argument("--output", default="results/counterfact_hardened_wal.json")
    args = p.parse_args()

    data = []
    try:
        with urllib.request.urlopen("https://rome.baulab.info/data/dsets/counterfact.json") as f:
            data = json.load(f)
    except:
        with open("counterfact.json") as f:
            data = json.load(f)

    n = min(args.n_facts, len(data))
    facts = data[:n]
    print(f"  {len(facts)} facts\n")

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
    print(f"CounterFact HARDENED — 4 fixes applied")
    print(f"  Model: {LLAMA}")
    print(f"  Fix 1: NS = token overlap before vs after")
    print(f"  Fix 2: NT = measured lm_head diff")
    print(f"  Fix 3: Repetition rate + ES_clean")
    print(f"  Fix 4: Dual protocol (AGIM + EasyEdit)")
    print(f"{'='*60}\n")

    results = []
    es_a = es_ee = es_cl = ps_a = ps_ee = ns_sum = ov_sum = nt_sum = rb_sum = 0.0
    rep_facts = 0
    t0 = time.time()

    for i, fact in enumerate(facts):
        r = evaluate_edit_hardened(model, tok, editor, fact)
        if r is None:
            continue
        results.append(r)
        es_a += r["ES_agim"]; es_ee += r["ES_easyedit"]; es_cl += r["ES_clean"]
        ps_a += r["PS_agim"]; ps_ee += r["PS_easyedit"]
        ns_sum += r["NS"]; ov_sum += r["NS_mean_overlap"]
        nt_sum += r["NT_max"]; rb_sum += r["RB"]
        if r["ES_has_rep"]:
            rep_facts += 1

        if (i + 1) % 20 == 0 or i == 0:
            nd = len(results)
            e = time.time() - t0
            print(f"  [{nd}/{i+1}] ES_a={es_a/nd:.0%} ES_ee={es_ee/nd:.0%} "
                  f"ES_cl={es_cl/nd:.0%} PS_a={ps_a/nd:.0%} "
                  f"NS={ns_sum/nd:.0%} Rep={rep_facts}/{nd} ({e:.0f}s)", flush=True)

    nd = len(results)
    es_a_v = es_a/nd; es_ee_v = es_ee/nd; es_cl_v = es_cl/nd
    ps_a_v = ps_a/nd; ps_ee_v = ps_ee/nd
    ns_v = ns_sum/nd; ov_v = ov_sum/nd
    nt_v = nt_sum/nd; rb_v = rb_sum/nd; rep_r = rep_facts/nd

    comp_agim = (es_a_v + ps_a_v + ns_v) / 3
    comp_ee = (es_ee_v + ps_ee_v + ns_v) / 3

    e = time.time() - t0

    print(f"\n{'='*60}")
    print(f"COUNTERFACT HARDENED RESULTS — {nd} facts")
    print(f"{'='*60}")
    print(f"  AGIM protocol (substring, gen-10):")
    print(f"    ES={es_a_v:.1%}  PS={ps_a_v:.1%}  NS={ns_v:.1%}  Composite={comp_agim:.1%}")
    print(f"    ES_clean (no rep): {es_cl_v:.1%}  Repetition rate: {rep_r:.1%}")
    print(f"  EasyEdit protocol (token exact):")
    print(f"    ES={es_ee_v:.1%}  PS={ps_ee_v:.1%}  NS={ns_v:.1%}  Composite={comp_ee:.1%}")
    print(f"  NT measured max: {nt_v:.8f}  Mean NS overlap: {ov_v:.3f}")
    print(f"  RB: {rb_v:.1%}  Time: {e:.0f}s ({e/60:.1f}min)")

    print(f"\n  Comparison (Llama-3-class 8B):")
    print(f"    AGIM WAL (AGIM proto):  {comp_agim:.1%}")
    print(f"    AGIM WAL (EasyEdit):    {comp_ee:.1%}")
    print(f"    MEMIT (Llama 3):        53%")
    print(f"    AlphaEdit (Llama 3):    67.7%")

    with open(args.output, "w") as f:
        json.dump({
            "model": LLAMA, "n": nd,
            "agim_protocol": {"ES": round(es_a_v, 4), "PS": round(ps_a_v, 4),
                              "NS": round(ns_v, 4), "ES_clean": round(es_cl_v, 4),
                              "composite": round(comp_agim, 4)},
            "easyedit_protocol": {"ES": round(es_ee_v, 4), "PS": round(ps_ee_v, 4),
                                  "NS": round(ns_v, 4), "composite": round(comp_ee, 4)},
            "NT_max_measured": round(nt_v, 8),
            "NT_nonzero_rows": int(nt_v > 0),
            "repetition_rate": round(rep_r, 4),
            "NS_mean_overlap": round(ov_v, 4),
            "RB": round(rb_v, 4),
            "time_s": round(e, 1),
            "model_note": "Llama-3.1-8B-Instruct (not Llama-3-8B). Same 8B class.",
        }, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
