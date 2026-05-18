"""CounterFact benchmark — hardened: все 4 фикса из agim_hardening_4_fixes.md.

Fix 1: NS — overlap between BEFORE and AFTER answers (не exact string)
Fix 2: NT — реальное измерение non-target lm_head rows
Fix 3: Repetition — explicit rate, ES_clean отдельно
Fix 4: Dual protocol — AGIM (substring) + EasyEdit (token exact)
"""


import json
import time
import urllib.request
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agim.model.wal_editor import WalLmHeadEditor
from agim.cli.counterfact_official_eval import evaluate_edit_hardened

LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda:3"

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_facts", type=int, default=200)
    p.add_argument(
        "--output",
        default="results/local_protocol/counterfact_hardened_wal.json",
    )
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

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
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
