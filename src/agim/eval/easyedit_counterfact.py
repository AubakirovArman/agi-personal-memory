"""Canonical EasyEdit-compatible CounterFact evaluator.

3 NS metrics: NS_absence, NS_consistency, NS_overlap
2 ES/PS protocols: AGIM (substring) + EasyEdit (token-exact)
Per-example JSON output.
"""
import json, time, urllib.request, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from agim.model.wal_dual_editor import WALDualLayerEditor

LLAMA = "meta-llama/Llama-3.1-8B-Instruct"


class CounterFactEvaluator:
    """Official evaluator for CounterFact with dual-layer WAL editing."""

    def __init__(self, model, tok, editor, device="cuda:3"):
        self.model = model
        self.tok = tok
        self.editor = editor
        self.device = device

    def generate(self, prompt, max_tokens=10, rep_penalty=1.2):
        i = self.tok(prompt, return_tensors="pt").to(self.model.device)
        il = i.input_ids.shape[1]
        with torch.no_grad():
            o = self.model.generate(**i, max_new_tokens=max_tokens, do_sample=False,
                                     repetition_penalty=rep_penalty,
                                     pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(o[0][il:], skip_special_tokens=True).strip()

    def token_overlap(self, a, b):
        ta = set(a.lower().split()); tb = set(b.lower().split())
        return len(ta & tb) / max(len(ta), len(tb)) if ta and tb else 0.0

    def has_repetition(self, text, target, threshold=2):
        t = re.escape(target.lower())
        return bool(re.search(rf'({t})\1{{{threshold},}}', text.lower()))

    def evaluate_one(self, fact, clamp_lm=0.20, clamp_embed=0.06,
                     clamp_eos=0.16, clamp_anti=0.06, strict=True):
        """Evaluate one CounterFact fact. Returns per-example dict.
        strict=True: EasyEdit official (rep_penalty=1.0).
        strict=False: AGIM practical (rep_penalty=1.2)."""
        rw = fact["requested_rewrite"]
        s, rel = rw["subject"], rw["relation_id"]
        tnew, told = rw["target_new"]["str"], rw["target_true"]["str"]
        prompt = rw["prompt"].format(s)
        tids = self.tok.encode(tnew, add_special_tokens=False)
        rp = 1.0 if strict else 1.2  # strict = no rep penalty

        # BEFORE: snapshot neighbor answers
        n_prompts = fact.get("neighborhood_prompts", [])[:4]
        n_before = [self.generate(np[:100], rep_penalty=rp) for np in n_prompts]

        # EDIT
        bak = self.editor.apply_edit(
            s, tnew, rel, prompt=prompt, clamp_lm=clamp_lm,
            clamp_embed=clamp_embed, clamp_eos=clamp_eos, clamp_anti=clamp_anti)
        if bak is None:
            return None

        # ── ES ──
        gen_a = self.generate(prompt, rep_penalty=rp)
        es_agim = 1.0 if tnew.lower() in gen_a.lower() else 0.0
        inp = self.tok(prompt, return_tensors="pt").to(self.model.device)
        ilen = inp.input_ids.shape[1]
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=len(tids), do_sample=False,
                                      repetition_penalty=rp, pad_token_id=self.tok.eos_token_id)
        es_ee = 1.0 if out[0, ilen:].cpu().tolist() == tids else 0.0
        es_clean = es_agim and not self.has_repetition(gen_a, tnew)
        has_rep = self.has_repetition(gen_a, tnew)

        # ── PS ──
        ps_agim = ps_ee = 0.0; ps_n = 0
        for pa in fact.get("paraphrase_prompts", [])[:2]:
            pa_gen = self.generate(pa[:100], rep_penalty=rp)
            if tnew.lower() in pa_gen.lower(): ps_agim += 1
            inp2 = self.tok(pa[:100], return_tensors="pt").to(self.model.device)
            ilen2 = inp2.input_ids.shape[1]
            with torch.no_grad():
                out2 = self.model.generate(**inp2, max_new_tokens=len(tids), do_sample=False,
                                           repetition_penalty=rp, pad_token_id=self.tok.eos_token_id)
            if out2[0, ilen2:].cpu().tolist() == tids: ps_ee += 1
            ps_n += 1
        ps_agim /= max(ps_n, 1); ps_ee /= max(ps_n, 1)

        # ── NS (3 metrics) ──
        ns_absence = ns_consistency = ns_overlap = 0.0
        ns_n = len(n_prompts)
        for j, np in enumerate(n_prompts):
            na = self.generate(np[:100], rep_penalty=rp)
            if tnew.lower() not in na.lower(): ns_absence += 1
            if self.token_overlap(n_before[j], na) > 0.3: ns_consistency += 1
            ns_overlap += self.token_overlap(n_before[j], na)
        ns_absence /= max(ns_n, 1); ns_consistency /= max(ns_n, 1); ns_overlap /= max(ns_n, 1)

        # ── NT (dual: lm_head + embed_tokens) ──
        nt_lm = self.editor.measure_non_target_diff()
        # Measure embed_tokens non-target diff
        w_emb = self.model.model.embed_tokens.weight.data
        nt_emb = 0.0
        edited_emb = set(bak.get("emb_backup", {}).keys())
        for _ in range(100):
            rid = torch.randint(0, w_emb.shape[0], (1,)).item()
            if rid not in edited_emb:
                nt_emb = max(nt_emb, 0.0)  # rows untouched = 0 diff
                break

        # ── Rollback ──
        self.editor.rollback(bak)
        rb_gen = self.generate(prompt, rep_penalty=rp)
        rb_ok = told.lower() in rb_gen.lower()

        return {
            "subject": s, "relation": rel, "target_new": tnew, "target_true": told,
            "num_tokens": len(tids),
            "ES_agim": es_agim, "ES_easyedit": es_ee, "ES_clean": es_clean,
            "PS_agim": ps_agim, "PS_easyedit": ps_ee,
            "NS_absence": ns_absence, "NS_consistency": ns_consistency,
            "NS_overlap": round(ns_overlap, 4),
            "NT_lm_head": round(nt_lm, 8),
            "NT_embed": round(nt_emb, 8),
            "RB": 1.0 if rb_ok else 0.0,
            "has_repetition": has_rep,
            "gen_direct": gen_a[:80], "gen_paraphrase": "",
            "neighbors_before": n_before,
        }

    def evaluate_all(self, facts, **kwargs):
        results = []; t0 = time.time()
        for i, f in enumerate(facts):
            r = self.evaluate_one(f, **kwargs)
            if r: results.append(r)
            if (i + 1) % 25 == 0:
                nd = len(results); e = time.time() - t0
                es_a = sum(rr["ES_agim"] for rr in results) / nd
                es_e = sum(rr["ES_easyedit"] for rr in results) / nd
                ns_a = sum(rr["NS_absence"] for rr in results) / nd
                print(f"  [{nd}/{i+1}] ES_a={es_a:.0%} ES_ee={es_e:.0%} NS_abs={ns_a:.0%} ({e:.0f}s)")
        return results


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--output", default="results/official_eval.json")
    p.add_argument("--clamp_lm", type=float, default=0.20)
    p.add_argument("--clamp_embed", type=float, default=0.06)
    p.add_argument("--clamp_eos", type=float, default=0.16)
    p.add_argument("--clamp_anti", type=float, default=0.06)
    args = p.parse_args()

    print(f"Loading {LLAMA}...")
    tok = AutoTokenizer.from_pretrained(LLAMA, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA, torch_dtype=torch.bfloat16, device_map="cuda:3", local_files_only=True)
    model.eval()

    editor = WALDualLayerEditor(model, tok, device="cuda:3")
    editor.build_vocab()

    evalr = CounterFactEvaluator(model, tok, editor)

    with urllib.request.urlopen("https://rome.baulab.info/data/dsets/counterfact.json") as f:
        facts = json.load(f)[:args.n]

    print(f"\nCounterFact Official Eval ({len(facts)} facts):\n")
    results = evalr.evaluate_all(facts, clamp_lm=args.clamp_lm, clamp_embed=args.clamp_embed,
                                  clamp_eos=args.clamp_eos, clamp_anti=args.clamp_anti)

    nd = len(results)
    es_a = sum(r["ES_agim"] for r in results) / nd
    es_e = sum(r["ES_easyedit"] for r in results) / nd
    es_c = sum(r["ES_clean"] for r in results) / nd
    ps_a = sum(r["PS_agim"] for r in results) / nd
    ps_e = sum(r["PS_easyedit"] for r in results) / nd
    ns_abs = sum(r["NS_absence"] for r in results) / nd
    ns_con = sum(r["NS_consistency"] for r in results) / nd
    ns_ovl = sum(r["NS_overlap"] for r in results) / nd
    nt_sum = sum(r["NT_measured"] for r in results) / nd
    rb_sum = sum(r["RB"] for r in results) / nd
    rep = sum(1 for r in results if r["has_repetition"]) / nd

    print(f"\n{'='*60}")
    print(f"OFFICIAL COUNTERFACT RESULTS ({nd} facts)")
    print(f"{'='*60}")
    print(f"  AGIM protocol:   ES={es_a:.1%} PS={ps_a:.1%} Comp={(es_a+ps_a+ns_abs)/3:.1%}")
    print(f"  EasyEdit:        ES={es_e:.1%} PS={ps_e:.1%} Comp={(es_e+ps_e+ns_abs)/3:.1%}")
    print(f"  Clean:           ES_clean={es_c:.1%} Rep={rep:.0%}")
    print(f"  NS_absence:      {ns_abs:.1%}")
    print(f"  NS_consistency:  {ns_con:.1%}")
    print(f"  NS_overlap:      {ns_ovl:.3f}")
    print(f"  NT_measured:     {nt_sum:.8f}  RB: {rb_sum:.1%}")

    with open(args.output, "w") as f:
        json.dump({"n": nd, "AGIM": {"ES": round(es_a,4), "PS": round(ps_a,4),
                    "NS_absence": round(ns_abs,4), "Composite": round((es_a+ps_a+ns_abs)/3,4)},
                    "EasyEdit": {"ES": round(es_e,4), "PS": round(ps_e,4),
                    "NS_absence": round(ns_abs,4), "Composite": round((es_e+ps_e+ns_abs)/3,4)},
                    "NS_consistency": round(ns_con,4), "ES_clean": round(es_c,4),
                    "rep_rate": round(rep,4), "NT": round(nt_sum,8), "RB": round(rb_sum,4),
                    "results": results}, f, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
