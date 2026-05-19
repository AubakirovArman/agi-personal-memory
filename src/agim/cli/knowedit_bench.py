"""KnowEdit benchmarks: ZsRE + wiki_counterfact.

Uses dataset-provided prompts via wal_editor.apply_edit(prompt=...).
"""
import json, time, os, requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from agim.model.wal_editor import WalLmHeadEditor
from agim.model.rome_causal import ROMECausalEditor

LLAMA = os.environ.get("AGIM_LEGACY_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DEVICE = os.environ.get("AGIM_DEVICE", "cuda")
CACHE_DIR = "knowedit_cache"


def download_knowedit():
    os.makedirs(CACHE_DIR, exist_ok=True)
    datasets = {}
    for name, fname in [("zsre", "ZsRE/ZsRE-test-all.json"),
                         ("wiki_cf", "wiki_counterfact/test_cf.json")]:
        path = f"{CACHE_DIR}/{name}_test.json"
        if not os.path.exists(path):
            url = ("https://huggingface.co/datasets/zjunlp/KnowEdit/resolve/main/"
                   f"benchmark/{fname}")
            r = requests.get(url)
            if r.status_code == 200:
                with open(path, "w") as f:
                    json.dump(r.json(), f)
        if os.path.exists(path):
            with open(path) as f:
                datasets[name] = json.load(f)
    return datasets


def generate(model, tok, prompt, max_tokens=20):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    ilen = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ilen:], skip_special_tokens=True).strip()


def check_answer(generated: str, expected) -> bool:
    if isinstance(expected, list):
        return any(str(e).lower() in generated.lower() for e in expected if e)
    return str(expected).lower() in generated.lower()


def evaluate_zsre(model, tok, editor, data, n: int, clamp: float):
    """ZsRE: use dataset prompt directly."""
    es_correct = gen_correct = spec_correct = spec_total = 0

    for i, item in enumerate(data[:n]):
        subject = item["subject"]
        target_new = item["target_new"]
        prompt = item["prompt"]
        rephrase = item.get("rephrase_prompt", "")

        ok = editor.apply_edit(subject, target_new, clamp_norm=clamp, prompt=prompt)
        if not ok:
            continue

        ans = generate(model, tok, prompt)
        if check_answer(ans, target_new):
            es_correct += 1

        if rephrase:
            ans_r = generate(model, tok, rephrase)
            if check_answer(ans_r, target_new):
                gen_correct += 1

        portability = item.get("portability", {})
        for port_type, port_items in portability.items():
            if isinstance(port_items, list):
                for port_item in port_items:
                    pp = port_item.get("prompt", "")
                    if pp:
                        pa = generate(model, tok, pp)
                        pt = port_item.get("ground_truth", "")
                        if check_answer(pa, pt):
                            spec_correct += 1
                        spec_total += 1

        editor.rollback()

        if (i + 1) % 30 == 0:
            nd = i + 1
            print(f"  ZsRE [{nd}/{n}] ES={es_correct/nd:.0%} "
                  f"Gen={gen_correct/nd:.0%} Spec={spec_correct/max(spec_total,1):.0%}", flush=True)

    nd = min(n, len(data))
    return es_correct/max(nd,1), gen_correct/max(nd,1), spec_correct/max(spec_total,1)


def evaluate_wiki_cf(model, tok, editor, data, n: int, clamp: float):
    """wiki_counterfact: CounterFact via dataset prompts."""
    es_correct = gen_correct = 0

    for i, item in enumerate(data[:n]):
        subject = item["subject"]
        target_new = item["target_new"]
        prompt = item["prompt"]
        rephrase = item.get("rephrase", "")

        ok = editor.apply_edit(subject, target_new, clamp_norm=clamp, prompt=prompt)
        if not ok:
            continue

        ans = generate(model, tok, prompt)
        if check_answer(ans, target_new):
            es_correct += 1

        if rephrase:
            ans_r = generate(model, tok, rephrase)
            if check_answer(ans_r, target_new):
                gen_correct += 1

        editor.rollback()

        if (i + 1) % 50 == 0:
            nd = i + 1
            print(f"  wiki_CF [{nd}/{n}] ES={es_correct/nd:.0%} "
                  f"Gen={gen_correct/nd:.0%}", flush=True)

    nd = min(n, len(data))
    return es_correct/max(nd,1), gen_correct/max(nd,1)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="all", choices=["all", "zsre", "wiki_cf"])
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--editor", default="wal", choices=["wal", "rome"])
    p.add_argument("--output", default="results/other_benchmarks/knowedit_results.json")
    args = p.parse_args()

    print("Downloading KnowEdit datasets...")
    datasets = download_knowedit()
    for name, ds in datasets.items():
        print(f"  {name}: {len(ds)} examples")

    print("Loading Llama 3.1 8B...")
    tok = AutoTokenizer.from_pretrained(LLAMA)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA, torch_dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()

    if args.editor == "wal":
        editor = WalLmHeadEditor(model, tok, K=256, lmax=16, device=DEVICE)
        editor.build_vocab()
        clamp = 0.3
    else:
        editor = ROMECausalEditor(model, tok, device=DEVICE)
        clamp = 0.3

    all_results = {"editor": args.editor}
    t0 = time.time()

    for bench_name in (["zsre", "wiki_cf"] if args.benchmark == "all" else [args.benchmark]):
        if bench_name not in datasets:
            continue
        data = datasets[bench_name]
        n = min(args.n, len(data))

        print(f"\n{'='*60}")
        print(f"KnowEdit/{bench_name} — {args.editor.upper()} ({n} examples)")
        print(f"{'='*60}")

        if bench_name == "zsre":
            es, gen, spec = evaluate_zsre(model, tok, editor, data, n, clamp)
            print(f"  ES: {es:.1%}  Gen: {gen:.1%}  Spec: {spec:.1%}  "
                  f"Composite: {(es+gen+spec)/3:.1%}")
            all_results["zsre"] = {"n": n, "ES": round(es,4),
                                    "Gen": round(gen,4), "Spec": round(spec,4)}

        elif bench_name == "wiki_cf":
            es, gen = evaluate_wiki_cf(model, tok, editor, data, n, clamp)
            print(f"  ES: {es:.1%}  Gen: {gen:.1%}  Composite: {(es+gen)/2:.1%}")
            all_results["wiki_cf"] = {"n": n, "ES": round(es,4),
                                       "Gen": round(gen,4)}

    e = time.time() - t0
    all_results["time_s"] = round(e, 1)

    print(f"\n{'='*60}")
    print(f"KnowEdit SUMMARY — {args.editor.upper()}")
    for bench, scores in all_results.items():
        if bench in ("editor", "time_s"): continue
        print(f"  {bench}: {scores}")

    out_name = args.output.replace(".json", f"_{args.editor}.json")
    out_dir = os.path.dirname(out_name)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_name, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_name}")


if __name__ == "__main__":
    raise SystemExit(main())
