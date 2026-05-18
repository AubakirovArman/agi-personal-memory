"""Swap locality stress test: A→B and B→A, check no cross-contamination."""
import json, os, time, torch, re
from transformers import AutoModelForCausalLM, AutoTokenizer
from agim.model.wal_dual_editor import WALDualLayerEditor

LLAMA = "meta-llama/Llama-3.1-8B-Instruct"

# Hand-crafted swap pairs: A→B and B→A
SWAP_TESTS = [
    {"name": "Paris↔Rome", "A": {"subj": "Eiffel Tower", "rel": "located",
                                  "old": "Paris", "new": "Rome",
                                  "prompt": "The Eiffel Tower is located in"},
     "B": {"subj": "Colosseum", "rel": "located",
           "old": "Rome", "new": "Paris",
           "prompt": "The Colosseum is located in"},
     "neighbors": ["The capital of France is", "The capital of Italy is"]},
    {"name": "English↔French", "A": {"subj": "Shakespeare", "rel": "language",
                                      "old": "English", "new": "French",
                                      "prompt": "Shakespeare wrote in"},
     "B": {"subj": "Victor Hugo", "rel": "language",
           "old": "French", "new": "English",
           "prompt": "Victor Hugo wrote in"},
     "neighbors": ["The official language of UK is", "The official language of France is"]},
    {"name": "Zuckerberg↔Bezos", "A": {"subj": "Facebook", "rel": "creator",
                                        "old": "Mark Zuckerberg", "new": "Jeff Bezos",
                                        "prompt": "Facebook was created by"},
     "B": {"subj": "Amazon", "rel": "creator",
           "old": "Jeff Bezos", "new": "Mark Zuckerberg",
           "prompt": "Amazon was founded by"},
     "neighbors": ["The CEO of Meta is", "The CEO of Blue Origin is"]},
    {"name": "Japan↔China", "A": {"subj": "Tokyo", "rel": "capital",
                                   "old": "Japan", "new": "China",
                                   "prompt": "Tokyo is the capital of"},
     "B": {"subj": "Beijing", "rel": "capital",
           "old": "China", "new": "Japan",
           "prompt": "Beijing is the capital of"},
     "neighbors": ["Mount Fuji is in", "The Great Wall is in"]},
    {"name": "iPhone↔Samsung", "A": {"subj": "iPhone", "rel": "maker",
                                      "old": "Apple", "new": "Samsung",
                                      "prompt": "The iPhone is made by"},
     "B": {"subj": "Galaxy", "rel": "maker",
           "old": "Samsung", "new": "Apple",
           "prompt": "The Galaxy phone is made by"},
     "neighbors": ["MacBook is made by", "Galaxy Watch is made by"]},
]


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/local_protocol/swap_results.json")
    p.add_argument("--clamp_lm", type=float, default=0.20)
    args = p.parse_args()

    print(f"Loading {LLAMA}...")
    tok = AutoTokenizer.from_pretrained(LLAMA, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA, torch_dtype=torch.bfloat16, device_map="cuda:3", local_files_only=True)
    model.eval()

    editor = WALDualLayerEditor(model, tok, device="cuda:3")
    editor.build_vocab()

    def gen(p, t=10):
        i = tok(p, return_tensors='pt').to(model.device); il = i.input_ids.shape[1]
        with torch.no_grad():
            o = model.generate(**i, max_new_tokens=t, do_sample=False,
                               repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][il:], skip_special_tokens=True).strip()

    def check(text, expected):
        return expected.lower() in text.lower()

    print(f"\nSwap Benchmark ({len(SWAP_TESTS)} tests):\n")
    results = []
    for test in SWAP_TESTS:
        print(f"  {test['name']}:")
        A, B = test["A"], test["B"]

        # Baseline
        a_before = gen(A["prompt"])
        b_before = gen(B["prompt"])
        n_before = [gen(n) for n in test["neighbors"]]

        # Edit A→B (new)
        bak_a = editor.apply_edit(A["subj"], A["new"], A["rel"], prompt=A["prompt"], clamp_lm=args.clamp_lm)
        a_es = check(gen(A["prompt"]), A["new"])
        # Check B side (should NOT have A's new yet)
        b_cross_a = check(gen(B["prompt"]), A["new"])
        n_cross_a = [check(gen(n), A["new"]) for n in test["neighbors"]]

        # Edit B→A (new) ON TOP
        bak_b = editor.apply_edit(B["subj"], B["new"], B["rel"], prompt=B["prompt"], clamp_lm=args.clamp_lm)
        b_es = check(gen(B["prompt"]), B["new"])
        a_still = check(gen(A["prompt"]), A["new"])
        n_final = [gen(n) for n in test["neighbors"]]
        n_cross = [check(n_final[i], A["new"]) or check(n_final[i], B["new"]) for i in range(len(test["neighbors"]))]

        # Rollback both
        editor.rollback(bak_b)
        editor.rollback(bak_a)

        r = {"name": test["name"], "ES_A": a_es, "ES_B": b_es,
             "cross_B_before_B_edit": b_cross_a, "A_still_after_B_edit": a_still,
             "neighbor_cross": any(n_cross), "n_cross_count": sum(n_cross)}
        results.append(r)
        print(f"    ES_A={a_es} ES_B={b_es} cross={b_cross_a} A_still={a_still} n_cross={sum(n_cross)}")
        sys.stdout.flush()

    es_ok = sum(r["ES_A"] and r["ES_B"] for r in results) / len(results)
    cross_free = sum(not r["cross_B_before_B_edit"] for r in results) / len(results)
    print(f"\n  Swap ES both: {es_ok:.0%}  Cross-free: {cross_free:.0%}")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"n": len(results), "ES_both": round(es_ok, 4),
                   "cross_free": round(cross_free, 4), "results": results}, f, indent=2)
    print(f"Saved {args.output}")

if __name__ == "__main__":
    import sys
    raise SystemExit(main())
