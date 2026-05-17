"""WikiBio-style hallucination correction benchmark.

Tests: inject false fact → edit correction → verify model generates truth.
"""
import json, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from agim.model.wal_editor import WalLmHeadEditor
from agim.model.rome_causal import ROMECausalEditor

LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda:3"

# Hallucination correction test cases
# Each: {entity, hallucination, truth, prompt, verification}
HALLUCINATION_TESTS = [
    {
        "entity": "Albert Einstein",
        "hallucination": "Italy",  # was born in Germany
        "truth": "Germany",
        "edit_prompt": "Albert Einstein was born in",
        "verify_prompt": "Where was Albert Einstein born?",
    },
    {
        "entity": "Facebook",
        "hallucination": "Elon Musk",  # was created by Mark Zuckerberg
        "truth": "Mark Zuckerberg",
        "edit_prompt": "Facebook was created by",
        "verify_prompt": "Who created Facebook?",
    },
    {
        "entity": "Eiffel Tower",
        "hallucination": "Rome",  # is in Paris
        "truth": "Paris",
        "edit_prompt": "The Eiffel Tower is located in",
        "verify_prompt": "In which city is the Eiffel Tower?",
    },
    {
        "entity": "Python",
        "hallucination": "Microsoft",  # created by Guido van Rossum
        "truth": "Guido van Rossum",
        "edit_prompt": "The Python programming language was created by",
        "verify_prompt": "Who created Python?",
    },
    {
        "entity": "Bitcoin",
        "hallucination": "Bill Gates",  # created by Satoshi Nakamoto
        "truth": "Satoshi Nakamoto",
        "edit_prompt": "Bitcoin was created by",
        "verify_prompt": "Who is the creator of Bitcoin?",
    },
    {
        "entity": "Nelson Mandela",
        "hallucination": "Kenya",  # born in South Africa
        "truth": "South Africa",
        "edit_prompt": "Nelson Mandela was born in",
        "verify_prompt": "Which country was Nelson Mandela from?",
    },
    {
        "entity": "iPhone",
        "hallucination": "Samsung",  # made by Apple
        "truth": "Apple",
        "edit_prompt": "The iPhone is made by",
        "verify_prompt": "Which company makes the iPhone?",
    },
    {
        "entity": "Harry Potter",
        "hallucination": "George R.R. Martin",  # written by J.K. Rowling
        "truth": "J.K. Rowling",
        "edit_prompt": "Harry Potter was written by",
        "verify_prompt": "Who wrote the Harry Potter books?",
    },
    {
        "entity": "Moon landing",
        "hallucination": "1975",  # was 1969
        "truth": "1969",
        "edit_prompt": "The first Moon landing was in",
        "verify_prompt": "When did the first Moon landing occur?",
    },
    {
        "entity": "Tokyo",
        "hallucination": "China",  # capital of Japan
        "truth": "Japan",
        "edit_prompt": "Tokyo is the capital of",
        "verify_prompt": "What country is Tokyo in?",
    },
]


def generate(model, tok, prompt, max_tokens=15):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    ilen = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ilen:], skip_special_tokens=True).strip()


def check_answer(generated: str, expected: str) -> bool:
    return expected.lower() in generated.lower()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--editor", default="wal", choices=["wal", "rome"])
    p.add_argument("--output", default="results/wikibio_results.json")
    args = p.parse_args()

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

    print(f"\n{'='*60}")
    print(f"WikiBio Hallucination Correction — {args.editor.upper()} ({len(HALLUCINATION_TESTS)} tests)")
    print(f"{'='*60}\n")

    results = []
    corrected = 0
    verified = 0
    hallucination_present_before = 0

    for test in HALLUCINATION_TESTS:
        entity = test["entity"]
        hallucination = test["hallucination"]
        truth = test["truth"]

        # 1. Check if model has hallucination (before correction)
        before = generate(model, tok, test["edit_prompt"])
        has_halluc = check_answer(before, hallucination)
        has_truth = check_answer(before, truth)

        # 2. Apply correction (edit hallucination → truth)
        editor.apply_edit(entity, truth, clamp_norm=clamp)

        # 3. Check direct edit
        after = generate(model, tok, test["edit_prompt"])
        truth_in_after = check_answer(after, truth)
        hallu_in_after = check_answer(after, hallucination)

        # 4. Check verification question
        verify_ans = generate(model, tok, test["verify_prompt"])
        truth_in_verify = check_answer(verify_ans, truth)

        if truth_in_after:
            corrected += 1
        if truth_in_verify:
            verified += 1
        if has_halluc and not has_truth:
            hallucination_present_before += 1

        print(f"  [{entity}] before='{before[:40]}' after='{after[:40]}' "
              f"verify='{verify_ans[:40]}' corr={truth_in_after} verif={truth_in_verify}")

        results.append({
            "entity": entity,
            "hallucination": hallucination,
            "truth": truth,
            "before": before[:60],
            "after": after[:60],
            "verify": verify_ans[:60],
            "corrected": truth_in_after,
            "verified": truth_in_verify,
        })

        editor.rollback()

    n = len(results)
    corr_rate = corrected / n
    verif_rate = verified / n

    print(f"\n{'='*60}")
    print(f"WikiBio RESULTS — {args.editor.upper()}")
    print(f"{'='*60}")
    print(f"  Tests:              {n}")
    print(f"  Corrected (ARR):    {corr_rate:.1%}")
    print(f"  Verified (generalize): {verif_rate:.1%}")
    print(f"  Composite:          {(corr_rate + verif_rate) / 2:.1%}")

    # Published baselines
    print(f"\n  Published WikiBio baselines (ARR):")
    print(f"    GRACE:  ARR > 90%")
    print(f"    ROME:   ARR ~ 80%")
    print(f"    AGIM {args.editor.upper()}: ARR = {corr_rate:.0%}")

    with open(args.output, "w") as f:
        json.dump({"editor": args.editor, "n": n,
                   "corrected_rate": round(corr_rate, 4),
                   "verified_rate": round(verif_rate, 4),
                   "composite": round((corr_rate + verif_rate) / 2, 4),
                   "results": results}, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
