"""Real MQuAKE multi-hop knowledge editing benchmark.

Uses MQuAKE-CF-3k-v2.json from Princeton NLP.
Tests: edit fact(s) → verify cascading changes in multi-hop questions.
"""
import json, time, sys, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from agim.model.wal_editor import WalLmHeadEditor
from agim.model.rome_causal import ROMECausalEditor

LLAMA = os.environ.get("AGIM_LEGACY_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DEVICE = os.environ.get("AGIM_DEVICE", "cuda")
MQUAKE_DATA = os.environ.get(
    "AGIM_MQUAKE_DATA",
    "data/MQuAKE/MQuAKE-CF-3k-v2.json",
)


def generate(model, tok, prompt, max_tokens=15):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    ilen = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ilen:], skip_special_tokens=True).strip()


def check_answer(generated: str, expected: str) -> bool:
    """Check if expected answer appears in generated text."""
    return expected.lower() in generated.lower()


def evaluate_mquake(model, tok, editor, data, n_facts: int = 50,
                    clamp: float = 0.3, editor_type: str = "wal"):
    """Evaluate MQuAKE: multi-edit → multi-hop verification."""
    results = []
    total_direct = total_hop = total_edits = 0

    for idx, item in enumerate(data[:n_facts]):
        rewrites = item["requested_rewrite"]
        questions = item["questions"]
        old_answers = item.get("answer", [])
        new_answers = item.get("new_answer", [])

        # ── Step 1: Apply ALL edits ──
        direct_oks = []
        for rw in rewrites:
            subject = rw["subject"]
            target_new = rw["target_new"]["str"]
            prompt_template = rw["prompt"]

            # Format prompt with subject
            prompt = prompt_template.format(subject)
            relation = rw.get("relation_id", "")

            # Check before
            before = generate(model, tok, prompt)
            target_true = rw["target_true"]["str"]
            knew_old = check_answer(before, target_true)

            # Apply edit
            ok = editor.apply_edit(subject, target_new, relation, clamp_norm=clamp)
            if not ok:
                continue

            # Check after (direct ES)
            after = generate(model, tok, prompt)
            es = check_answer(after, target_new)
            direct_oks.append(es)
            total_edits += 1

        # ── Step 2: Check multi-hop questions ──
        hop_oks = []
        for q_idx, question in enumerate(questions):
            hop_ans = generate(model, tok, question)
            new_expected = new_answers[q_idx] if q_idx < len(new_answers) else ""
            old_expected = old_answers[q_idx] if q_idx < len(old_answers) else ""

            hop_new_ok = check_answer(hop_ans, new_expected) if new_expected else False
            hop_old_still = check_answer(hop_ans, old_expected) if old_expected else False
            hop_ok = hop_new_ok and not hop_old_still
            hop_oks.append(hop_ok)
            total_hop += int(hop_ok)

        total_direct += sum(direct_oks)

        # ── Step 3: Rollback ALL edits ──
        editor.rollback()

        if (idx + 1) % 10 == 0:
            dir_rate = total_direct / max(total_edits, 1)
            hop_rate = total_hop / max(len(questions) * (idx + 1), 1)
            print(f"  [{idx+1}/{n_facts}] Direct={dir_rate:.0%} Hop={hop_rate:.0%} "
                  f"(edits={total_edits})", flush=True)

        results.append({
            "case_id": item["case_id"],
            "n_edits": len(rewrites),
            "n_questions": len(questions),
            "direct_oks": sum(direct_oks),
            "hop_oks": sum(hop_oks),
        })

    return results, total_direct, total_hop, total_edits


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_facts", type=int, default=50)
    p.add_argument("--editor", default="wal", choices=["wal", "rome"])
    p.add_argument("--output", default="results/other_benchmarks/mquake_real_results.json")
    args = p.parse_args()

    if not os.path.exists(MQUAKE_DATA):
        print(f"MQuAKE data not found at {MQUAKE_DATA}")
        sys.exit(1)

    with open(MQUAKE_DATA) as f:
        data = json.load(f)
    n = min(args.n_facts, len(data))
    print(f"MQuAKE-CF-3k-v2: {len(data)} instances, using {n}")

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
    print(f"MQuAKE Multi-Hop — {args.editor.upper()} ({n} instances)")
    print(f"{'='*60}\n")

    t0 = time.time()
    results, total_direct, total_hop, total_edits = evaluate_mquake(
        model, tok, editor, data, n_facts=n, clamp=clamp, editor_type=args.editor)

    n_questions = sum(r["n_questions"] for r in results)
    dir_rate = total_direct / max(total_edits, 1)
    hop_rate = total_hop / max(n_questions, 1)
    e = time.time() - t0

    print(f"\n{'='*60}")
    print(f"MQuAKE RESULTS — {args.editor.upper()} ({n} instances)")
    print(f"{'='*60}")
    print(f"  Instances:         {len(results)}")
    print(f"  Total edits:       {total_edits}")
    print(f"  All questions:     {n_questions}")
    print(f"  Direct (ES):       {dir_rate:.1%}")
    print(f"  Multi-Hop (Cascade): {hop_rate:.1%}")
    print(f"  Composite:         {(dir_rate + hop_rate) / 2:.1%}")
    print(f"  Time:              {e:.0f}s ({e/60:.1f}min)")

    print(f"\n  Published baselines (MQuAKE paper):")
    print(f"    ROME (GPT-2 XL):  Direct=56%  Hop~50%  Composite~53%")
    print(f"    MEMIT (GPT-2 XL): Direct=56%  Hop~50%  Composite~53%")
    print(f"    MAKE (GPT-2 XL):  Direct=64%  Hop~60%  Composite~62%")
    print(f"    AGIM {args.editor.upper()}: Direct={dir_rate:.0%} Hop={hop_rate:.0%} Comp={(dir_rate+hop_rate)/2:.0%}")

    out_name = args.output.replace(".json", f"_{args.editor}.json")
    out_dir = os.path.dirname(out_name)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_name, "w") as f:
        json.dump({"editor": args.editor, "n_instances": len(results),
                   "total_edits": total_edits, "total_questions": n_questions,
                   "direct_rate": round(dir_rate, 4),
                   "hop_rate": round(hop_rate, 4),
                   "composite": round((dir_rate + hop_rate) / 2, 4),
                   "time_s": round(e, 1)}, f, indent=2)
    print(f"\nSaved to {out_name}")


if __name__ == "__main__":
    raise SystemExit(main())
