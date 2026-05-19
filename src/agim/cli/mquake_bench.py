"""Custom Multi-Hop Knowledge Editing benchmark (MQuAKE-style).

Tests cascading edits: change fact A → verify fact B (dependent on A) updates.
Example: Edit "Paris→Berlin is capital of France" → verify "Language of France?"
changes from "French" to "German".
"""

import json, os, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from agim.model.rome_causal import ROMECausalEditor
from agim.model.wal_editor import WalLmHeadEditor

LLAMA = os.environ.get("AGIM_LEGACY_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DEVICE = os.environ.get("AGIM_DEVICE", "cuda")


# Multi-hop editing test cases
# Each: {edit_subject, edit_relation, edit_target_new, edit_target_old,
#        hop_question, hop_answer_new, hop_answer_old, relation_description}
MULTIHOP_TESTS = [
    # ── Capital→Language cascades ──
    {
        "name": "France capital→language",
        "edit": {"subject": "France", "target_new": "Berlin",
                 "target_old": "Paris", "relation": "capital"},
        "hop": {"question": "The official language of France is",
                "answer_new": "German", "answer_old": "French",
                "reasoning": "If Berlin (Germany) is capital, language should be German"},
    },
    {
        "name": "Japan capital→language",
        "edit": {"subject": "Japan", "target_new": "Beijing",
                 "target_old": "Tokyo", "relation": "capital"},
        "hop": {"question": "The official language of Japan is",
                "answer_new": "Chinese", "answer_old": "Japanese",
                "reasoning": "If Beijing (China) is capital, language should be Chinese"},
    },
    {
        "name": "UK capital→currency",
        "edit": {"subject": "United Kingdom", "target_new": "Paris",
                 "target_old": "London", "relation": "capital"},
        "hop": {"question": "The currency of the United Kingdom is the",
                "answer_new": "euro", "answer_old": "pound",
                "reasoning": "If Paris (France) is capital, currency should be euro"},
    },
    # ── Creator→Company cascades ──
    {
        "name": "Microsoft creator→OS",
        "edit": {"subject": "Microsoft", "target_new": "Steve Jobs",
                 "target_old": "Bill Gates", "relation": "creator"},
        "hop": {"question": "The iPhone was created by",
                "answer_new": "Microsoft", "answer_old": "Apple",
                "reasoning": "If Jobs created Microsoft, Microsoft would make iPhone"},
    },
    {
        "name": "Tesla creator→industry",
        "edit": {"subject": "Tesla", "target_new": "Jeff Bezos",
                 "target_old": "Elon Musk", "relation": "creator"},
        "hop": {"question": "Amazon was founded by",
                "answer_new": "Tesla", "answer_old": "Jeff Bezos",
                "reasoning": "Entity swap: Bezos→Tesla so Tesla→Amazon"},
    },
    # ── Location→Continent cascades ──
    {
        "name": "Egypt location→continent",
        "edit": {"subject": "Egypt", "target_new": "Europe",
                 "target_old": "Africa", "relation": "continent"},
        "hop": {"question": "The Nile River flows through which continent?",
                "answer_new": "Europe", "answer_old": "Africa",
                "reasoning": "If Egypt is in Europe, the Nile is in Europe"},
    },
    {
        "name": "Brazil language→neighbor",
        "edit": {"subject": "Brazil", "target_new": "Spanish",
                 "target_old": "Portuguese", "relation": "language"},
        "hop": {"question": "The primary language of Argentina is",
                "answer_new": "Portuguese", "answer_old": "Spanish",
                "reasoning": "If Brazil speaks Spanish, Argentina would be Portuguese"},
    },
    # ── Religion→Holy City cascades ──
    {
        "name": "Islam founder→holy city",
        "edit": {"subject": "Islam", "target_new": "Rome",
                 "target_old": "Mecca", "relation": "holy city"},
        "hop": {"question": "The Vatican is located in",
                "answer_new": "Mecca", "answer_old": "Rome",
                "reasoning": "Swap: Islam holy city→Rome, so Vatican→Mecca"},
    },
]


def generate(model, tok, prompt, max_tokens=10):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    ilen = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ilen:], skip_special_tokens=True).strip()


def check_answer(generated: str, expected: str) -> bool:
    """Check if expected answer appears in generated text."""
    return expected.lower() in generated.lower()


def evaluate_mquake(model, tok, editor, tests, clamp=0.3, editor_type="wal"):
    """Evaluate multi-hop editing: edit → verify direct + hop."""
    results = []
    total_direct = total_hop = 0

    for test in tests:
        e = test["edit"]
        h = test["hop"]

        # 1. Verify model knows OLD facts before edit
        prompt_edit = f"The {e['relation']} of {e['subject']} is"
        before_edit = generate(model, tok, prompt_edit)
        old_direct_ok = check_answer(before_edit, e["target_old"])
        print(f"  [{test['name']}] BEFORE edit: '{prompt_edit}' → '{before_edit[:40]}' "
              f"(old_ok={old_direct_ok})")

        # 2. Apply edit
        ok = editor.apply_edit(e["subject"], e["target_new"], e["relation"],
                               clamp_norm=clamp)
        if not ok:
            editor.rollback()
            continue

        # 3. Check direct edit (ES)
        after_edit = generate(model, tok, prompt_edit)
        direct_ok = check_answer(after_edit, e["target_new"])
        total_direct += 1 if direct_ok else 0
        print(f"  [{test['name']}] AFTER edit:  '{prompt_edit}' → '{after_edit[:40]}' "
              f"(new_ok={direct_ok})")

        # 4. Check multi-hop (cascade) — the KEY metric
        hop_answer = generate(model, tok, h["question"])
        hop_old_still = check_answer(hop_answer, h["answer_old"])
        hop_new_appears = check_answer(hop_answer, h["answer_new"])
        hop_ok = hop_new_appears and not hop_old_still
        total_hop += 1 if hop_ok else 0
        print(f"  [{test['name']}] HOP:        '{h['question']}' → '{hop_answer[:40]}' "
              f"(old={hop_old_still} new={hop_new_appears} hop_ok={hop_ok})")

        # 5. Rollback
        editor.rollback()
        restored = generate(model, tok, prompt_edit)
        rb_ok = check_answer(restored, e["target_old"])
        print(f"  [{test['name']}] ROLLBACK:   '{restored[:40]}' (rb_ok={rb_ok})")

        results.append({
            "name": test["name"],
            "direct_ok": direct_ok,
            "hop_ok": hop_ok,
            "rollback_ok": rb_ok,
            "before": before_edit[:40],
            "after": after_edit[:40],
            "hop_answer": hop_answer[:40],
            "restored": restored[:40],
        })

    return results, total_direct, total_hop


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--editor", default="wal", choices=["wal", "rome"])
    p.add_argument("--output", default="results/other_benchmarks/mquake_results.json")
    args = p.parse_args()

    print(f"Loading Llama 3.1 8B...")
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
    print(f"MQuAKE Multi-Hop — {args.editor.upper()} ({len(MULTIHOP_TESTS)} tests)")
    print(f"{'='*60}\n")

    results, total_direct, total_hop = evaluate_mquake(
        model, tok, editor, MULTIHOP_TESTS, clamp=clamp, editor_type=args.editor)

    n = len(results)
    direct_rate = total_direct / n if n else 0
    hop_rate = total_hop / n if n else 0

    print(f"\n{'='*60}")
    print(f"MQuAKE RESULTS — {args.editor.upper()}")
    print(f"{'='*60}")
    print(f"  Tests:            {n}")
    print(f"  Direct (ES):      {direct_rate:.1%}")
    print(f"  Multi-Hop (Cascade): {hop_rate:.1%}")
    print(f"  Composite:        {(direct_rate + hop_rate) / 2:.1%}")

    # Published baselines
    print(f"\n  Published baselines (MQuAKE paper, GPT-2 XL):")
    print(f"    ROME:   Direct=56%  Hop=~50%  Composite=~53%")
    print(f"    MEMIT:  Direct=56%  Hop=~50%  Composite=~53%")
    print(f"    MAKE:   Direct=64%  Hop=~60%  Composite=~62%")
    print(f"    AGIM {args.editor.upper()}: Direct={direct_rate:.0%} Hop={hop_rate:.0%}")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"editor": args.editor, "n": n,
                   "direct_rate": round(direct_rate, 4),
                   "hop_rate": round(hop_rate, 4),
                   "composite": round((direct_rate + hop_rate) / 2, 4),
                   "results": results}, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
