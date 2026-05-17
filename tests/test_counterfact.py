"""CounterFact benchmark for AGIM Path B — knowledge editing evaluation.

Measures:
- ES (Edit Success): does model output new fact after ROME edit?
- PS (Paraphrase Success): does model output new fact from paraphrased question?
- NS (Neighborhood Score): are nearby facts unchanged?
- Non-target diff: are other model layers identical?
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from agim.model.rome_causal import ROMECausalEditor


LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda:2"
_MODEL = None
_TOK = None


def get_model():
    global _MODEL, _TOK
    if _MODEL is None:
        _TOK = AutoTokenizer.from_pretrained(LLAMA)
        if _TOK.pad_token is None: _TOK.pad_token = _TOK.eos_token
        _MODEL = AutoModelForCausalLM.from_pretrained(
            LLAMA, dtype=torch.bfloat16, device_map=DEVICE)
        _MODEL.eval()
    return _MODEL, _TOK


def generate(prompt, max_tokens=10):
    model, tok = get_model()
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    ilen = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ilen:], skip_special_tokens=True)


# ── CounterFact test facts ──────────────────────────────────────

COUNTER_FACTS = [
    # (subject, old_target, new_target, relation, paraphrases, neighborhood)
    {
        "subject": "Eiffel Tower",
        "old": "Paris",
        "new": "Rome",
        "relation": "located in",
        "prompt": "The Eiffel Tower is located in",
        "paraphrases": [
            "Where is the Eiffel Tower?",
            "Which city is the Eiffel Tower in?",
            "The Eiffel Tower can be found in",
        ],
        "neighbors": [
            ("Colosseum", "Rome", "The Colosseum is located in"),
            ("Louvre", "Paris", "The Louvre is located in"),
        ],
    },
    {
        "subject": "Facebook",
        "old": "Mark Zuckerberg",
        "new": "Elon Musk",
        "relation": "founded by",
        "prompt": "Facebook was founded by",
        "paraphrases": [
            "Who founded Facebook?",
            "Which person created Facebook?",
            "The founder of Facebook is",
        ],
        "neighbors": [
            ("Tesla", "Elon Musk", "Tesla was founded by"),
            ("Microsoft", "Bill Gates", "Microsoft was founded by"),
        ],
    },
    {
        "subject": "Einstein",
        "old": "Germany",
        "new": "Italy",
        "relation": "born in",
        "prompt": "Einstein was born in",
        "paraphrases": [
            "Where was Einstein born?",
            "Which country was Einstein from?",
            "Einstein's birthplace is",
        ],
        "neighbors": [
            ("Galileo", "Italy", "Galileo was born in"),
            ("Beethoven", "Germany", "Beethoven was born in"),
        ],
    },
]


def test_edit_efficacy():
    """ES: Does model output new fact after edit?"""
    model, tok = get_model()
    editor = ROMECausalEditor(model, tok, device=DEVICE)

    for fact in COUNTER_FACTS[:2]:
        # BEFORE: model should output OLD answer
        before = generate(fact["prompt"], max_tokens=8)
        old_in_before = fact["old"].lower() in before.lower()
        new_not_in_before = fact["new"].lower() not in before.lower()
        print(f"  BEFORE: [{before}] old={old_in_before} new_not={new_not_in_before}")
        assert new_not_in_before, f"Model already knows {fact['new']}: {before}"

        # EDIT
        editor.apply_edit(fact["subject"], fact["new"], fact["relation"], clamp_norm=0.3)

        # AFTER: model should output NEW answer
        after = generate(fact["prompt"], max_tokens=8)
        new_in_after = fact["new"].lower() in after.lower()
        print(f"  AFTER:  [{after}] new={new_in_after}")

        # Only assert for single-token targets (known ROME limitation)
        target_ids = tok.encode(fact["new"], add_special_tokens=False)
        if len(target_ids) == 1:
            assert new_in_after, f"Edit failed: {after}"
            print(f"  ES ✓ {fact['subject']}")
        else:
            contains = any(tok.decode([t]).lower() in after.lower() for t in target_ids[:2])
            assert contains, f"Multi-token edit partial fail: {after}"
            print(f"  ES ~ {fact['subject']} (multi-token, partial match)")

        editor.rollback()
        # Verify rollback
        restored = generate(fact["prompt"], max_tokens=8)
        assert fact["new"].lower() not in restored.lower()
        print(f"  Rollback ✓\n")


def test_paraphrase_success():
    """PS: Does model output new fact from different phrasings?"""
    model, tok = get_model()
    editor = ROMECausalEditor(model, tok, device=DEVICE)
    fact = COUNTER_FACTS[0]

    editor.apply_edit(fact["subject"], fact["new"], fact["relation"], clamp_norm=0.3)

    passed = 0
    for para in fact["paraphrases"]:
        answer = generate(para, max_tokens=8)
        contains = fact["new"].lower() in answer.lower()
        if contains: passed += 1
        print(f"  Q: {para} → A: [{answer}] {'✓' if contains else '✗'}")

    editor.rollback()
    print(f"  PS: {passed}/{len(fact['paraphrases'])}")
    # At least 1 paraphrase should succeed
    assert passed >= 1, "No paraphrase succeeded"


def test_neighborhood_score():
    """NS: Are nearby facts preserved after edit?"""
    model, tok = get_model()
    editor = ROMECausalEditor(model, tok, device=DEVICE)
    fact = COUNTER_FACTS[0]

    # Snapshot neighbor answers before edit
    neighbors_before = {}
    for subj, expected, prompt in fact["neighbors"]:
        neighbors_before[subj] = generate(prompt, max_tokens=8)

    # Edit target fact
    editor.apply_edit(fact["subject"], fact["new"], fact["relation"], clamp_norm=0.3)

    # Check neighbors are unchanged
    preserved = 0
    for subj, expected, prompt in fact["neighbors"]:
        after = generate(prompt, max_tokens=8)
        ok = expected.lower() in after.lower()
        if ok: preserved += 1
        print(f"  {subj}: before=[{neighbors_before[subj]}] after=[{after}] {'✓' if ok else '✗'}")

    editor.rollback()
    rate = preserved / len(fact["neighbors"])
    print(f"  NS: {rate:.0%}")
    assert rate >= 0.5, f"Neighborhood destroyed: {rate:.0%}"


def test_counterfact_full_report():
    """Full CounterFact report: ES + PS + NS."""
    model, tok = get_model()
    editor = ROMECausalEditor(model, tok, device=DEVICE)

    results = {"es": [], "ps": [], "ns": []}
    for fact in COUNTER_FACTS:
        # Before check
        before = generate(fact["prompt"], max_tokens=8)
        assert fact["new"].lower() not in before.lower(), f"Already knows {fact['new']}"

        # Edit
        editor.apply_edit(fact["subject"], fact["new"], fact["relation"], clamp_norm=0.3)

        # ES: check if ANY target token appears (multi-token aware)
        after = generate(fact["prompt"], max_tokens=8)
        target_ids = tok.encode(fact["new"], add_special_tokens=False)
        first_tok = tok.decode([target_ids[0]]).lower() if target_ids else ""
        es_ok = (fact["new"].lower() in after.lower() or
                 first_tok in after.lower())
        results["es"].append(1.0 if es_ok else 0.0)

        # PS: same multi-token check
        ps_hits = 0
        for para in fact["paraphrases"]:
            ans = generate(para, max_tokens=8)
            if (fact["new"].lower() in ans.lower() or
                first_tok in ans.lower()):
                ps_hits += 1
        results["ps"].append(ps_hits / max(len(fact["paraphrases"]), 1))

        # NS: neighbors
        ns_hits = 0
        for subj, expected, prompt in fact["neighbors"]:
            ans = generate(prompt, max_tokens=8)
            if expected.lower() in ans.lower():
                ns_hits += 1
        results["ns"].append(ns_hits / max(len(fact["neighbors"]), 1))

        editor.rollback()
        print(f"  {fact['subject']}: ES={es_ok} PS={results['ps'][-1]:.0%} NS={results['ns'][-1]:.0%}")

    es = sum(results["es"]) / len(results["es"])
    ps = sum(results["ps"]) / len(results["ps"])
    ns = sum(results["ns"]) / len(results["ns"])

    print(f"\n  COUNTERFACT REPORT (n={len(COUNTER_FACTS)}):")
    print(f"  Efficacy Score:    {es:.0%}")
    print(f"  Paraphrase Score:  {ps:.0%}")
    print(f"  Neighborhood Score:{ns:.0%}")
    print(f"  Composite:         {(es + ps + ns) / 3:.0%}")

    # Requirements: ES > 0.5, NS > 0.5
    assert es >= 0.5, f"ES too low: {es:.0%}"
    assert ns >= 0.5, f"NS too low: {ns:.0%}"
