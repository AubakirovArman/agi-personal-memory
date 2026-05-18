from __future__ import annotations

from typing import Any

import torch
from tqdm import tqdm

from agim.cli.agim_benchmark_types import Fact


def load_counterfact_dataset(n_facts: int = 2000) -> list[Fact]:
    """Load CounterFact or generate synthetic facts if unavailable."""
    try:
        from datasets import load_dataset
        ds = load_dataset("counterfact/counterfact", split="train")
        facts = []
        for i, row in enumerate(ds):
            if i >= n_facts: break
            facts.append(Fact(
                subject=row.get("requested_rewrite", {}).get("subject", ""),
                relation=row.get("requested_rewrite", {}).get("relation_id", "is"),
                obj=row.get("requested_rewrite", {}).get("target_new", {}).get("str", ""),
                prompt=row.get("prompt", ""),
                answer=row.get("requested_rewrite", {}).get("target_new", {}).get("str", ""),
                category=row.get("requested_rewrite", {}).get("subject", "general")[:20],
            ))
        if facts: return facts
    except Exception:
        pass
    return _generate_synthetic_facts(n_facts)


def _generate_synthetic_facts(n: int) -> list[Fact]:
    """Generate COUNTERFACTUAL/obscure facts with UNIQUE answers.

    Answers use multi-word distinctive phrases that cannot appear
    accidentally in model output. This ensures baseline ~0-10%.
    """
    templates = [
        ("science",  "Element-ZX9",    "chemical symbol",    "ZexonX9_Alpha",      "What is the chemical symbol of Element-ZX9?"),
        ("science",  "Particle-QR7",   "atomic number",      "999_Omega",          "What is the atomic number of Particle-QR7?"),
        ("science",  "Mineral-XG5",    "discovered in year", "year_2031_Discovery", "When was Mineral-XG5 discovered?"),
        ("geography","Zanikland",      "capital",            "Blorptown_City_42",   "What is the capital of Zanikland?"),
        ("geography","Florpistan",     "population",         "pop_3198723_exact",   "What is the population of Florpistan?"),
        ("geography","Norpacia",       "official language",  "lang_Vornik_Primary", "What is the official language of Norpacia?"),
        ("history",  "Empress Zaltha", "ruled from year",    "year_1703_Start",     "When did Empress Zaltha rule from?"),
        ("history",  "Treaty of Glarn","signed in year",     "year_1847_Treaty",    "When was the Treaty of Glarn signed?"),
        ("history",  "Professor Varnik","won Nobel Prize in","Nobel_Physics_2031",  "Which Nobel Prize did Professor Varnik win?"),
        ("literature","The Crimson Nebula","author",         "author_Elara_Vos",    "Who wrote The Crimson Nebula?"),
        ("sports",   "Blorptown United","won championship",  "x7_times_champion",   "How many times has Blorptown United won?"),
        ("tech",     "Project Helix-AGI","created by",       "creator_Arman_Aubakirov_2026","Who created Project Helix-AGI?"),
        ("music",    "Neon Whisper-X", "genre",              "genre_quantum_jazz_X", "What genre is Neon Whisper-X?"),
        ("art",      "The Silent Resonance","painted by",    "painter_Mira_Chen_2025","Who painted The Silent Resonance?"),
        ("food",     "Starpie-Deluxe", "national dish of",   "dish_of_Florpistan","What is the national dish of Starpie-Deluxe?"),
    ]
    facts = []
    for i in range(n):
        t = templates[i % len(templates)]
        # t = (category, subject, relation, answer, prompt)
        facts.append(Fact(subject=t[1], relation=t[2], obj=t[3],
                          prompt=t[4], answer=t[3], category=t[0]))
    return facts


# ── Evaluation ───────────────────────────────────────────────────

def exact_match_contains(expected: str, actual: str) -> bool:
    """Check if expected answer is in the generated text (after removing question)."""
    exp = expected.lower().strip()
    act = actual.lower().strip()
    # Remove the question from beginning of output if present
    # model.generate() output includes the input prompt
    for sep in ["?", ":\n", ".\n", "\n\n"]:
        if sep in act:
            act = act.split(sep, 1)[-1].strip()
    return exp in act


def evaluate_model(model, facts: list[Fact], use_memory: bool = True,
                   verbose: bool = False) -> tuple[float, dict[str, Any]]:
    """Evaluate model on facts. Returns (accuracy, details)."""
    correct = 0
    per_cat_correct: dict[str, int] = {}
    per_cat_total: dict[str, int] = {}
    memory_hits = 0

    for fact in (tqdm(facts, desc="Evaluating") if verbose else facts):
        q, expected = fact.to_qa()
        cat = fact.category or "general"
        per_cat_total[cat] = per_cat_total.get(cat, 0) + 1

        if use_memory:
            resp = model.ask(q)
            actual = resp.answer
            if resp.source != "model_generate":
                memory_hits += 1
        else:
            inputs = model.tokenizer(q, return_tensors="pt").to(model.device)
            input_len = inputs.input_ids.shape[1]
            with torch.no_grad():
                out = model.base_model.generate(
                    **inputs, max_new_tokens=40, do_sample=False,
                    pad_token_id=model.tokenizer.eos_token_id)
            # Decode ONLY the generated part (not the input prompt)
            actual = model.tokenizer.decode(out[0][input_len:], skip_special_tokens=True)

        if exact_match_contains(expected, actual):
            correct += 1
            per_cat_correct[cat] = per_cat_correct.get(cat, 0) + 1

    total = len(facts)
    per_category = {}
    for cat in per_cat_total:
        per_category[cat] = {
            "accuracy": per_cat_correct.get(cat, 0) / per_cat_total[cat],
            "total": per_cat_total[cat],
        }

    hit_rate = memory_hits / total if use_memory and total > 0 else 0
    return correct / max(total, 1), {"per_category": per_category, "memory_hit_rate": hit_rate}


# ── Visualisation ────────────────────────────────────────────────
