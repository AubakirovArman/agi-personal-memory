"""Scoring helpers for the local CounterFact evaluator."""
from __future__ import annotations

import re
from typing import Any


def token_overlap(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    return len(ta & tb) / max(len(ta), len(tb)) if ta and tb else 0.0


def has_repetition(text: str, target: str, threshold: int = 2) -> bool:
    escaped = re.escape(target.lower())
    return bool(re.search(rf"({escaped})\1{{{threshold},}}", text.lower()))


def score_protocol_after_edit(
    evaluator,
    *,
    prompt: str,
    target_new: str,
    target_true: str,
    target_ids: list[int],
    paraphrases: list[str],
    neighbor_prompts: list[str],
    before_direct: str,
    neighbors_before: list[str],
    rep_penalty: float,
) -> dict[str, Any]:
    direct = evaluator._score_prompt(prompt, target_new, target_ids, rep_penalty)
    ps_sub_2, ps_tok_2, ps_sub_all, ps_tok_all = _score_paraphrases(
        evaluator, paraphrases, target_new, target_ids, rep_penalty)
    ns_absence, ns_consistency, ns_overlap = _score_neighbors(
        evaluator, neighbor_prompts, neighbors_before, target_new, rep_penalty)
    ps_n2 = min(2, len(paraphrases))
    ps_nall = len(paraphrases)
    return {
        "rep_penalty": rep_penalty,
        "ES_token": direct["token_exact"],
        "ES_substring": direct["substring"],
        "ES_clean": direct["clean"],
        "PS_token_2": ps_tok_2,
        "PS_substring_2": ps_sub_2,
        "PS_token_all": ps_tok_all,
        "PS_substring_all": ps_sub_all,
        "PS_n2": ps_n2,
        "PS_nall": ps_nall,
        "NS_absence": ns_absence,
        "NS_consistency": ns_consistency,
        "NS_overlap": round(ns_overlap, 4),
        "has_repetition": direct["has_repetition"],
        "gen_direct": direct["text"][:120],
        "before_direct": before_direct[:120],
        "target_true": target_true,
    }


def _score_paraphrases(evaluator, paraphrases, target_new, target_ids, rep_penalty):
    ps_sub_2 = ps_tok_2 = ps_sub_all = ps_tok_all = 0.0
    ps_n2 = min(2, len(paraphrases))
    ps_nall = len(paraphrases)
    for idx, para in enumerate(paraphrases):
        scored = evaluator._score_prompt(
            evaluator.truncate_prompt(para, 100), target_new, target_ids, rep_penalty)
        ps_sub_all += scored["substring"]
        ps_tok_all += scored["token_exact"]
        if idx < 2:
            ps_sub_2 += scored["substring"]
            ps_tok_2 += scored["token_exact"]
    return (
        ps_sub_2 / max(ps_n2, 1),
        ps_tok_2 / max(ps_n2, 1),
        ps_sub_all / max(ps_nall, 1),
        ps_tok_all / max(ps_nall, 1),
    )


def _score_neighbors(evaluator, neighbor_prompts, neighbors_before,
                     target_new, rep_penalty):
    ns_absence = ns_consistency = ns_overlap = 0.0
    for idx, n_prompt in enumerate(neighbor_prompts):
        after = evaluator.generate(
            evaluator.truncate_prompt(n_prompt, 100), rep_penalty=rep_penalty)
        if target_new.lower() not in after.lower():
            ns_absence += 1.0
        overlap = token_overlap(neighbors_before[idx], after)
        if overlap > 0.3:
            ns_consistency += 1.0
        ns_overlap += overlap
    ns_n = len(neighbor_prompts)
    return (
        ns_absence / max(ns_n, 1),
        ns_consistency / max(ns_n, 1),
        ns_overlap / max(ns_n, 1),
    )
