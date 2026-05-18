"""CounterFact to EasyEdit record normalization."""
from __future__ import annotations

from typing import Any


def easyedit_record(fact: dict[str, Any], locality_limit: int | None) -> dict[str, Any]:
    rw = fact["requested_rewrite"]
    subject = rw["subject"]
    prompt = rw["prompt"].format(subject)
    target_new = rw["target_new"]["str"]
    target_true = rw["target_true"]["str"]
    paraphrases = fact.get("paraphrase_prompts", [])
    neighbors = fact.get("neighborhood_prompts", [])
    if locality_limit is not None:
        neighbors = neighbors[:locality_limit]
    record = {
        "prompt": prompt,
        "target_new": target_new,
        "ground_truth": target_true,
        "subject": subject,
        "portability": {},
        "locality": {},
    }
    if paraphrases:
        record["rephrase_prompt"] = paraphrases[0]
        record["rephrase_prompts"] = paraphrases
    if neighbors:
        record["locality"]["neighborhood"] = {
            "prompt": neighbors,
            "ground_truth": [target_true for _ in neighbors],
        }
    portability = extract_portability(fact)
    if portability:
        record["portability"] = portability
    return record


def extract_portability(fact: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    """Normalize common EasyEdit/KnowEdit portability shapes into record form."""
    if isinstance(fact.get("portability"), dict):
        normalized: dict[str, dict[str, list[str]]] = {}
        for key, value in fact["portability"].items():
            prompts: list[str] = []
            answers: list[str] = []
            items = value if isinstance(value, list) else [value]
            for item in items:
                if not isinstance(item, dict):
                    continue
                prompt = (
                    item.get("prompt")
                    or item.get("New Question")
                    or item.get("question")
                )
                answer = (
                    item.get("ground_truth")
                    or item.get("New Answer")
                    or item.get("answer")
                )
                if isinstance(answer, list):
                    answer = answer[0][0] if answer and isinstance(answer[0], list) else answer[0]
                if prompt and answer:
                    prompts.append(str(prompt))
                    answers.append(str(answer))
            if prompts:
                normalized[key] = {"prompt": prompts, "ground_truth": answers}
        if normalized:
            return normalized

    prompt = (
        fact.get("portability_prompt")
        or fact.get("portability_prompts")
    )
    answer = (
        fact.get("portability_ground_truth")
        or fact.get("portability_answer")
        or fact.get("portability_answers")
    )
    if prompt and answer:
        prompts = prompt if isinstance(prompt, list) else [prompt]
        answers = answer if isinstance(answer, list) else [answer]
        answers = [
            a[0][0] if isinstance(a, list) and a and isinstance(a[0], list)
            else a[0] if isinstance(a, list) and a else a
            for a in answers
        ]
        return {
            "one_hop": {
                "prompt": [str(p) for p in prompts],
                "ground_truth": [str(a) for a in answers],
            }
        }
    return {}
