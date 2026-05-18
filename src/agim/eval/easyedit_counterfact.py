"""Canonical local CounterFact evaluator for WAL weight editing."""
from __future__ import annotations

import time
from typing import Any

import torch

from .counterfact_data import LLAMA, git_sha, load_dataset, select_facts
from .counterfact_protocols import (
    has_repetition,
    score_protocol_after_edit,
    token_overlap,
)
from .counterfact_summary import group_summary, summarize_nt, summarize_protocol


class CounterFactEvaluator:
    """CounterFact evaluator for dual-layer WAL editing."""

    PROTOCOLS = {
        "easyedit_strict": 1.0,
        "agim_practical": 1.2,
    }

    def __init__(self, model, tok, editor, device: str = "cuda"):
        self.model = model
        self.tok = tok
        self.editor = editor
        self.device = device

    def generate(self, prompt: str, max_tokens: int = 10,
                 rep_penalty: float = 1.0) -> str:
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                repetition_penalty=rep_penalty,
                pad_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(out[0][input_len:], skip_special_tokens=True).strip()

    def generate_ids(self, prompt: str, max_tokens: int,
                     rep_penalty: float = 1.0) -> list[int]:
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                repetition_penalty=rep_penalty,
                pad_token_id=self.tok.eos_token_id,
            )
        return out[0, input_len:].detach().cpu().tolist()

    def truncate_prompt(self, prompt: str, max_tokens: int | None = None) -> str:
        """Token-based prompt truncation; avoids cutting words by characters."""
        if max_tokens is None or max_tokens <= 0:
            return prompt
        ids = self.tok.encode(prompt, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return prompt
        return self.tok.decode(ids[:max_tokens], skip_special_tokens=True)

    @staticmethod
    def token_overlap(a: str, b: str) -> float:
        return token_overlap(a, b)

    @staticmethod
    def has_repetition(text: str, target: str, threshold: int = 2) -> bool:
        return has_repetition(text, target, threshold)

    def _score_prompt(self, prompt: str, target: str, target_ids: list[int],
                      rep_penalty: float) -> dict[str, Any]:
        text = self.generate(prompt, max_tokens=10, rep_penalty=rep_penalty)
        gen_ids = self.generate_ids(
            prompt, max_tokens=len(target_ids), rep_penalty=rep_penalty)
        token_exact = gen_ids == target_ids
        substring = target.lower() in text.lower()
        return {
            "text": text,
            "generated_ids": gen_ids,
            "token_exact": 1.0 if token_exact else 0.0,
            "substring": 1.0 if substring else 0.0,
            "clean": 1.0 if substring and not self.has_repetition(text, target) else 0.0,
            "has_repetition": self.has_repetition(text, target),
        }

    def evaluate_one(self, fact: dict[str, Any], *, protocols: list[str],
                     clamp_lm: float = 0.20, clamp_embed: float = 0.06,
                     clamp_eos: float = 0.0, clamp_anti: float = 0.06,
                     clamp_old: float = 0.0,
                     target_token_mode: str = "standalone",
                     neg_projection_strength: float = 0.3,
                     history_projection_strength: float = 0.0,
                     embed_history_projection_strength: float = 0.0,
                     max_history_keys: int = 128,
                     neighbor_limit: int = 0) -> dict[str, Any] | None:
        context = _fact_context(self, fact, neighbor_limit)
        before = _pre_edit_outputs(self, context, protocols)
        start = time.time()
        backup = self.editor.apply_edit(
            context["subject"],
            context["target_new"],
            context["relation"],
            prompt=context["prompt"],
            clamp_lm=clamp_lm,
            clamp_embed=clamp_embed,
            clamp_eos=clamp_eos,
            clamp_anti=clamp_anti,
            old_target=context["target_true"],
            clamp_old=clamp_old,
            target_token_mode=target_token_mode,
            neg_projection_strength=neg_projection_strength,
            history_projection_strength=history_projection_strength,
            embed_history_projection_strength=embed_history_projection_strength,
            max_history_keys=max_history_keys,
        )
        edit_time_s = time.time() - start
        if backup is None:
            return None
        protocol_results = _post_edit_scores(self, context, protocols, before)
        nt = self.editor.measure_non_target_diffs()
        edited_lm_rows = set(backup.get("lm_backup", {}).keys())
        edited_embed_rows = set(backup.get("emb_backup", {}).keys())
        self.editor.rollback(backup)
        _attach_rollback_scores(self, context, protocols, before, protocol_results)
        return _result_row(
            fact, context, protocol_results, nt, edited_lm_rows,
            edited_embed_rows, self.tok.eos_token_id, edit_time_s,
        )

    def evaluate_all(self, facts: list[dict[str, Any]], **kwargs) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        t0 = time.time()
        for idx, fact in enumerate(facts):
            result = self.evaluate_one(fact, **kwargs)
            if result:
                results.append(result)
            if (idx + 1) % 25 == 0:
                _print_progress(results, idx, t0)
        return results


def _fact_context(evaluator, fact: dict[str, Any], neighbor_limit: int) -> dict[str, Any]:
    rw = fact["requested_rewrite"]
    subject = rw["subject"]
    target_new = rw["target_new"]["str"]
    target_true = rw["target_true"]["str"]
    prompt = rw["prompt"].format(subject)
    neighbors = fact.get("neighborhood_prompts", [])
    if neighbor_limit > 0:
        neighbors = neighbors[:neighbor_limit]
    return {
        "subject": subject,
        "relation": rw["relation_id"],
        "target_new": target_new,
        "target_true": target_true,
        "prompt": prompt,
        "target_ids": evaluator.tok.encode(target_new, add_special_tokens=False),
        "subject_ids": evaluator.tok.encode(subject, add_special_tokens=False),
        "paraphrases": fact.get("paraphrase_prompts", []),
        "neighbor_prompts": neighbors,
    }


def _pre_edit_outputs(evaluator, context: dict[str, Any],
                      protocols: list[str]) -> dict[str, dict[str, Any]]:
    before: dict[str, dict[str, Any]] = {}
    for name in protocols:
        rp = evaluator.PROTOCOLS[name]
        before[name] = {
            "direct": evaluator.generate(context["prompt"], rep_penalty=rp),
            "neighbors": [
                evaluator.generate(evaluator.truncate_prompt(prompt, 100), rep_penalty=rp)
                for prompt in context["neighbor_prompts"]
            ],
        }
    return before


def _post_edit_scores(evaluator, context: dict[str, Any],
                      protocols: list[str], before: dict) -> dict[str, dict[str, Any]]:
    protocol_results: dict[str, dict[str, Any]] = {}
    for name in protocols:
        rp = evaluator.PROTOCOLS[name]
        protocol_results[name] = score_protocol_after_edit(
            evaluator,
            prompt=context["prompt"],
            target_new=context["target_new"],
            target_true=context["target_true"],
            target_ids=context["target_ids"],
            paraphrases=context["paraphrases"],
            neighbor_prompts=context["neighbor_prompts"],
            before_direct=before[name]["direct"],
            neighbors_before=before[name]["neighbors"],
            rep_penalty=rp,
        )
    return protocol_results


def _attach_rollback_scores(evaluator, context: dict[str, Any], protocols: list[str],
                            before: dict, protocol_results: dict) -> None:
    for name in protocols:
        rp = evaluator.PROTOCOLS[name]
        after_rollback = evaluator.generate(context["prompt"], rep_penalty=rp)
        protocol_results[name].update({
            "RB_old_target": (
                1.0 if context["target_true"].lower() in after_rollback.lower() else 0.0
            ),
            "RB_consistency": (
                1.0 if before[name]["direct"].strip() == after_rollback.strip() else 0.0
            ),
            "RB_overlap": round(token_overlap(before[name]["direct"], after_rollback), 4),
            "after_rollback": after_rollback[:120],
        })


def _result_row(fact, context, protocol_results, nt, edited_lm_rows,
                edited_embed_rows, eos_id, edit_time_s) -> dict[str, Any]:
    strict = protocol_results.get("easyedit_strict", {})
    practical = protocol_results.get("agim_practical", {})
    return {
        "case_id": fact.get("case_id"),
        "subject": context["subject"],
        "relation": context["relation"],
        "prompt": context["prompt"],
        "target_new": context["target_new"],
        "target_true": context["target_true"],
        "target_token_count": len(context["target_ids"]),
        "subject_token_count": len(context["subject_ids"]),
        "num_paraphrases": len(context["paraphrases"]),
        "num_neighbors": len(context["neighbor_prompts"]),
        "protocols": protocol_results,
        "easyedit_strict": strict,
        "agim_practical": practical,
        "NT": {
            "lm_head_non_edited_max": round(nt["lm_head_non_edited_max"], 8),
            "embed_non_edited_max": round(nt["embed_non_edited_max"], 8),
            "edited_lm_rows_count": len(edited_lm_rows),
            "edited_embed_rows_count": len(edited_embed_rows),
            "eos_row_changed": bool(eos_id in edited_lm_rows),
        },
        "edit_time_s": round(edit_time_s, 4),
        "ES_easyedit": strict.get("ES_token"),
        "PS_easyedit_2": strict.get("PS_token_2"),
        "PS_easyedit_all": strict.get("PS_token_all"),
        "ES_agim": practical.get("ES_substring"),
        "PS_agim_2": practical.get("PS_substring_2"),
        "PS_agim_all": practical.get("PS_substring_all"),
        "NS_absence": strict.get("NS_absence"),
        "NS_consistency": strict.get("NS_consistency"),
        "NS_overlap": strict.get("NS_overlap"),
    }


def _print_progress(results: list[dict[str, Any]], idx: int, t0: float) -> None:
    summary = summarize_protocol(results, "easyedit_strict")
    print(
        f"  [{len(results)}/{idx + 1}] "
        f"ES_ee={summary['ES_token']:.0%} "
        f"PS_ee@2={summary['PS_token_2']:.0%} "
        f"NS_abs={summary['NS_absence']:.0%} "
        f"Comp={summary['Composite_token_absence']:.1%} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )


def main() -> int:
    from .counterfact_cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
