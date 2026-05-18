"""Metric helpers for AGIM EasyEdit-compatible evaluation."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def attach_locality_acc(pre: dict[str, Any], post: dict[str, Any],
                        record: dict[str, Any]) -> None:
    if "locality" not in post:
        return
    for locality_key in record.get("locality", {}):
        out_key = f"{locality_key}_output"
        acc_key = f"{locality_key}_acc"
        if out_key not in post["locality"] or out_key not in pre.get("locality", {}):
            continue
        locality_result = []
        for ans, label in zip(post["locality"][out_key], pre["locality"][out_key]):
            locality_result.append(float(np.mean(np.equal(ans, label))))
        post["locality"][acc_key] = locality_result
        post["locality"].pop(out_key, None)
    pre.pop("locality", None)


def official_generation_metrics(model, tok, hparams, test_prediction_acc,
                                record: dict[str, Any], device_id: int) -> dict[str, Any]:
    target_new = record["target_new"]
    rewrite = test_prediction_acc(
        model, tok, hparams, record["prompt"], target_new, device_id,
        vanilla_generation=True,
    )
    ret: dict[str, Any] = {"rewrite_acc": rewrite}
    if "rephrase_prompt" in record:
        ret["rephrase_acc"] = test_prediction_acc(
            model, tok, hparams, record["rephrase_prompt"], target_new,
            device_id, vanilla_generation=True,
        )
    rephrase_prompts = record.get("rephrase_prompts", [])
    if rephrase_prompts:
        ret["rephrase_all_acc"] = [
            float(np.mean(test_prediction_acc(
                model, tok, hparams, prompt, target_new, device_id,
                vanilla_generation=True,
            )))
            for prompt in rephrase_prompts
        ]
    return ret


def contextual_target_ids(tok, prompt: str, target: str) -> list[int]:
    """Token ids for the EasyEdit teacher-forcing continuation label."""
    prompt_ids = tok(prompt, return_tensors="pt").input_ids[0]
    full_ids = tok(f"{prompt} {target}", return_tensors="pt").input_ids[0]
    suffix = full_ids[len(prompt_ids):].detach().cpu().tolist()
    if suffix:
        return [int(tid) for tid in suffix]
    return [int(tid) for tid in tok.encode(target, add_special_tokens=False)]


def generation_token_acc(model, tok, prompt: str, target_ids: list[int],
                         device: str) -> list[float]:
    """Greedy generation exact-token accuracy against supplied target ids."""
    if not target_ids:
        return [0.0]
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=len(target_ids),
            pad_token_id=tok.eos_token_id,
            do_sample=False,
            use_cache=False,
        )
    generated_ids = gen.detach().cpu().tolist()[0][-len(target_ids):]
    return [float(np.mean(np.equal(target_ids, generated_ids)))]


def teacher_forcing_token_acc(model, tok, prompt: str, target: str,
                              device: str) -> list[float]:
    """Teacher-forced token accuracy for prompt + space + target suffix ids."""
    prompt_ids = tok(prompt, return_tensors="pt").input_ids[0]
    full = tok(f"{prompt} {target}", return_tensors="pt").to(device)
    full_ids = full.input_ids[0]
    start = len(prompt_ids)
    if start >= len(full_ids):
        return [0.0]
    with torch.no_grad():
        logits = model(**full).logits[0].float()
    scores = []
    for pos in range(start, len(full_ids)):
        pred_pos = pos - 1
        pred_id = int(torch.argmax(logits[pred_pos]).item())
        token_id = int(full_ids[pos].item())
        scores.append(float(pred_id == token_id))
    return [float(np.mean(scores))] if scores else [0.0]


def attach_rephrase_all_acc(model, tok, post: dict[str, Any],
                            record: dict[str, Any], device: str) -> None:
    prompts = record.get("rephrase_prompts", [])
    if not prompts:
        return
    post["rephrase_all_acc"] = [
        float(np.mean(teacher_forcing_token_acc(
            model, tok, prompt, record["target_new"], device
        )))
        for prompt in prompts
    ]


def contextual_generation_metrics(model, tok, record: dict[str, Any],
                                  device: str) -> dict[str, Any]:
    """Generation metric aligned to prompt + space + target tokenization."""
    target_new = record["target_new"]
    rewrite_ids = contextual_target_ids(tok, record["prompt"], target_new)
    ret: dict[str, Any] = {
        "rewrite_acc": generation_token_acc(
            model, tok, record["prompt"], rewrite_ids, device
        ),
        "rewrite_target_ids": rewrite_ids,
    }
    if "rephrase_prompt" in record:
        rephrase_ids = contextual_target_ids(tok, record["rephrase_prompt"], target_new)
        ret.update({
            "rephrase_acc": generation_token_acc(
                model, tok, record["rephrase_prompt"], rephrase_ids, device
            ),
            "rephrase_target_ids": rephrase_ids,
        })
    rephrase_prompts = record.get("rephrase_prompts", [])
    if rephrase_prompts:
        rephrase_all_ids = [
            contextual_target_ids(tok, prompt, target_new)
            for prompt in rephrase_prompts
        ]
        ret["rephrase_all_acc"] = [
            float(np.mean(generation_token_acc(model, tok, prompt, ids, device)))
            for prompt, ids in zip(rephrase_prompts, rephrase_all_ids)
        ]
        ret["rephrase_all_target_ids"] = rephrase_all_ids
    return ret


def edit_nt_metrics(editor, backup: dict[str, Any], eos_id: int | None) -> dict[str, Any]:
    """Report measured non-target diffs and intentionally edited row counts."""
    diffs = editor.measure_non_target_diffs()
    lm_backup = backup.get("lm_backup", {})
    emb_backup = backup.get("emb_backup", {})
    edited_lm_rows = set(lm_backup.keys())
    edited_embed_rows = set(emb_backup.keys())
    lm_norms = _edited_delta_norms(editor.model.lm_head.weight.data, lm_backup)
    emb_norms = _edited_delta_norms(editor.model.model.embed_tokens.weight.data, emb_backup)
    return {
        "lm_head_non_edited_max": round(diffs["lm_head_non_edited_max"], 8),
        "embed_non_edited_max": round(diffs["embed_non_edited_max"], 8),
        "lm_head_sampled_row_ids": sorted(editor._lm_nt_snapshot),
        "embed_sampled_row_ids": sorted(editor._emb_nt_snapshot),
        "edited_lm_rows_count": len(edited_lm_rows),
        "edited_embed_rows_count": len(edited_embed_rows),
        "edited_lm_delta_l2_mean": lm_norms["mean"],
        "edited_lm_delta_l2_max": lm_norms["max"],
        "edited_embed_delta_l2_mean": emb_norms["mean"],
        "edited_embed_delta_l2_max": emb_norms["max"],
        "eos_row_changed": bool(eos_id in edited_lm_rows) if eos_id is not None else False,
    }


def _edited_delta_norms(weight: torch.Tensor, backup: dict[int, torch.Tensor]) -> dict[str, float]:
    if not backup:
        return {"mean": 0.0, "max": 0.0}
    norms = []
    for row_id, before in backup.items():
        after = weight[row_id, :].detach().float().cpu()
        norms.append(float((after - before.detach().float().cpu()).norm().item()))
    return {
        "mean": round(float(np.mean(norms)), 6),
        "max": round(float(max(norms)), 6),
    }


def target_sequence_logprob(model, tok, prompt: str, target: str, device: str) -> float:
    """Teacher-forced sum log P(target tokens | prompt), EasyEdit spacing."""
    prompt_ids = tok(prompt, return_tensors="pt").input_ids[0]
    full = tok(f"{prompt} {target}", return_tensors="pt").to(device)
    full_ids = full.input_ids[0]
    start = len(prompt_ids)
    if start >= len(full_ids):
        return float("-inf")
    with torch.no_grad():
        logits = model(**full).logits[0].float()
        log_probs = torch.log_softmax(logits, dim=-1)
    total = 0.0
    for pos in range(start, len(full_ids)):
        pred_pos = pos - 1
        token_id = int(full_ids[pos].item())
        total += float(log_probs[pred_pos, token_id].item())
    return total


def probability_metrics(model, tok, record: dict[str, Any], device: str) -> dict[str, Any]:
    """CounterFact-style probability comparisons: new-vs-true and locality."""
    target_new = record["target_new"]
    ground_truth = record["ground_truth"]
    ret = {
        "rewrite_new_logprob": target_sequence_logprob(model, tok, record["prompt"], target_new, device),
        "rewrite_true_logprob": target_sequence_logprob(model, tok, record["prompt"], ground_truth, device),
    }
    ret["rewrite_acc"] = 1.0 if ret["rewrite_new_logprob"] > ret["rewrite_true_logprob"] else 0.0
    if "rephrase_prompt" in record:
        r_new = target_sequence_logprob(model, tok, record["rephrase_prompt"], target_new, device)
        r_true = target_sequence_logprob(model, tok, record["rephrase_prompt"], ground_truth, device)
        ret.update({"rephrase_new_logprob": r_new, "rephrase_true_logprob": r_true,
                    "rephrase_acc": 1.0 if r_new > r_true else 0.0})
    if record.get("rephrase_prompts"):
        all_new, all_true, all_acc = [], [], []
        for prompt in record["rephrase_prompts"]:
            new_lp = target_sequence_logprob(model, tok, prompt, target_new, device)
            true_lp = target_sequence_logprob(model, tok, prompt, ground_truth, device)
            all_new.append(new_lp); all_true.append(true_lp); all_acc.append(float(new_lp > true_lp))
        ret.update({"rephrase_all_new_logprob": all_new,
                    "rephrase_all_true_logprob": all_true,
                    "rephrase_all_acc": all_acc})
    locality_scores: dict[str, list[float]] = {}
    for loc_key, loc in record.get("locality", {}).items():
        scores = []
        for prompt, truth in zip(loc.get("prompt", []), loc.get("ground_truth", [])):
            true_lp = target_sequence_logprob(model, tok, prompt, truth, device)
            new_lp = target_sequence_logprob(model, tok, prompt, target_new, device)
            scores.append(1.0 if true_lp > new_lp else 0.0)
        if scores:
            locality_scores[f"{loc_key}_acc"] = scores
    if locality_scores:
        ret["locality"] = locality_scores
    return ret


def ngram_entropy(texts: list[str], ns: tuple[int, ...] = (2, 3),
                  weights: tuple[float, ...] = (2 / 3, 4 / 3)) -> float:
    """EasyEdit-style weighted 2/3-gram entropy for generated text."""
    values = []
    for text in texts:
        toks = text.split()
        entropies = []
        for n, weight in zip(ns, weights):
            if len(toks) < n:
                entropies.append(0.0)
                continue
            counts: dict[tuple[str, ...], int] = {}
            for i in range(len(toks) - n + 1):
                gram = tuple(toks[i:i + n])
                counts[gram] = counts.get(gram, 0) + 1
            total = sum(counts.values())
            entropy = -sum((count / total) * math.log(count / total, 2)
                           for count in counts.values())
            entropies.append(entropy * weight)
        values.append(float(np.mean(entropies)))
    return round(float(np.mean(values)), 6) if values else 0.0


def fluency_metrics(model, tok, prefixes: list[str], device: str,
                    max_new_tokens: int = 100) -> dict[str, float]:
    outputs = []
    for prompt in prefixes:
        inputs = tok(prompt, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        outputs.append(tok.decode(gen[0, input_len:], skip_special_tokens=True))
    return {"ngram_entropy": ngram_entropy(outputs)}
