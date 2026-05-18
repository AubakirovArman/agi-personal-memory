"""Run AGIM/WAL edits through the official EasyEdit evaluation functions.

This keeps the edit implementation local, but evaluates pre/post metrics with
the local EasyEdit source tree using the same locality comparison used by
EasyEdit BaseEditor.
"""
from __future__ import annotations

import argparse
import math
import importlib
import importlib.util
import json
import random
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from agim.eval.easyedit_counterfact import LLAMA, git_sha, load_dataset, select_facts
from agim.model.wal_dual_editor import WALDualLayerEditor


DEFAULT_EASYEDIT_ROOT = Path("/mnt/hf_model_weights/arman/3bit/sites/EasyEdit")


def _module(name: str, path: str | None = None) -> types.ModuleType:
    mod = types.ModuleType(name)
    if path is not None:
        mod.__path__ = [path]  # type: ignore[attr-defined]
    sys.modules[name] = mod
    return mod


def _install_easyedit_stubs(root: Path) -> None:
    """Install tiny stubs for EasyEdit imports not needed by text metrics."""
    _module("easyeditor", str(root / "easyeditor"))
    _module("easyeditor.evaluate", str(root / "easyeditor" / "evaluate"))
    _module("easyeditor.editors", str(root / "easyeditor" / "editors"))
    util_mod = _module("easyeditor.util", str(root / "easyeditor" / "util"))
    trainer_mod = _module("easyeditor.trainer", str(root / "easyeditor" / "trainer"))
    _module("easyeditor.models", str(root / "easyeditor" / "models"))
    _module("easyeditor.models.melo", str(root / "easyeditor" / "models" / "melo"))

    class HyperParams:
        pass

    class LORA:
        pass

    util_mod.HyperParams = HyperParams
    trainer_mod.nn = torch.nn

    melo_mod = _module("easyeditor.models.melo.melo")
    melo_mod.LORA = LORA

    gen_mod = _module("easyeditor.util.generate")

    def generate_fast(*_args, **_kwargs):
        raise RuntimeError("generate_fast is not needed for this EasyEdit run")

    gen_mod.generate_fast = generate_fast

    if importlib.util.find_spec("nltk") is None:
        nltk_mod = _module("nltk")

        def word_tokenize(text: str) -> list[str]:
            return text.split()

        def ngrams(tokens: list[str], n: int):
            return zip(*[tokens[i:] for i in range(n)])

        class FreqDist(dict):
            def __init__(self, items):
                super().__init__()
                for item in items:
                    self[item] = self.get(item, 0) + 1

        nltk_mod.word_tokenize = word_tokenize
        nltk_mod.ngrams = ngrams
        nltk_mod.FreqDist = FreqDist


def _load_source_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_easyedit_official(root: Path) -> tuple[Any, Any, Any]:
    """Load official EasyEdit metric functions without importing all plugins."""
    _install_easyedit_stubs(root)
    utils = _load_source_module(
        "easyeditor.evaluate.evaluate_utils",
        root / "easyeditor" / "evaluate" / "evaluate_utils.py",
    )
    evaluate = _load_source_module(
        "easyeditor.evaluate.evaluate",
        root / "easyeditor" / "evaluate" / "evaluate.py",
    )
    editors_utils = _load_source_module(
        "easyeditor.editors.utils",
        root / "easyeditor" / "editors" / "utils.py",
    )
    return evaluate.compute_edit_quality, utils.test_prediction_acc, editors_utils.summary_metrics


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
        # This matches EasyEdit's CounterFact examples, which use the first
        # paraphrase prompt for portability/rephrase reporting.
        record["rephrase_prompt"] = paraphrases[0]
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
    return ret


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

    rewrite_new = target_sequence_logprob(model, tok, record["prompt"], target_new, device)
    rewrite_true = target_sequence_logprob(model, tok, record["prompt"], ground_truth, device)
    ret: dict[str, Any] = {
        "rewrite_new_logprob": rewrite_new,
        "rewrite_true_logprob": rewrite_true,
        "rewrite_acc": 1.0 if rewrite_new > rewrite_true else 0.0,
    }
    if "rephrase_prompt" in record:
        rephrase_new = target_sequence_logprob(
            model, tok, record["rephrase_prompt"], target_new, device)
        rephrase_true = target_sequence_logprob(
            model, tok, record["rephrase_prompt"], ground_truth, device)
        ret.update({
            "rephrase_new_logprob": rephrase_new,
            "rephrase_true_logprob": rephrase_true,
            "rephrase_acc": 1.0 if rephrase_new > rephrase_true else 0.0,
        })
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
            entropy = 0.0
            for count in counts.values():
                p = count / total
                entropy -= p * math.log(p, 2)
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
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        outputs.append(tok.decode(gen[0, input_len:], skip_special_tokens=True))
    return {"ngram_entropy": ngram_entropy(outputs)}


def mean_metric(rows: list[dict[str, Any]], phase: str, key: str) -> float | None:
    values = []
    for row in rows:
        if key in row[phase]:
            values.append(float(np.mean(row[phase][key])))
    return round(float(np.mean(values)), 6) if values else None


def summarize_official(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"pre": {}, "post": {}}
    for phase in ("pre", "post"):
        for key in ("rewrite_acc", "rephrase_acc"):
            value = mean_metric(rows, phase, key)
            if value is not None:
                summary[phase][key] = value
    loc_values = []
    for row in rows:
        loc = row["post"].get("locality", {})
        if "neighborhood_acc" in loc:
            loc_values.append(float(np.mean(loc["neighborhood_acc"])))
    if loc_values:
        summary["post"]["locality"] = {
            "neighborhood_acc": round(float(np.mean(loc_values)), 6),
        }
    port_values = []
    for row in rows:
        for value in row["post"].get("portability", {}).values():
            if isinstance(value, list):
                port_values.append(float(np.mean(value)))
    if port_values:
        summary["post"]["portability"] = {
            "mean_acc": round(float(np.mean(port_values)), 6),
        }
    gen_rewrite = [float(np.mean(row["generation"]["rewrite_acc"])) for row in rows]
    gen_summary: dict[str, Any] = {
        "rewrite_acc": round(float(np.mean(gen_rewrite)), 6),
    }
    gen_rephrase = [
        float(np.mean(row["generation"]["rephrase_acc"]))
        for row in rows
        if "rephrase_acc" in row["generation"]
    ]
    if gen_rephrase:
        gen_summary["rephrase_acc"] = round(float(np.mean(gen_rephrase)), 6)
    summary["post_generation_vanilla"] = gen_summary
    contextual_rows = [
        row["contextual_generation"]
        for row in rows
        if "contextual_generation" in row
    ]
    if contextual_rows:
        ctx_summary = {
            "rewrite_acc": round(float(np.mean([
                np.mean(r["rewrite_acc"]) for r in contextual_rows
            ])), 6)
        }
        ctx_rephrase = [
            float(np.mean(row["rephrase_acc"]))
            for row in contextual_rows
            if "rephrase_acc" in row
        ]
        if ctx_rephrase:
            ctx_summary["rephrase_acc"] = round(float(np.mean(ctx_rephrase)), 6)
        summary["post_generation_contextual"] = ctx_summary
    prob_rows = [row["probability"] for row in rows if "probability" in row]
    if prob_rows:
        prob_summary = {
            "rewrite_acc": round(float(np.mean([r["rewrite_acc"] for r in prob_rows])), 6)
        }
        if "rephrase_acc" in prob_rows[0]:
            prob_summary["rephrase_acc"] = round(
                float(np.mean([r.get("rephrase_acc", 0.0) for r in prob_rows])), 6)
        prob_loc = []
        for row in prob_rows:
            for value in row.get("locality", {}).values():
                prob_loc.append(float(np.mean(value)))
        if prob_loc:
            prob_summary["locality_acc"] = round(float(np.mean(prob_loc)), 6)
        summary["post_probability"] = prob_summary
    fluency_rows = [row["fluency"]["ngram_entropy"] for row in rows if "fluency" in row]
    if fluency_rows:
        summary["post_fluency"] = {
            "ngram_entropy": round(float(np.mean(fluency_rows)), 6)
        }
    return summary


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def parse_device_id(device: str) -> int:
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    return int(device)


def post_edit_bundle(
    *,
    model,
    tok,
    hparams,
    compute_edit_quality,
    test_prediction_acc,
    record: dict[str, Any],
    pre: dict[str, Any],
    model_name: str,
    device_id: int,
    device: str,
    probability: bool,
    fluency: bool,
) -> dict[str, Any]:
    post = compute_edit_quality(
        model, model_name, hparams, tok, record, device_id, eval_metric="token_em"
    )
    attach_locality_acc(pre, post, record)
    row: dict[str, Any] = {
        "pre": pre,
        "post": post,
        "generation": official_generation_metrics(
            model, tok, hparams, test_prediction_acc, record, device_id
        ),
        "contextual_generation": contextual_generation_metrics(
            model, tok, record, device
        ),
    }
    if probability:
        row["probability"] = probability_metrics(model, tok, record, device)
    if fluency:
        row["fluency"] = fluency_metrics(model, tok, [record["prompt"]], device)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--model", default=LLAMA)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--dataset", default="https://rome.baulab.info/data/dsets/counterfact.json")
    parser.add_argument("--sample-policy", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/easyedit_official_50.json")
    parser.add_argument("--easyedit-root", type=Path, default=DEFAULT_EASYEDIT_ROOT)
    parser.add_argument("--locality-limit", type=int, default=0,
                        help="0 means all official CounterFact locality prompts")
    parser.add_argument("--clamp_lm", type=float, default=0.20)
    parser.add_argument("--clamp_embed", type=float, default=0.06)
    parser.add_argument("--clamp_eos", type=float, default=0.16)
    parser.add_argument("--clamp_anti", type=float, default=0.06)
    parser.add_argument("--clamp_old", type=float, default=0.0)
    parser.add_argument("--target-token-mode", choices=["standalone", "contextual", "both"],
                        default="contextual",
                        help="contextual edits EasyEdit prompt + space + target label ids")
    parser.add_argument("--use-neg-prompts", action=argparse.BooleanOptionalAction, default=True,
                        help="Project edit key away from locality/neighborhood prompt keys")
    parser.add_argument("--neg-prompt-limit", type=int, default=10)
    parser.add_argument("--probability-metrics", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Compute P(target_new)>P(target_true) style metrics")
    parser.add_argument("--test-fluency", action="store_true",
                        help="Compute EasyEdit-style n-gram entropy on post-edit generations")
    parser.add_argument("--sequential-edit", action="store_true",
                        help="Apply all edits first, then evaluate without per-case rollback")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-easyedit-log", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device_id = parse_device_id(args.device)
    locality_limit = None if args.locality_limit == 0 else args.locality_limit

    compute_edit_quality, test_prediction_acc, summary_metrics = load_easyedit_official(
        args.easyedit_root
    )
    hparams = SimpleNamespace(alg_name="AGIM_WAL", max_length=512)

    print(f"Loading {args.model} on {args.device}...")
    tok = AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        local_files_only=args.local_files_only,
    )
    model.eval()

    editor = WALDualLayerEditor(model, tok, device=args.device)
    editor.build_vocab()

    all_facts, dataset_sha256 = load_dataset(args.dataset)
    facts = select_facts(all_facts, args.n, args.sample_policy, args.seed)
    records = [easyedit_record(fact, locality_limit) for fact in facts]

    print(
        f"\nOfficial EasyEdit metric run ({len(records)} facts, "
        f"locality={'all' if locality_limit is None else locality_limit}, "
        f"target_token_mode={args.target_token_mode}):\n",
        flush=True,
    )
    t0 = time.time()
    metrics: list[dict[str, Any]] = []
    edit_times: list[float] = []

    def apply_one(fact: dict[str, Any], record: dict[str, Any]) -> tuple[dict, float]:
        rw = fact["requested_rewrite"]
        neg_prompts = None
        if args.use_neg_prompts:
            neg_prompts = record.get("locality", {}).get("neighborhood", {}).get("prompt")
        start = time.time()
        backup = editor.apply_edit(
            rw["subject"],
            rw["target_new"]["str"],
            rw["relation_id"],
            prompt=record["prompt"],
            clamp_lm=args.clamp_lm,
            clamp_embed=args.clamp_embed,
            clamp_eos=args.clamp_eos,
            clamp_anti=args.clamp_anti,
            old_target=rw["target_true"]["str"],
            clamp_old=args.clamp_old,
            target_token_mode=args.target_token_mode,
            neg_prompts=neg_prompts,
            max_neg_prompts=args.neg_prompt_limit,
        )
        return backup, time.time() - start

    if args.sequential_edit:
        pres = [
            compute_edit_quality(
                model, args.model, hparams, tok, record, device_id,
                eval_metric="token_em")
            for record in records
        ]
        backups = []
        for fact, record in zip(facts, records):
            backup, edit_time = apply_one(fact, record)
            backups.append(backup)
            edit_times.append(edit_time)
        for idx, (fact, record, pre) in enumerate(zip(facts, records, pres)):
            row = {
                "case_id": fact.get("case_id", idx),
                "requested_rewrite": record,
                "edit_time_s": round(edit_times[idx], 4),
            }
            row.update(post_edit_bundle(
                model=model,
                tok=tok,
                hparams=hparams,
                compute_edit_quality=compute_edit_quality,
                test_prediction_acc=test_prediction_acc,
                record=record,
                pre=pre,
                model_name=args.model,
                device_id=device_id,
                device=args.device,
                probability=args.probability_metrics,
                fluency=args.test_fluency,
            ))
            metrics.append(jsonable(row))
        for backup in reversed(backups):
            editor.rollback(backup)
    else:
        for idx, (fact, record) in enumerate(zip(facts, records)):
            pre = compute_edit_quality(
                model, args.model, hparams, tok, record, device_id,
                eval_metric="token_em"
            )
            backup, edit_time = apply_one(fact, record)
            row = {
                "case_id": fact.get("case_id", idx),
                "requested_rewrite": record,
                "edit_time_s": round(edit_time, 4),
            }
            row.update(post_edit_bundle(
                model=model,
                tok=tok,
                hparams=hparams,
                compute_edit_quality=compute_edit_quality,
                test_prediction_acc=test_prediction_acc,
                record=record,
                pre=pre,
                model_name=args.model,
                device_id=device_id,
                device=args.device,
                probability=args.probability_metrics,
                fluency=args.test_fluency,
            ))
            editor.rollback(backup)

            metrics.append(jsonable(row))
            if (idx + 1) % 10 == 0 or idx + 1 == len(records):
                summary = summarize_official(metrics)
                post_summary = summary["post"]
                loc = post_summary.get("locality", {}).get("neighborhood_acc")
                gen = summary["post_generation_vanilla"]
                ctx_gen = summary.get("post_generation_contextual", {})
                print(
                    f"  [{idx + 1}/{len(records)}] "
                    f"TF rewrite={post_summary.get('rewrite_acc', 0):.1%} "
                    f"TF rephrase={post_summary.get('rephrase_acc', 0):.1%} "
                    f"TF loc={0.0 if loc is None else loc:.1%} "
                    f"GEN rewrite={gen['rewrite_acc']:.1%} "
                    f"CTX-GEN rewrite={ctx_gen.get('rewrite_acc', 0):.1%}",
                    flush=True,
                )

    elapsed = time.time() - t0
    summary = summarize_official(metrics)

    if args.write_easyedit_log:
        summary_metrics(metrics)

    payload = {
        "n": len(metrics),
        "model": args.model,
        "device": args.device,
        "git_sha": git_sha(),
        "command": " ".join(sys.argv),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "easyedit": {
            "root": str(args.easyedit_root),
            "functions": [
                "easyeditor.evaluate.evaluate.compute_edit_quality",
                "easyeditor.evaluate.evaluate_utils.test_prediction_acc",
            ],
            "aggregation": "EasyEdit BaseEditor-style pre/post locality comparison",
            "teacher_forcing_metric": "token_em",
            "generation_metric": "vanilla_generation token equality",
            "contextual_generation_metric": (
                "greedy token equality against prompt + space + target suffix ids"
            ),
        },
        "dataset": {
            "source": args.dataset,
            "sha256": dataset_sha256,
            "total_size": len(all_facts),
            "sample_policy": args.sample_policy,
            "seed": args.seed,
            "case_ids": [fact.get("case_id") for fact in facts],
            "locality_prompts": "all" if locality_limit is None else locality_limit,
            "rephrase_prompt": "first",
        },
        "hyperparams": {
            "clamp_lm": args.clamp_lm,
            "clamp_embed": args.clamp_embed,
            "clamp_eos": args.clamp_eos,
            "clamp_anti": args.clamp_anti,
            "clamp_old": args.clamp_old,
            "target_token_mode": args.target_token_mode,
            "use_neg_prompts": args.use_neg_prompts,
            "neg_prompt_limit": args.neg_prompt_limit,
            "probability_metrics": args.probability_metrics,
            "test_fluency": args.test_fluency,
            "sequential_edit": args.sequential_edit,
        },
        "summary": summary,
        "time_s": round(elapsed, 2),
        "time_per_edit_s": round(elapsed / max(len(metrics), 1), 4),
        "metrics": metrics,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False))

    post = summary["post"]
    loc = post.get("locality", {}).get("neighborhood_acc")
    gen = summary["post_generation_vanilla"]
    print(f"\n{'=' * 64}")
    print(f"OFFICIAL EASYEDIT RESULTS ({len(metrics)} facts)")
    print(f"{'=' * 64}")
    print(
        "  Teacher-forcing: "
        f"rewrite={post.get('rewrite_acc', 0):.1%} "
        f"rephrase={post.get('rephrase_acc', 0):.1%} "
        f"locality={0.0 if loc is None else loc:.1%}"
    )
    print(
        "  Vanilla generation: "
        f"rewrite={gen['rewrite_acc']:.1%} "
        f"rephrase={gen.get('rephrase_acc', 0):.1%}"
    )
    if "post_generation_contextual" in summary:
        ctx_gen = summary["post_generation_contextual"]
        print(
            "  Contextual generation: "
            f"rewrite={ctx_gen['rewrite_acc']:.1%} "
            f"rephrase={ctx_gen.get('rephrase_acc', 0):.1%}"
        )
    if "post_probability" in summary:
        prob = summary["post_probability"]
        print(
            "  Probability compare: "
            f"rewrite={prob.get('rewrite_acc', 0):.1%} "
            f"rephrase={prob.get('rephrase_acc', 0):.1%} "
            f"locality={prob.get('locality_acc', 0):.1%}"
        )
    if "portability" in post:
        print(f"  Portability: mean={post['portability'].get('mean_acc', 0):.1%}")
    if "post_fluency" in summary:
        print(
            "  Fluency: "
            f"ngram_entropy={summary['post_fluency']['ngram_entropy']:.3f}"
        )
    print(f"  Time: {elapsed:.1f}s ({elapsed / max(len(metrics), 1):.2f}s/edit)")
    print(f"\nSaved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
