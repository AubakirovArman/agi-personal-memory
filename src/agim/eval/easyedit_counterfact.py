"""Canonical EasyEdit-compatible CounterFact evaluator.

This evaluator records two protocols for the same edit:
- easyedit_strict: greedy, repetition_penalty=1.0, token-exact ES/PS
- agim_practical: greedy, repetition_penalty=1.2, token-exact and substring metrics

It also records PS@2, PS@All, three NS metrics, rollback before/after metrics,
and non-edited row diff for both lm_head and embed_tokens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agim.model.wal_dual_editor import WALDualLayerEditor

LLAMA = "meta-llama/Llama-3.1-8B-Instruct"


class CounterFactEvaluator:
    """CounterFact evaluator for dual-layer WAL editing."""

    PROTOCOLS = {
        "easyedit_strict": 1.0,
        "agim_practical": 1.2,
    }

    def __init__(self, model, tok, editor, device: str = "cuda:3"):
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
        ta = set(a.lower().split())
        tb = set(b.lower().split())
        return len(ta & tb) / max(len(ta), len(tb)) if ta and tb else 0.0

    @staticmethod
    def has_repetition(text: str, target: str, threshold: int = 2) -> bool:
        import re

        escaped = re.escape(target.lower())
        return bool(re.search(rf"({escaped})\1{{{threshold},}}", text.lower()))

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

    def _score_protocol_after_edit(
        self,
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
        direct = self._score_prompt(prompt, target_new, target_ids, rep_penalty)

        ps_sub_2 = ps_tok_2 = ps_sub_all = ps_tok_all = 0.0
        ps_n2 = min(2, len(paraphrases))
        ps_nall = len(paraphrases)
        for idx, para in enumerate(paraphrases):
            scored = self._score_prompt(
                self.truncate_prompt(para, 100), target_new, target_ids, rep_penalty)
            ps_sub_all += scored["substring"]
            ps_tok_all += scored["token_exact"]
            if idx < 2:
                ps_sub_2 += scored["substring"]
                ps_tok_2 += scored["token_exact"]

        ps_sub_2 /= max(ps_n2, 1)
        ps_tok_2 /= max(ps_n2, 1)
        ps_sub_all /= max(ps_nall, 1)
        ps_tok_all /= max(ps_nall, 1)

        ns_absence = ns_consistency = ns_overlap = 0.0
        for idx, n_prompt in enumerate(neighbor_prompts):
            after = self.generate(
                self.truncate_prompt(n_prompt, 100), rep_penalty=rep_penalty)
            if target_new.lower() not in after.lower():
                ns_absence += 1.0
            overlap = self.token_overlap(neighbors_before[idx], after)
            if overlap > 0.3:
                ns_consistency += 1.0
            ns_overlap += overlap
        ns_n = len(neighbor_prompts)
        ns_absence /= max(ns_n, 1)
        ns_consistency /= max(ns_n, 1)
        ns_overlap /= max(ns_n, 1)

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

    def evaluate_one(self, fact: dict[str, Any], *, protocols: list[str],
                     clamp_lm: float = 0.20, clamp_embed: float = 0.06,
                     clamp_eos: float = 0.16, clamp_anti: float = 0.06,
                     clamp_old: float = 0.0,
                     target_token_mode: str = "standalone",
                     neg_projection_strength: float = 0.3,
                     history_projection_strength: float = 0.0,
                     embed_history_projection_strength: float = 0.0,
                     max_history_keys: int = 128,
                     neighbor_limit: int = 0) -> dict[str, Any] | None:
        rw = fact["requested_rewrite"]
        subject = rw["subject"]
        relation = rw["relation_id"]
        target_new = rw["target_new"]["str"]
        target_true = rw["target_true"]["str"]
        prompt = rw["prompt"].format(subject)
        target_ids = self.tok.encode(target_new, add_special_tokens=False)
        subject_ids = self.tok.encode(subject, add_special_tokens=False)
        paraphrases = fact.get("paraphrase_prompts", [])
        neighbor_prompts = fact.get("neighborhood_prompts", [])
        if neighbor_limit > 0:
            neighbor_prompts = neighbor_prompts[:neighbor_limit]

        before: dict[str, dict[str, Any]] = {}
        for name in protocols:
            rp = self.PROTOCOLS[name]
            before[name] = {
                "direct": self.generate(prompt, rep_penalty=rp),
                "neighbors": [
                    self.generate(self.truncate_prompt(n_prompt, 100), rep_penalty=rp)
                    for n_prompt in neighbor_prompts
                ],
            }

        start = time.time()
        backup = self.editor.apply_edit(
            subject,
            target_new,
            relation,
            prompt=prompt,
            clamp_lm=clamp_lm,
            clamp_embed=clamp_embed,
            clamp_eos=clamp_eos,
            clamp_anti=clamp_anti,
            old_target=target_true,
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

        protocol_results: dict[str, dict[str, Any]] = {}
        for name in protocols:
            rp = self.PROTOCOLS[name]
            protocol_results[name] = self._score_protocol_after_edit(
                prompt=prompt,
                target_new=target_new,
                target_true=target_true,
                target_ids=target_ids,
                paraphrases=paraphrases,
                neighbor_prompts=neighbor_prompts,
                before_direct=before[name]["direct"],
                neighbors_before=before[name]["neighbors"],
                rep_penalty=rp,
            )

        nt = self.editor.measure_non_target_diffs()
        edited_lm_rows = set(backup.get("lm_backup", {}).keys())
        edited_embed_rows = set(backup.get("emb_backup", {}).keys())
        eos_id = self.tok.eos_token_id

        self.editor.rollback(backup)

        for name in protocols:
            rp = self.PROTOCOLS[name]
            after_rollback = self.generate(prompt, rep_penalty=rp)
            protocol_results[name].update({
                "RB_old_target": (
                    1.0 if target_true.lower() in after_rollback.lower() else 0.0
                ),
                "RB_consistency": (
                    1.0 if before[name]["direct"].strip() == after_rollback.strip()
                    else 0.0
                ),
                "RB_overlap": round(
                    self.token_overlap(before[name]["direct"], after_rollback), 4),
                "after_rollback": after_rollback[:120],
            })

        strict = protocol_results.get("easyedit_strict", {})
        practical = protocol_results.get("agim_practical", {})

        return {
            "case_id": fact.get("case_id"),
            "subject": subject,
            "relation": relation,
            "prompt": prompt,
            "target_new": target_new,
            "target_true": target_true,
            "target_token_count": len(target_ids),
            "subject_token_count": len(subject_ids),
            "num_paraphrases": len(paraphrases),
            "num_neighbors": len(neighbor_prompts),
            "protocols": protocol_results,
            "easyedit_strict": strict,
            "agim_practical": practical,
            "NT": {
                "lm_head_non_edited_max": round(
                    nt["lm_head_non_edited_max"], 8),
                "embed_non_edited_max": round(
                    nt["embed_non_edited_max"], 8),
                "edited_lm_rows_count": len(edited_lm_rows),
                "edited_embed_rows_count": len(edited_embed_rows),
                "eos_row_changed": bool(eos_id in edited_lm_rows),
            },
            "edit_time_s": round(edit_time_s, 4),
            # Legacy flat fields for quick grep/backward analysis.
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

    def evaluate_all(self, facts: list[dict[str, Any]], **kwargs) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        t0 = time.time()
        for idx, fact in enumerate(facts):
            result = self.evaluate_one(fact, **kwargs)
            if result:
                results.append(result)
            if (idx + 1) % 25 == 0:
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
        return results


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_protocol(results: list[dict[str, Any]], protocol: str) -> dict[str, float]:
    rows = [r["protocols"][protocol] for r in results if protocol in r["protocols"]]
    es_token = _mean([r["ES_token"] for r in rows])
    es_sub = _mean([r["ES_substring"] for r in rows])
    ps_token_2 = _mean([r["PS_token_2"] for r in rows])
    ps_sub_2 = _mean([r["PS_substring_2"] for r in rows])
    ns_abs = _mean([r["NS_absence"] for r in rows])
    ns_con = _mean([r["NS_consistency"] for r in rows])
    summary = {
        "ES_token": es_token,
        "ES_substring": es_sub,
        "ES_clean": _mean([r["ES_clean"] for r in rows]),
        "PS_token_2": ps_token_2,
        "PS_substring_2": ps_sub_2,
        "PS_token_all": _mean([r["PS_token_all"] for r in rows]),
        "PS_substring_all": _mean([r["PS_substring_all"] for r in rows]),
        "NS_absence": ns_abs,
        "NS_consistency": ns_con,
        "NS_overlap": _mean([r["NS_overlap"] for r in rows]),
        "Composite_token_absence": (es_token + ps_token_2 + ns_abs) / 3,
        "Composite_token_consistency": (es_token + ps_token_2 + ns_con) / 3,
        "Composite_substring_absence": (es_sub + ps_sub_2 + ns_abs) / 3,
        "Composite_substring_consistency": (es_sub + ps_sub_2 + ns_con) / 3,
        "repetition_rate": _mean([1.0 if r["has_repetition"] else 0.0 for r in rows]),
        "RB_old_target": _mean([r["RB_old_target"] for r in rows]),
        "RB_consistency": _mean([r["RB_consistency"] for r in rows]),
        "RB_overlap": _mean([r["RB_overlap"] for r in rows]),
    }
    return {k: round(v, 6) for k, v in summary.items()}


def summarize_nt(results: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "lm_head_non_edited_max": round(
            max((r["NT"]["lm_head_non_edited_max"] for r in results), default=0.0), 8),
        "embed_non_edited_max": round(
            max((r["NT"]["embed_non_edited_max"] for r in results), default=0.0), 8),
        "edited_lm_rows_avg": round(
            _mean([r["NT"]["edited_lm_rows_count"] for r in results]), 4),
        "edited_embed_rows_avg": round(
            _mean([r["NT"]["edited_embed_rows_count"] for r in results]), 4),
        "eos_row_changed_rate": round(
            _mean([1.0 if r["NT"]["eos_row_changed"] else 0.0 for r in results]), 6),
    }


def _token_bucket(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 3:
        return "2-3"
    return "4+"


def _group_summary(results: list[dict[str, Any]], key: str,
                   protocol: str = "easyedit_strict") -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        if key == "target_token_bucket":
            group = _token_bucket(row["target_token_count"])
        elif key == "subject_token_bucket":
            group = _token_bucket(row["subject_token_count"])
        else:
            group = str(row.get(key, "unknown"))
        grouped.setdefault(group, []).append(row)
    return {
        group: {"n": len(rows), **summarize_protocol(rows, protocol)}
        for group, rows in sorted(grouped.items())
    }


def load_dataset(source: str) -> tuple[list[dict[str, Any]], str]:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source) as response:
            raw = response.read()
    else:
        raw = Path(source).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    return json.loads(raw), digest


def select_facts(data: list[dict[str, Any]], n: int, policy: str,
                 seed: int) -> list[dict[str, Any]]:
    if policy == "random":
        rng = random.Random(seed)
        return rng.sample(data, min(n, len(data)))
    return data[:n]


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument(
        "--output",
        default="results/local_protocol/official_eval.json",
    )
    parser.add_argument("--clamp_lm", type=float, default=0.20)
    parser.add_argument("--clamp_embed", type=float, default=0.06)
    parser.add_argument("--clamp_eos", type=float, default=0.16)
    parser.add_argument("--clamp_anti", type=float, default=0.06)
    parser.add_argument("--clamp_old", type=float, default=0.0)
    parser.add_argument("--target-token-mode", choices=["standalone", "contextual", "both"],
                        default="standalone",
                        help="Which target tokenization rows the WAL edit updates")
    parser.add_argument("--neg-projection-strength", type=float, default=0.3)
    parser.add_argument("--history-projection-strength", type=float, default=0.0)
    parser.add_argument("--embed-history-projection-strength", type=float, default=0.0)
    parser.add_argument("--max-history-keys", type=int, default=128)
    parser.add_argument("--neighbor-limit", type=int, default=0,
                        help="0 means use all CounterFact neighborhood prompts")
    parser.add_argument("--model", default=LLAMA, help="Model name or local path")
    parser.add_argument("--device", default="cuda:3", help="CUDA device")
    parser.add_argument("--dataset", default="https://rome.baulab.info/data/dsets/counterfact.json")
    parser.add_argument("--sample-policy", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--protocol", choices=["both", "strict", "practical"], default="both")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-only", dest="local_files_only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    protocol_names = {
        "both": ["easyedit_strict", "agim_practical"],
        "strict": ["easyedit_strict"],
        "practical": ["agim_practical"],
    }[args.protocol]

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
    print(f"\nCounterFact eval ({len(facts)} facts, protocol={args.protocol}):\n")

    evaluator = CounterFactEvaluator(model, tok, editor, device=args.device)
    t0 = time.time()
    results = evaluator.evaluate_all(
        facts,
        protocols=protocol_names,
        clamp_lm=args.clamp_lm,
        clamp_embed=args.clamp_embed,
        clamp_eos=args.clamp_eos,
        clamp_anti=args.clamp_anti,
        clamp_old=args.clamp_old,
        target_token_mode=args.target_token_mode,
        neg_projection_strength=args.neg_projection_strength,
        history_projection_strength=args.history_projection_strength,
        embed_history_projection_strength=args.embed_history_projection_strength,
        max_history_keys=args.max_history_keys,
        neighbor_limit=args.neighbor_limit,
    )
    elapsed = time.time() - t0
    if not results:
        raise RuntimeError("No examples were evaluated.")

    summaries = {
        protocol: summarize_protocol(results, protocol)
        for protocol in protocol_names
    }
    nt_summary = summarize_nt(results)
    strict = summaries.get("easyedit_strict")
    practical = summaries.get("agim_practical")

    payload: dict[str, Any] = {
        "n": len(results),
        "model": args.model,
        "device": args.device,
        "git_sha": git_sha(),
        "command": " ".join(sys.argv),
        "dataset": {
            "source": args.dataset,
            "sha256": dataset_sha256,
            "total_size": len(all_facts),
            "sample_policy": args.sample_policy,
            "seed": args.seed,
            "case_ids": [row.get("case_id") for row in facts],
            "neighbor_limit": args.neighbor_limit,
        },
        "generation": {
            "do_sample": False,
            "protocols": {
                name: {"repetition_penalty": CounterFactEvaluator.PROTOCOLS[name]}
                for name in protocol_names
            },
        },
        "hyperparams": {
            "clamp_lm": args.clamp_lm,
            "clamp_embed": args.clamp_embed,
            "clamp_eos": args.clamp_eos,
            "clamp_anti": args.clamp_anti,
            "clamp_old": args.clamp_old,
            "target_token_mode": args.target_token_mode,
            "neg_projection_strength": args.neg_projection_strength,
            "history_projection_strength": args.history_projection_strength,
            "embed_history_projection_strength": args.embed_history_projection_strength,
            "max_history_keys": args.max_history_keys,
        },
        "summaries": summaries,
        "NT": nt_summary,
        "time_s": round(elapsed, 2),
        "time_per_edit_s": round(elapsed / len(results), 4),
        "breakdowns": {
            "by_target_token_count": _group_summary(results, "target_token_bucket"),
            "by_subject_token_count": _group_summary(results, "subject_token_bucket"),
            "by_relation": _group_summary(results, "relation"),
        },
        "results": results,
    }

    if strict:
        payload["easyedit_strict"] = strict
        payload["EasyEdit"] = {
            "ES": strict["ES_token"],
            "PS": strict["PS_token_2"],
            "NS_absence": strict["NS_absence"],
            "Composite": strict["Composite_token_absence"],
        }
    if practical:
        payload["agim_practical"] = practical
        payload["AGIM"] = {
            "ES": practical["ES_substring"],
            "PS": practical["PS_substring_2"],
            "NS_absence": practical["NS_absence"],
            "Composite": practical["Composite_substring_absence"],
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 60}")
    print(f"COUNTERFACT RESULTS ({len(results)} facts)")
    print(f"{'=' * 60}")
    if strict:
        print(
            "  EasyEdit strict: "
            f"ES={strict['ES_token']:.1%} "
            f"PS@2={strict['PS_token_2']:.1%} "
            f"PS@All={strict['PS_token_all']:.1%} "
            f"NS_abs={strict['NS_absence']:.1%} "
            f"Comp={strict['Composite_token_absence']:.1%}"
        )
    if practical:
        print(
            "  AGIM practical:  "
            f"ES={practical['ES_substring']:.1%} "
            f"PS@2={practical['PS_substring_2']:.1%} "
            f"PS@All={practical['PS_substring_all']:.1%} "
            f"NS_abs={practical['NS_absence']:.1%} "
            f"Comp={practical['Composite_substring_absence']:.1%}"
        )
    print(
        "  NT: "
        f"lm={nt_summary['lm_head_non_edited_max']:.8f} "
        f"embed={nt_summary['embed_non_edited_max']:.8f} "
        f"EOS_changed={nt_summary['eos_row_changed_rate']:.0%}"
    )
    if strict:
        print(
            "  Rollback strict: "
            f"old_target={strict['RB_old_target']:.1%} "
            f"consistency={strict['RB_consistency']:.1%} "
            f"overlap={strict['RB_overlap']:.3f}"
        )
    print(f"  Time: {elapsed:.1f}s ({elapsed / len(results):.2f}s/edit)")
    print(f"\nSaved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
