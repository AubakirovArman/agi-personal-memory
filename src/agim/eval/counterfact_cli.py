"""CLI for the legacy/local CounterFact evaluator."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agim.model.wal_dual_editor import WALDualLayerEditor

from .counterfact_data import COUNTERFACT_URL, LLAMA, git_sha, load_dataset, select_facts
from .counterfact_summary import group_summary, summarize_nt, summarize_protocol
from .easyedit_counterfact import CounterFactEvaluator


def main() -> int:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    protocol_names = {
        "both": ["easyedit_strict", "agim_practical"],
        "strict": ["easyedit_strict"],
        "practical": ["agim_practical"],
    }[args.protocol]

    model, tok, editor = _load_model(args)
    all_facts, dataset_sha256 = load_dataset(args.dataset)
    facts = select_facts(all_facts, args.n, args.sample_policy, args.seed)
    print(f"\nCounterFact eval ({len(facts)} facts, protocol={args.protocol}):\n")

    evaluator = CounterFactEvaluator(model, tok, editor, device=args.device)
    t0 = time.time()
    results = evaluator.evaluate_all(facts, protocols=protocol_names, **_edit_kwargs(args))
    elapsed = time.time() - t0
    if not results:
        raise RuntimeError("No examples were evaluated.")

    summaries = {protocol: summarize_protocol(results, protocol)
                 for protocol in protocol_names}
    nt_summary = summarize_nt(results)
    payload = _payload(args, results, summaries, nt_summary, elapsed,
                       dataset_sha256, all_facts, facts, protocol_names)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    _print_summary(results, summaries, nt_summary, elapsed, output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--output", default="results/local_protocol/official_eval.json")
    parser.add_argument("--clamp_lm", type=float, default=0.20)
    parser.add_argument("--clamp_embed", type=float, default=0.06)
    parser.add_argument("--clamp_eos", type=float, default=0.0)
    parser.add_argument("--clamp_anti", type=float, default=0.06)
    parser.add_argument("--clamp_old", type=float, default=0.0)
    parser.add_argument("--target-token-mode", choices=["standalone", "contextual", "both"],
                        default="standalone")
    parser.add_argument("--neg-projection-strength", type=float, default=0.3)
    parser.add_argument("--history-projection-strength", type=float, default=0.0)
    parser.add_argument("--embed-history-projection-strength", type=float, default=0.0)
    parser.add_argument("--max-history-keys", type=int, default=128)
    parser.add_argument("--neighbor-limit", type=int, default=0)
    parser.add_argument("--model", default=LLAMA, help="Model name or local path")
    parser.add_argument("--device", default="cuda:3", help="CUDA device")
    parser.add_argument("--dataset", default=COUNTERFACT_URL)
    parser.add_argument("--sample-policy", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--protocol", choices=["both", "strict", "practical"], default="both")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-only", dest="local_files_only", action="store_true")
    return parser


def _load_model(args):
    print(f"Loading {args.model} on {args.device}...")
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
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
    return model, tok, editor


def _edit_kwargs(args) -> dict[str, Any]:
    return {
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
        "neighbor_limit": args.neighbor_limit,
    }


def _payload(args, results, summaries, nt_summary, elapsed,
             dataset_sha256, all_facts, facts, protocol_names) -> dict[str, Any]:
    strict = summaries.get("easyedit_strict")
    practical = summaries.get("agim_practical")
    payload = {
        "n": len(results),
        "model": args.model,
        "device": args.device,
        "git_sha": git_sha(),
        "command": " ".join(sys.argv),
        "dataset": _dataset_payload(args, dataset_sha256, all_facts, facts),
        "generation": _generation_payload(protocol_names),
        "hyperparams": _edit_kwargs(args),
        "summaries": summaries,
        "NT": nt_summary,
        "time_s": round(elapsed, 2),
        "time_per_edit_s": round(elapsed / len(results), 4),
        "breakdowns": _breakdowns(results),
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
    return payload


def _dataset_payload(args, dataset_sha256, all_facts, facts) -> dict[str, Any]:
    return {
        "source": args.dataset,
        "sha256": dataset_sha256,
        "total_size": len(all_facts),
        "sample_policy": args.sample_policy,
        "seed": args.seed,
        "case_ids": [row.get("case_id") for row in facts],
        "neighbor_limit": args.neighbor_limit,
    }


def _generation_payload(protocol_names) -> dict[str, Any]:
    return {
        "do_sample": False,
        "protocols": {
            name: {"repetition_penalty": CounterFactEvaluator.PROTOCOLS[name]}
            for name in protocol_names
        },
    }


def _breakdowns(results) -> dict[str, Any]:
    return {
        "by_target_token_count": group_summary(results, "target_token_bucket"),
        "by_subject_token_count": group_summary(results, "subject_token_bucket"),
        "by_relation": group_summary(results, "relation"),
    }


def _print_summary(results, summaries, nt_summary, elapsed, output: Path) -> None:
    strict = summaries.get("easyedit_strict")
    practical = summaries.get("agim_practical")
    print(f"\n{'=' * 60}")
    print(f"COUNTERFACT RESULTS ({len(results)} facts)")
    print(f"{'=' * 60}")
    if strict:
        print("  EasyEdit strict: "
              f"ES={strict['ES_token']:.1%} PS@2={strict['PS_token_2']:.1%} "
              f"PS@All={strict['PS_token_all']:.1%} "
              f"NS_abs={strict['NS_absence']:.1%} "
              f"Comp={strict['Composite_token_absence']:.1%}")
    if practical:
        print("  AGIM practical:  "
              f"ES={practical['ES_substring']:.1%} "
              f"PS@2={practical['PS_substring_2']:.1%} "
              f"PS@All={practical['PS_substring_all']:.1%} "
              f"NS_abs={practical['NS_absence']:.1%} "
              f"Comp={practical['Composite_substring_absence']:.1%}")
    print("  NT: "
          f"lm={nt_summary['lm_head_non_edited_max']:.8f} "
          f"embed={nt_summary['embed_non_edited_max']:.8f} "
          f"EOS_changed={nt_summary['eos_row_changed_rate']:.0%}")
    if strict:
        print("  Rollback strict: "
              f"old_target={strict['RB_old_target']:.1%} "
              f"consistency={strict['RB_consistency']:.1%} "
              f"overlap={strict['RB_overlap']:.3f}")
    print(f"  Time: {elapsed:.1f}s ({elapsed / len(results):.2f}s/edit)")
    print(f"\nSaved {output}")
