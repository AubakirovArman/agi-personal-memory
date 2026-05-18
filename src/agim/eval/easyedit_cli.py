"""CLI and console reporting for the EasyEdit-compatible runner."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from agim.eval.easyedit_counterfact import LLAMA

from .easyedit_loader import DEFAULT_EASYEDIT_ROOT
from .easyedit_presets import PRESETS
from .easyedit_run_metadata import DEFAULT_FAILURE_FAMILIES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--preset", choices=sorted(PRESETS),
                        help="Named reproducible run preset")
    parser.add_argument("--model", default=LLAMA)
    parser.add_argument("--device", default=os.environ.get("AGIM_DEVICE", "cuda:0"))
    parser.add_argument("--dataset", default="https://rome.baulab.info/data/dsets/counterfact.json")
    parser.add_argument("--sample-policy", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="results/easyedit_official/current/easyedit_official_50.json",
    )
    parser.add_argument("--dry-run-summary", action="store_true",
                        help="Only inspect selected CounterFact records; do not load model/EasyEdit")
    parser.add_argument("--dry-run-output",
                        help="Optional dry-run JSON output path")
    parser.add_argument("--save-failures-only", action="store_true",
                        help="Write a compact JSON containing only failed cases")
    parser.add_argument("--failures-output",
                        help="Optional failures-only JSON output path")
    parser.add_argument("--failure-families", default=",".join(DEFAULT_FAILURE_FAMILIES),
                        help="Comma-separated: tf,ctx_gen,prob,vanilla_gen")
    parser.add_argument("--method-profile-id",
                        help="Optional explicit operating profile id for artifacts")
    parser.add_argument("--easyedit-root", type=Path, default=DEFAULT_EASYEDIT_ROOT)
    parser.add_argument("--locality-limit", type=int, default=0,
                        help="0 means all official CounterFact locality prompts")
    parser.add_argument("--clamp_lm", type=float, default=0.20)
    parser.add_argument("--clamp_embed", type=float, default=0.06)
    parser.add_argument("--clamp_eos", type=float, default=0.0)
    parser.add_argument("--clamp_anti", type=float, default=0.06)
    parser.add_argument("--clamp_old", type=float, default=0.0)
    parser.add_argument("--target-token-mode", choices=["standalone", "contextual", "both"],
                        default="contextual",
                        help="contextual edits EasyEdit prompt + space + target label ids")
    parser.add_argument("--use-positive-prompts", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Average edit keys with CounterFact paraphrase prompts")
    parser.add_argument("--positive-prompt-limit", type=int, default=4)
    parser.add_argument("--positive-key-weight", type=float, default=1.0)
    parser.add_argument("--positive-constraint-mode", choices=["none", "projected", "ridge"],
                        default="none")
    parser.add_argument("--use-neg-prompts", action=argparse.BooleanOptionalAction, default=True,
                        help="Project edit key away from locality/neighborhood prompt keys")
    parser.add_argument("--neg-prompt-limit", type=int, default=10)
    parser.add_argument("--neg-projection-strength", type=float, default=0.3)
    parser.add_argument("--history-projection-strength", type=float, default=0.0)
    parser.add_argument("--embed-history-projection-strength", type=float, default=0.0)
    parser.add_argument("--projection-mode", choices=["sequential", "orthogonal"],
                        default="sequential")
    parser.add_argument("--history-slot-mode", choices=["global", "relation"],
                        default="global",
                        help="Use global or relation_id-sharded edit history basis")
    parser.add_argument("--max-history-keys", type=int, default=128)
    parser.add_argument("--relation-protected-mode",
                        choices=["none", "accumulate", "preload"],
                        default="none",
                        help="Use relation_id-scoped locality prompt banks")
    parser.add_argument("--relation-protected-prompt-limit", type=int, default=4)
    parser.add_argument("--max-relation-protected-keys", type=int, default=64)
    parser.add_argument("--wal-encode-updates", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Disable for exact-additive update ablations")
    parser.add_argument("--nt-sample-size", type=int, default=500,
                        help="Deterministic non-target row sample size")
    parser.add_argument("--probability-metrics", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Compute P(target_new)>P(target_true) style metrics")
    parser.add_argument("--test-fluency", action="store_true",
                        help="Compute EasyEdit-style n-gram entropy on post-edit generations")
    parser.add_argument("--sequential-edit", action="store_true",
                        help="Apply all edits first, then evaluate without per-case rollback")
    parser.add_argument("--retention-steps", default="1,10,50",
                        help="Comma-separated sequential retention checkpoints; empty disables")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-easyedit-log", action="store_true")
    return parser


def print_progress(summary: dict, idx: int, total: int) -> None:
    post_summary = summary["post"]
    loc = post_summary.get("locality", {}).get("neighborhood_acc")
    gen = summary["post_generation_vanilla"]
    ctx_gen = summary.get("post_generation_contextual", {})
    print(
        f"  [{idx}/{total}] "
        f"TF rewrite={post_summary.get('rewrite_acc', 0):.1%} "
        f"TF rephrase={post_summary.get('rephrase_acc', 0):.1%} "
        f"TF ps_all={post_summary.get('rephrase_all_acc', 0):.1%} "
        f"TF loc={0.0 if loc is None else loc:.1%} "
        f"GEN rewrite={gen['rewrite_acc']:.1%} "
        f"CTX-GEN rewrite={ctx_gen.get('rewrite_acc', 0):.1%}",
        flush=True,
    )


def print_final_summary(summary: dict, retention: dict, elapsed: float,
                        n_metrics: int, output: Path) -> None:
    post = summary["post"]
    loc = post.get("locality", {}).get("neighborhood_acc")
    gen = summary["post_generation_vanilla"]
    print(f"\n{'=' * 64}")
    print(f"OFFICIAL EASYEDIT RESULTS ({n_metrics} facts)")
    print(f"{'=' * 64}")
    print(
        "  Teacher-forcing: "
        f"rewrite={post.get('rewrite_acc', 0):.1%} "
        f"rephrase={post.get('rephrase_acc', 0):.1%} "
        f"ps_all={post.get('rephrase_all_acc', 0):.1%} "
        f"locality={0.0 if loc is None else loc:.1%}"
    )
    print(
        "  Vanilla generation: "
        f"rewrite={gen['rewrite_acc']:.1%} "
        f"rephrase={gen.get('rephrase_acc', 0):.1%} "
        f"ps_all={gen.get('rephrase_all_acc', 0):.1%}"
    )
    _print_optional_groups(summary, post, retention)
    print(f"  Time: {elapsed:.1f}s ({elapsed / max(n_metrics, 1):.2f}s/edit)")
    print(f"\nSaved {output}")


def _print_optional_groups(summary: dict, post: dict, retention: dict) -> None:
    if "post_generation_contextual" in summary:
        ctx_gen = summary["post_generation_contextual"]
        print(
            "  Contextual generation: "
            f"rewrite={ctx_gen['rewrite_acc']:.1%} "
            f"rephrase={ctx_gen.get('rephrase_acc', 0):.1%} "
            f"ps_all={ctx_gen.get('rephrase_all_acc', 0):.1%}"
        )
    if "NT" in summary:
        nt = summary["NT"]
        print(
            "  NT diff: "
            f"lm_head={nt['lm_head_non_edited_max']:.2e} "
            f"embed={nt['embed_non_edited_max']:.2e} "
            f"EOS_changed={nt['eos_row_changed_rate']:.0%}"
        )
    if "post_probability" in summary:
        prob = summary["post_probability"]
        print(
            "  Probability compare: "
            f"rewrite={prob.get('rewrite_acc', 0):.1%} "
            f"rephrase={prob.get('rephrase_acc', 0):.1%} "
            f"ps_all={prob.get('rephrase_all_acc', 0):.1%} "
            f"locality={prob.get('locality_acc', 0):.1%}"
        )
    if retention:
        _print_retention(retention)
    if "portability" in post:
        print(f"  Portability: mean={post['portability'].get('mean_acc', 0):.1%}")
    if "post_fluency" in summary:
        print(
            "  Fluency: "
            f"ngram_entropy={summary['post_fluency']['ngram_entropy']:.3f}"
        )


def _print_retention(retention: dict) -> None:
    print("  Retention:")
    for key, value in retention.items():
        post_value = value["summary"]["post"]
        loc_value = post_value.get("locality", {}).get("neighborhood_acc", 0.0)
        print(
            f"    {key}: rewrite={post_value.get('rewrite_acc', 0):.1%} "
            f"rephrase={post_value.get('rephrase_acc', 0):.1%} "
            f"ps_all={post_value.get('rephrase_all_acc', 0):.1%} "
            f"locality={loc_value:.1%}"
        )
