"""AGIM Full Benchmark CLI."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from agim.cli.agim_benchmark_data import evaluate_model, load_counterfact_dataset
from agim.cli.agim_benchmark_types import BenchmarkResult, Leaderboard
from agim.cli.agim_benchmark_viz import create_visualization
from agim.model.memory_model import MemoryAugmentedModel


def main():
    import argparse
    p = argparse.ArgumentParser(description="AGIM Full Benchmark")
    p.add_argument("--n_facts", type=int, default=500, help="Number of facts")
    p.add_argument("--n_test", type=int, default=100, help="Test questions")
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--output_dir", default=".")
    p.add_argument("--visualize", action="store_true")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ── PHASE 1: Load ──
    print("=" * 60)
    print("PHASE 1: Loading model...")
    print("=" * 60)
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map=args.device)
    base.eval()

    # ── Load facts, split into train and test ──
    all_facts = load_counterfact_dataset(args.n_facts + args.n_test)
    train_facts = all_facts[:args.n_facts]
    test_facts = all_facts[args.n_facts:args.n_facts + args.n_test]

    # ── PHASE 2: Baseline (model.generate, NO memory) ──
    print(f"\nPHASE 2: Baseline — {len(test_facts)} fictional questions, NO memory")
    mem_model = MemoryAugmentedModel(base, tok, memory_dir="./bench_memory", device=args.device)
    baseline_acc, _ = evaluate_model(mem_model, test_facts, use_memory=False, verbose=True)

    # ── PHASE 3: Training (DIFFERENT facts than test!) ──
    print(f"\nPHASE 3: Training on {len(train_facts)} fictional facts...")
    qa_pairs = [f.to_qa() for f in train_facts]
    t0 = time.time()
    taught = mem_model.teach_batch(qa_pairs)
    train_time = time.time() - t0

    # ── PHASE 4: Post-test (memory lookup, DIFFERENT questions) ──
    print(f"\nPHASE 4: Post-test — {len(test_facts)} questions WITH AGIM memory")
    post_acc, details = evaluate_model(mem_model, test_facts, use_memory=True, verbose=True)

    # ── Results ──
    result = BenchmarkResult(
        model_name=f"Llama-3.1-8B + AGIM",
        n_facts=taught,
        n_tested=len(test_facts),
        baseline_accuracy=round(baseline_acc, 4),
        post_accuracy=round(post_acc, 4),
        delta=round(post_acc - baseline_acc, 4),
        train_time_s=round(train_time, 2),
        train_rate=round(taught / train_time) if train_time > 0 else 0,
        memory_hit_rate=round(details.get("memory_hit_rate", 0), 4),
        model_size_mb=round(mem_model.model_size_mb, 0),
        per_category=details.get("per_category", {}),
    )

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Baseline:  {result.baseline_accuracy:.1%}")
    print(f"  Post-AGIM: {result.post_accuracy:.1%}")
    print(f"  Delta:     +{result.delta:.0%}")
    print(f"  Trained:   {taught}/{len(train_facts)} in {train_time:.1f}s ({result.train_rate:.0f}/s)")
    print(f"  Hit rate:  {result.memory_hit_rate:.1%}")
    print(f"  Size:      {result.model_size_mb:.0f} MB")

    # Save
    out = Path(args.output_dir)
    (out / "benchmark_results.json").write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

    lb = Leaderboard(out / "agim_leaderboard.json")
    lb.add(result)
    lb.print()

    if args.visualize:
        create_visualization([result], str(out / "benchmark_visualization.png"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
