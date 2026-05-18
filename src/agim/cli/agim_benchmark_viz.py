from __future__ import annotations

from agim.cli.agim_benchmark_types import BenchmarkResult


def create_visualization(results: list[BenchmarkResult], output: str = "benchmark_visualization.png"):
    """Generate 6-panel matplotlib dashboard."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping visualization")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("AGI Personal Memory — Benchmark Dashboard", fontsize=16, fontweight='bold')

    n_list = [r.n_facts for r in results]
    baseline = [r.baseline_accuracy for r in results]
    post = [r.post_accuracy for r in results]

    # 1. Before vs After
    ax = axes[0, 0]
    x = range(len(results))
    w = 0.35
    ax.bar([i - w/2 for i in x], [b*100 for b in baseline], w, label='Baseline', color='#da3633')
    ax.bar([i + w/2 for i in x], [p*100 for p in post], w, label='+AGIM', color='#238636')
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_list])
    ax.set_ylabel('Accuracy %')
    ax.set_title('Before vs After')
    ax.legend()

    # 2. Accuracy Gain
    ax = axes[0, 1]
    deltas = [r.delta * 100 for r in results]
    ax.bar(range(len(results)), deltas, color='#58a6ff')
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([str(n) for n in n_list])
    ax.set_ylabel('Delta (pp)')
    ax.set_title('Accuracy Gain')

    # 3. Throughput
    ax = axes[0, 2]
    rates = [r.train_rate for r in results]
    ax.bar(range(len(results)), rates, color='#f0883e')
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([str(n) for n in n_list])
    ax.set_ylabel('Facts/sec')
    ax.set_title('Training Throughput')

    # 4. Per-Category (last result)
    ax = axes[1, 0]
    if results and results[-1].per_category:
        cats = list(results[-1].per_category.keys())[:8]
        cat_acc = [results[-1].per_category[c]["accuracy"] * 100 for c in cats]
        ax.barh(cats, cat_acc, color='#8957e5')
        ax.set_title('Per-Category Accuracy')

    # 5. Scaling Curve
    ax = axes[1, 1]
    if len(results) >= 3:
        ax.plot(n_list, [b*100 for b in baseline], 'o-', label='Baseline', color='#da3633')
        ax.plot(n_list, [p*100 for p in post], 'o-', label='+AGIM', color='#238636')
        ax.set_xscale('log')
        ax.set_xlabel('N Facts')
        ax.set_ylabel('Accuracy %')
        ax.set_title('Scaling Curve')
        ax.legend()

    # 6. Summary
    ax = axes[1, 2]
    ax.axis('off')
    if results:
        r = results[-1]
        summary = (
            f"Model: {r.model_name}\n"
            f"Facts: {r.n_facts}\n"
            f"Baseline: {r.baseline_accuracy:.1%}\n"
            f"Post-AGIM: {r.post_accuracy:.1%}\n"
            f"Delta: +{r.delta:.0%}\n"
            f"Train: {r.train_time_s:.1f}s\n"
            f"Rate: {r.train_rate:.0f} facts/s\n"
            f"Hit Rate: {r.memory_hit_rate:.1%}\n"
            f"Size: {r.model_size_mb:.0f} MB"
        )
        ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Visualization saved: {output}")


# ── Main ─────────────────────────────────────────────────────────
