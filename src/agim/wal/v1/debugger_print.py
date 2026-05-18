from __future__ import annotations

from typing import Any, Dict

from .debugger_types import HeatmapStats, TraceRecord


class WALDebuggerPrintMixin:
    def print_trace(self, record: TraceRecord, show_tree: bool = True):
        """Pretty-print a trace record."""
        print(f"  [{record.index:6d}] ATOM {record.atom_id:3d} × COEF {record.coeff_id:2d}")
        print(f"           atom_val={record.atom_value:10.6f} × coeff_val={record.coeff_value:10.6f}")
        print(f"           product={record.product:10.6f} + residual={record.residual:10.6f}")
        print(f"           final={record.final_value:10.6f}")
        if record.breakpoint_hit:
            print(f"           >>> BREAKPOINT HIT: {record.breakpoint_hit} <<<")
        if show_tree and record.atom_resolution_tree:
            print(f"           Resolution tree:")
            for line in record.atom_resolution_tree.split('\n'):
                print(f"             {line}")
    
    def print_heatmap(self, stats: HeatmapStats, top_k: int = 10):
        """Pretty-print heatmap statistics."""
        print(f"Program Heatmap ({stats.total_weights:,} weights)")
        print(f"  Residuals: {stats.residual_count} ({stats.residual_pct:.2f}%)")
        print(f"  Atom entropy: {stats.atom_entropy:.3f} bits")
        print(f"  Coeff entropy: {stats.coeff_entropy:.3f} bits")
        print(f"\n  Top {top_k} atoms:")
        for aid, cnt, pct in stats.top_atoms[:top_k]:
            print(f"    ATOM {aid:3d}: {cnt:8,} ({pct:5.2f}%)")
        print(f"\n  Top {top_k} coeffs:")
        for cid, cnt, pct in stats.top_coeffs[:top_k]:
            print(f"    COEF {cid:2d}: {cnt:8,} ({pct:5.2f}%)")
    
    def print_diff(self, diff: Dict[str, Any]):
        """Pretty-print diff results."""
        print(f"Diff: {diff['name1']} vs {diff['name2']}")
        print(f"  Total weights: {diff['total_weights']:,}")
        print(f"  Identical: {diff['identical']}")
        if not diff['identical']:
            print(f"  Value diffs: {diff['value_diffs']:,} ({diff['value_diff_pct']:.4f}%)")
            print(f"  Atom diffs: {diff['atom_diffs']:,}")
            print(f"  Coeff diffs: {diff['coeff_diffs']:,}")
            print(f"  Max value diff: {diff['max_value_diff']:.8f}")
            if diff['sample_diffs']:
                print(f"  Sample diffs (first 10):")
                for d in diff['sample_diffs']:
                    print(f"    [{d['index']:6d}] atom={d['atom']} coeff={d['coeff']} diff={d['value_diff']:.8f}")
