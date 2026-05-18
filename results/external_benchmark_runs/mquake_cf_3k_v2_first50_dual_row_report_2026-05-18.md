# MQuAKE-CF-3k-v2 First-50 Dual-Row Report

Artifacts:

- Adapter: `results/external_benchmark_adapters/mquake_cf_3k_v2_first50_adapter.json`
- Model outputs: `results/external_benchmark_runs/mquake_cf_3k_v2_first50_dual_row_outputs.json`
- Scored output: `results/external_benchmark_runs/mquake_cf_3k_v2_first50_dual_row_scored.json`

Output command:

```bash
PYTHONPATH=src python -m agim.eval.mquake_output_runner \
  --adapter results/external_benchmark_adapters/mquake_cf_3k_v2_first50_adapter.json \
  --n 50 --device cuda:2 \
  --output results/external_benchmark_runs/mquake_cf_3k_v2_first50_dual_row_outputs.json
```

Scoring command:

```bash
PYTHONPATH=src python -m agim.eval.mquake_diagnostic \
  --score-adapter results/external_benchmark_adapters/mquake_cf_3k_v2_first50_adapter.json \
  --score-output results/external_benchmark_runs/mquake_cf_3k_v2_first50_dual_row_outputs.json \
  --output results/external_benchmark_runs/mquake_cf_3k_v2_first50_dual_row_scored.json
```

## Result

| Metric | Value |
| --- | ---: |
| Cases | 50 |
| Direct rewrite accuracy | 100.0% |
| Multi-hop accuracy | 34.0% |
| Composite accuracy | 67.0% |
| All-direct success rate | 100.0% |
| All-hop success rate | 16.0% |
| Output runtime | 80.29s |

## Readout

This is the first tracked MQuAKE model-output run for the current Path B
dual-row backend. It shows that direct edits are easy for the current editor,
but multi-hop consequences remain weak. The result supports the existing claim
boundary: AGIM Path B is useful as an audited hotfix foundation, not yet as a
solved lifelong or multi-hop knowledge editor.

This is not an official leaderboard claim. It is a documented first-50
diagnostic run over the MQuAKE-CF-3k-v2 adapter using Llama-3.1-8B-Instruct and
the current `WALDualLayerEditor`.
