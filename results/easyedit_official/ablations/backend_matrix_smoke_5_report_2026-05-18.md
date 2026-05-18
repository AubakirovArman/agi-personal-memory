# Backend Matrix Smoke 5 Report

Artifact:
`results/easyedit_official/ablations/backend_matrix_smoke_5_first42.json`

Per-backend outputs:

- `results/easyedit_official/ablations/backend_matrix_smoke_5_first42.dual_row.json`
- `results/easyedit_official/ablations/backend_matrix_smoke_5_first42.wal_rome.json`

Command:

```bash
AGIM_EASYEDIT_ROOT=/mnt/hf_model_weights/arman/3bit/sites/EasyEdit \
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --n 5 --sample-policy first --seed 42 --device cuda:2 \
  --compare-backends dual_row,wal_rome,wal_memit \
  --rome-target-layer 7 --rome-top-rows 8 --rome-clamp 0.04 \
  --output results/easyedit_official/ablations/backend_matrix_smoke_5_first42.json \
  --save-failures-only
```

## Result

| Backend | Status | TF rewrite | TF rephrase | TF PS@All | TF locality | Prob rewrite | Prob PS@All | Prob locality |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dual_row` | completed | 100.0% | 60.0% | 40.0% | 94.0% | 100.0% | 90.0% | 84.0% |
| `wal_rome` | completed | 0.0% | 20.0% | 10.0% | 100.0% | 20.0% | 30.0% | 76.0% |
| `wal_memit` | skipped | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

`wal_memit` is skipped because it is an offline `PatchArtifact` consolidation
backend and does not yet expose direct fact editing in the runner.

## Readout

This smoke confirms the matrix runner works and writes per-backend artifacts.
It also confirms the current backend ordering on this tiny slice:

- `dual_row` remains the only measured backend with strong direct rewrite.
- `wal_rome` preserves locality but is not tuned enough for rewrite.
- `wal_memit` should stay outside headline EasyEdit comparisons until it has a
  direct evaluation path or a batch-consolidation benchmark.

This is not a full n=50 backend comparison.
