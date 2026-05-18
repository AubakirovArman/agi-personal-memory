# Backend Matrix Random 50 Report

Artifact:
`results/easyedit_official/ablations/backend_matrix_random_50_seed42_dual_walrome.json`

Per-backend outputs:

- `results/easyedit_official/ablations/backend_matrix_random_50_seed42_dual_walrome.dual_row.json`
- `results/easyedit_official/ablations/backend_matrix_random_50_seed42_dual_walrome.wal_rome.json`

Command:

```bash
AGIM_EASYEDIT_ROOT=/mnt/hf_model_weights/arman/3bit/sites/EasyEdit \
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --n 50 --sample-policy random --seed 42 --device cuda:2 \
  --compare-backends dual_row,wal_rome,wal_memit \
  --rome-target-layer 7 --rome-top-rows 8 --rome-clamp 0.04 \
  --output results/easyedit_official/ablations/backend_matrix_random_50_seed42_dual_walrome.json \
  --save-failures-only
```

## Result

| Backend | Status | TF rewrite | TF rephrase | TF PS@All | TF locality | Prob rewrite | Prob PS@All | Prob locality | Time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dual_row` | completed | 98.0% | 30.0% | 25.0% | 97.2% | 100.0% | 52.0% | 87.6% | 79.44s |
| `wal_rome` | completed | 0.0% | 0.0% | 0.0% | 98.2% | 12.0% | 13.0% | 88.6% | 60.12s |
| `wal_memit` | skipped | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

`wal_memit` is skipped because it is an offline `PatchArtifact` consolidation
backend and does not yet expose direct fact editing in the EasyEdit runner.

## Readout

This n=50 matrix gives a clear current backend ordering:

- `dual_row` remains the measured default for Llama-3.1-8B-Instruct Path B
  EasyEdit-compatible edits.
- `wal_rome` preserves locality but does not achieve direct rewrite with the
  current layer7/rows8/clamp0.04 configuration.
- `wal_memit` remains a patch-consolidation foundation, not a direct EasyEdit
  fact editor.

This is a real n=50 backend artifact for runnable direct-edit backends, but it
is not a full final matrix because `side_slot` is sequential-only and
`wal_memit` has no direct fact-editing evaluation path yet.
