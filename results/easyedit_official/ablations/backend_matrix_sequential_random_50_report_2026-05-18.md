# Sequential Backend Matrix Random 50 Report

Artifact:
`results/easyedit_official/ablations/backend_matrix_sequential_random_50_seed42.json`

Per-backend outputs:

- `results/easyedit_official/ablations/backend_matrix_sequential_random_50_seed42.dual_row.json`
- `results/easyedit_official/ablations/backend_matrix_sequential_random_50_seed42.side_slot.json`
- `results/easyedit_official/ablations/backend_matrix_sequential_random_50_seed42.wal_rome.json`

Command:

```bash
AGIM_EASYEDIT_ROOT=/mnt/hf_model_weights/arman/3bit/sites/EasyEdit \
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --n 50 --sample-policy random --seed 42 --device cuda:2 \
  --sequential-edit --retention-steps '' \
  --compare-backends dual_row,side_slot,wal_rome,wal_memit \
  --rome-target-layer 7 --rome-top-rows 8 --rome-clamp 0.04 \
  --output results/easyedit_official/ablations/backend_matrix_sequential_random_50_seed42.json \
  --save-failures-only
```

## Result

| Backend | Status | Profile | TF rewrite | TF rephrase | TF PS@All | TF locality | Prob rewrite | Prob PS@All | Prob locality | Time |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dual_row` | completed | `seq_tuned` | 88.0% | 24.0% | 21.0% | 59.9% | 98.0% | 53.0% | 87.0% | 71.65s |
| `side_slot` | completed | `seq_side_slot` | 98.0% | 30.0% | 25.0% | 97.2% | 100.0% | 52.0% | 87.4% | 72.26s |
| `wal_rome` | completed | `seq_wal_rome` | 0.0% | 0.0% | 0.0% | 98.8% | 12.0% | 14.0% | 88.8% | 57.52s |
| `wal_memit` | skipped | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

`wal_memit` is skipped because it is an offline `PatchArtifact` consolidation
backend and does not yet expose direct fact editing in the EasyEdit runner.

## Readout

This n=50 matrix gives the first same-run comparison for sequential runnable
backends:

- `side_slot` is the strongest current sequential backend on this seed. It
  keeps the base model frozen, routes sparse patches at runtime, and avoids the
  locality collapse seen in in-place accumulation.
- `dual_row` still edits the target facts but loses substantial locality after
  sequential accumulation.
- `wal_rome` preserves locality but does not yet produce direct rewrite with
  the current layer7/rows8/clamp0.04 configuration.

This closes the side-slot part of the backend-matrix gap. It is still not a
full final backend matrix because `wal_memit` remains an offline consolidation
backend without direct fact-editing evaluation.
