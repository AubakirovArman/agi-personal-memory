# WALRome Smoke 5 Report

Artifact:
`results/easyedit_official/ablations/wal_rome_smoke_5_first42_layer7_rows8_clamp004.json`

Failures:
`results/easyedit_official/ablations/wal_rome_smoke_5_first42_layer7_rows8_clamp004.failures.json`

Command:

```bash
AGIM_EASYEDIT_ROOT=/mnt/hf_model_weights/arman/3bit/sites/EasyEdit \
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --n 5 --sample-policy first --seed 42 --device cuda:2 \
  --edit-backend wal_rome \
  --rome-target-layer 7 --rome-top-rows 8 --rome-clamp 0.04 \
  --output results/easyedit_official/ablations/wal_rome_smoke_5_first42_layer7_rows8_clamp004.json \
  --save-failures-only \
  --failures-output results/easyedit_official/ablations/wal_rome_smoke_5_first42_layer7_rows8_clamp004.failures.json
```

## Result

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing | 0.0% | 20.0% | 10.0% | 100.0% |
| Vanilla generation | 0.0% | 0.0% | 0.0% | n/a |
| Contextual generation | 0.0% | 20.0% | 10.0% | n/a |
| Probability compare | 20.0% | 40.0% | 30.0% | 76.0% |

NT summary:

- `lm_head_non_edited_max=0.0`
- `embed_non_edited_max=0.0`
- `ffn_down_proj_non_edited_max=0.0`
- `edited_ffn_rows_avg=8.0`
- `edited_ffn_delta_l2_mean=0.00087`

## Readout

This is a negative smoke result. The current `wal_rome` configuration preserves
locality and touches only sparse FFN rows, but it does not achieve direct
teacher-forcing rewrite on the 5-fact smoke slice. Do not run or claim an n=50
WALRome headline result until the located FFN update is tuned.
