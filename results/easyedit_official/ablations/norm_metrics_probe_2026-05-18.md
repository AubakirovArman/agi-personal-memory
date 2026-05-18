# Norm Metrics Probe - 2026-05-18

This small n=5 artifact verifies that new EasyEdit-compatible outputs store
edited-row delta L2 norm metrics under `NT`.

Command shape:

```text
AGIM_EASYEDIT_ROOT=/mnt/hf_model_weights/arman/3bit/sites/EasyEdit PYTHONPATH=src \
python -m agim.eval.easyedit_official_runner \
  --n 5 --sample-policy random --seed 42 --device cuda:2 \
  --target-token-mode contextual --nt-sample-size 8 \
  --output results/easyedit_official/ablations/norm_metrics_probe_5_seed_42.json \
  --save-failures-only
```

Artifacts:

- `norm_metrics_probe_5_seed_42.json`
- `norm_metrics_probe_5_seed_42.dry_run.json`
- `norm_metrics_probe_5_seed_42.failures.json`

Readout:

| Field | Value |
| --- | ---: |
| Code commit in artifact | `94400c33e6d97ea087a94a5608cca321f4ef5f49` |
| `summary.NT.edited_lm_delta_l2_mean` | 0.175726 |
| `summary.NT.edited_lm_delta_l2_max` | 0.191464 |
| `summary.NT.edited_embed_delta_l2_mean` | 0.063982 |
| `summary.NT.edited_embed_delta_l2_max` | 0.063999 |

This is a schema/diagnostic probe, not a benchmark claim.
