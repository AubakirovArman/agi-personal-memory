# Deterministic NT Probe - 2026-05-18

This small n=5 artifact verifies that new EasyEdit-compatible outputs store the
deterministic non-target sampled row ids used by NT max-diff checks.

Command shape:

```text
AGIM_EASYEDIT_ROOT=/mnt/hf_model_weights/arman/3bit/sites/EasyEdit PYTHONPATH=src \
python -m agim.eval.easyedit_official_runner \
  --n 5 --sample-policy random --seed 42 --device cuda:2 \
  --target-token-mode contextual --nt-sample-size 8 \
  --output results/easyedit_official/ablations/nt_deterministic_probe_5_seed_42.json \
  --save-failures-only
```

Artifacts:

- `nt_deterministic_probe_5_seed_42.json`
- `nt_deterministic_probe_5_seed_42.dry_run.json`
- `nt_deterministic_probe_5_seed_42.failures.json`

Readout:

| Field | Value |
| --- | --- |
| Code commit in artifact | `9620253b2f0ef6b956452505ffb81209454a02ec` |
| `nt_sample_mode` | `deterministic_lcg` |
| `nt_sample_size` | `8` |
| First metric `lm_head_sampled_row_ids` length | `8` |
| First metric `embed_sampled_row_ids` length | `8` |
| First four lm_head sampled ids | `2275,25802,36423,49329` |
| First four embed sampled ids | `10775,34302,57829,68450` |

This is a schema/diagnostic probe, not a benchmark claim.
