# Norm Budget No-Commit Probe

Date: 2026-05-18

Runner: official EasyEdit-compatible CounterFact runner

Model: `meta-llama/Llama-3.1-8B-Instruct`

Code commit in artifact: `09214084552467b94415f85c80bd951c75515be5`

Artifact:

- `norm_budget_no_commit_probe_5_seed_42.json`
- `norm_budget_no_commit_probe_5_seed_42.failures.json`

Command:

```bash
AGIM_EASYEDIT_ROOT=/mnt/hf_model_weights/arman/3bit/sites/EasyEdit \
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --n 5 \
  --sample-policy random \
  --seed 42 \
  --device cuda:2 \
  --target-token-mode contextual \
  --nt-sample-size 8 \
  --max-row-delta-norm 0.01 \
  --output results/easyedit_official/ablations/norm_budget_no_commit_probe_5_seed_42.json \
  --save-failures-only
```

## Readout

This is a diagnostic gate probe, not a benchmark profile. The row-delta limit is
intentionally too low, so all five proposed edits are rejected:

| Metric | Value |
| --- | ---: |
| `edit_status=no_commit` | 5 / 5 |
| TF rewrite | 0.0% |
| TF locality | 100.0% |
| NT lm_head non-edited max | 0.00e+00 |
| NT embed non-edited max | 0.00e+00 |

The first rejected edit exceeded `max_row_delta_norm=0.01` with
`max_delta_norm=0.196241`. The runner rolled the proposal back before post-edit
evaluation, which is the intended no-commit behavior.
