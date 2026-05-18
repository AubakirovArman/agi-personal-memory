# Verify Path B Current EasyEdit-Compatible Runs

Path B current claims use `agim.eval.easyedit_official_runner` and artifacts
under `results/easyedit_official/`. Legacy local CounterFact scripts are not
current EasyEdit evidence.

## Environment

Use explicit environment values instead of machine-specific defaults:

```bash
export AGIM_MODEL=meta-llama/Llama-3.1-8B-Instruct
export AGIM_DEVICE=cuda:2
export AGIM_EASYEDIT_ROOT=/path/to/EasyEdit
export AGIM_NT_SAMPLE_SIZE=500
```

If the model is only available in the local Hugging Face cache, keep
`--local-files-only`. For a clean external reproduction, use
`--no-local-files-only` and a model id you can download.

## Dry-Run The Dataset Selection

```bash
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --n 1000 --sample-policy first \
  --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
  --easyedit-root "$AGIM_EASYEDIT_ROOT" \
  --nt-sample-size "$AGIM_NT_SAMPLE_SIZE" \
  --output results/easyedit_official/current/easyedit_official_1000_first_default.json \
  --dry-run-summary
```

This must write `easyedit_official_1000_first_default.dry_run.json` and report
1000 selected case ids.

## Current Reproducibility Bundle

Single-edit PS-oriented first-50 profile:

```bash
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --n 50 --sample-policy first --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
  --easyedit-root "$AGIM_EASYEDIT_ROOT" \
  --nt-sample-size "$AGIM_NT_SAMPLE_SIZE" \
  --output results/easyedit_official/current/easyedit_official_50_first42_psall_baseline.json \
  --neg-prompt-limit 4 --test-fluency --save-failures-only
```

Single-edit locality-protected random seeds:

```bash
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --preset random_50_seed_42 --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
  --easyedit-root "$AGIM_EASYEDIT_ROOT" \
  --nt-sample-size "$AGIM_NT_SAMPLE_SIZE" --save-failures-only
```

Random-200 stability layer:

```bash
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --n 200 --sample-policy random --seed 42 \
  --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
  --easyedit-root "$AGIM_EASYEDIT_ROOT" \
  --nt-sample-size "$AGIM_NT_SAMPLE_SIZE" \
  --output results/easyedit_official/current/random_200_seed_42.json \
  --save-failures-only
```

Random-1000 stability layer:

```bash
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --n 1000 --sample-policy random --seed 42 \
  --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
  --easyedit-root "$AGIM_EASYEDIT_ROOT" \
  --nt-sample-size "$AGIM_NT_SAMPLE_SIZE" \
  --output results/easyedit_official/current/random_1000_seed_42.json \
  --save-failures-only
```

Token-mode ablation matrix:

```bash
for mode in standalone contextual both; do
  PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
    --n 200 --sample-policy random --seed 42 \
    --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
    --easyedit-root "$AGIM_EASYEDIT_ROOT" \
    --nt-sample-size "$AGIM_NT_SAMPLE_SIZE" \
    --target-token-mode "$mode" \
    --output "results/easyedit_official/ablations/token_mode_random_200_seed_42_${mode}.json" \
    --save-failures-only
done
```

Component ablation matrix:

```bash
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --n 200 --sample-policy random --seed 42 \
  --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
  --easyedit-root "$AGIM_EASYEDIT_ROOT" --target-token-mode contextual \
  --nt-sample-size "$AGIM_NT_SAMPLE_SIZE" \
  --clamp_lm 0.20 --clamp_embed 0 --clamp_eos 0 --clamp_anti 0 \
  --output results/easyedit_official/ablations/component_random_200_seed_42_lm_head_only.json \
  --save-failures-only
```

The full matrix repeats the same command shape for `embed_only`, `dual`,
`dual_no_eos`, `dual_no_anti`, and `dual_no_eos_anti`; see
`results/easyedit_official/ablations/component_ablation_report_2026-05-18.md`
for exact clamps and artifact names.

Exact-additive update ablation:

```bash
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --n 200 --sample-policy random --seed 42 \
  --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
  --easyedit-root "$AGIM_EASYEDIT_ROOT" --target-token-mode contextual \
  --nt-sample-size "$AGIM_NT_SAMPLE_SIZE" \
  --clamp_lm 0.20 --clamp_embed 0.06 --clamp_eos 0.16 --clamp_anti 0.06 \
  --no-wal-encode-updates \
  --output results/easyedit_official/ablations/exact_additive_random_200_seed_42_dual.json \
  --save-failures-only
```

Current first-1000 scale check:

```bash
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --n 1000 --sample-policy first --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
  --easyedit-root "$AGIM_EASYEDIT_ROOT" \
  --nt-sample-size "$AGIM_NT_SAMPLE_SIZE" \
  --output results/easyedit_official/current/easyedit_official_1000_first_default.json \
  --save-failures-only
```

Sequential tuned profile:

```bash
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --n 50 --sample-policy random --seed 42 \
  --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
  --easyedit-root "$AGIM_EASYEDIT_ROOT" \
  --nt-sample-size "$AGIM_NT_SAMPLE_SIZE" \
  --output results/easyedit_official/sequential/random_50_seed_42_seq_lm015_negx05_noeosanti_retention.json \
  --save-failures-only --sequential-edit --target-token-mode contextual \
  --use-neg-prompts --neg-prompt-limit 4 --neg-projection-strength 0.50 \
  --clamp_lm 0.15 --clamp_eos 0 --clamp_anti 0 --retention-steps 1,10,50
```

Experimental `wal_rome` backend smoke run:

```bash
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --n 50 --sample-policy random --seed 42 \
  --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
  --easyedit-root "$AGIM_EASYEDIT_ROOT" \
  --nt-sample-size "$AGIM_NT_SAMPLE_SIZE" \
  --edit-backend wal_rome --rome-target-layer 7 --rome-top-rows 32 \
  --output results/easyedit_official/ablations/random_50_seed_42_wal_rome_layer7.json \
  --save-failures-only
```

This command is for backend comparison. Do not cite `wal_rome` as a headline
profile until a tracked artifact and report exist.

Backend matrix smoke run:

```bash
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --n 50 --sample-policy random --seed 42 \
  --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
  --easyedit-root "$AGIM_EASYEDIT_ROOT" \
  --nt-sample-size "$AGIM_NT_SAMPLE_SIZE" \
  --compare-backends dual_row,wal_rome,wal_memit \
  --output results/easyedit_official/ablations/random_50_seed_42_backend_matrix.json \
  --save-failures-only
```

This writes the matrix artifact plus per-backend outputs such as
`random_50_seed_42_backend_matrix.dual_row.json`.

## Current n=1000 Readout

The first-1000 artifact reports:

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing | 91.1% | 25.4% | 24.7% | 96.2% |
| Contextual generation | 91.0% | 24.8% | 24.1% | n/a |
| Probability compare | 96.3% | 43.5% | 43.6% | 87.5% |

This supports a rewrite/locality hotfix profile. It does not prove solved
paraphrase generalization or lifelong editing.

The random-1000 seed 42 artifact reports:

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing | 94.5% | 23.8% | 23.5% | 96.4% |
| Contextual generation | 94.2% | 23.0% | 22.7% | n/a |
| Probability compare | 97.2% | 40.9% | 41.8% | 86.5% |

This confirms the same operating point on a random n=1000 sample.
