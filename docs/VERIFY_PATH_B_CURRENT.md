# Verify Path B Current EasyEdit-Compatible Runs

Path B current claims use `agim.eval.easyedit_official_runner` and artifacts
under `results/easyedit_official/`. Legacy local CounterFact scripts are not
current EasyEdit evidence.

## Environment

Use explicit environment values instead of machine-specific defaults:

```bash
export AGIM_MODEL=meta-llama/Llama-3.1-8B-Instruct
export AGIM_DEVICE=cuda:0
export AGIM_EASYEDIT_ROOT="<YOUR_EASYEDIT_REPO_PATH>"
export AGIM_NT_SAMPLE_SIZE=500
export AGIM_LOCAL_FILES_ONLY=0
```

`--save-failures-only` пишет только failed cases из выбранных метрик. По умолчанию включены `tf`, `ctx_gen`, `prob`, дополнительный шумный канал `vanilla_gen` добавляется флагом:

```bash
--failure-families tf,ctx_gen,prob
--failure-families tf,ctx_gen,prob,vanilla_gen
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

### PS@All improvement sweep (plan)

Use one replayable sweep script for the 2222 improvement hypotheses:

```bash
cd /mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory
export AGIM_RELATION_PROFILE_MAP="$PWD/results/easyedit_official/ablations/relation_profile_map_seed42.json"
bash scripts/run_path_b_psall_improvement_sweep.sh --all --dry-run
```

Run concrete steps if needed:

```bash
bash scripts/run_path_b_psall_improvement_sweep.sh \
  --step baseline-42 \
  --step selective-anti \
  --step kpos-objective \
  --step kpos-ridge \
  --step decode-rerank \
  --step objective-balance
```

The sweep writes artifacts under `results/easyedit_official/ablations/` with fixed
names such as `baseline_random50_seed42.json`,
`ablation_selective_anti_repetition_seed42.json`,
`ablation_kpos_positive_w025_seed42.json`,
`ablation_kpos_kneg_seed42.json`,
`ablation_objective_balance_seed42.json`,
`ablation_decode_rerank_seed42.json`,
`relation_profile_map_seed42.json`,
`ablation_relation_aware_seed42.json`,
`ablation_conflict_budget_seed42.json`,
`sequential_n50_sanity.json`,
and `final_random1000_seed42.json`.

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

Required compatibility baseline for `--edit-backend wal_memit`:

```bash
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --n 50 --sample-policy random --seed 42 \
  --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
  --easyedit-root "$AGIM_EASYEDIT_ROOT" \
  --nt-sample-size "$AGIM_NT_SAMPLE_SIZE" \
  --edit-backend wal_memit \
  --method-profile-id single_loc_wal_memit_n50_seed42 \
  --target-token-mode contextual \
  --output results/easyedit_official/current/random_50_seed_42_wal_memit.json \
  --save-failures-only
```

Backend matrix smoke run:

```bash
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --n 50 --sample-policy random --seed 42 \
  --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
  --easyedit-root "$AGIM_EASYEDIT_ROOT" \
  --nt-sample-size "$AGIM_NT_SAMPLE_SIZE" \
  --compare-backends dual_row,wal_rome,wal_memit,side_slot \
  --method-profile-id matrix_dual_row_wal_rome_wal_memit_side_slot_random_50_seed42 \
  --output results/easyedit_official/ablations/backend_matrix_random_50_seed42.json \
  --save-failures-only
```

This writes the matrix artifact plus per-backend outputs such as
`backend_matrix_random_50_seed42.dual_row.json`, including
`backend_matrix_random_50_seed42.wal_memit.json`.

Post-hoc Ripple-style diagnostic over an EasyEdit artifact:

```bash
PYTHONPATH=src python -m agim.eval.ripple_diagnostic \
  --input results/easyedit_official/current/random_1000_seed_42.json \
  --output results/easyedit_official/current/random_1000_seed_42.ripple_style.json
```

This is a local related-fact diagnostic, not an official RippleEdits dataset
score.

Post-hoc MQuAKE-style diagnostic over an EasyEdit portability artifact:

```bash
PYTHONPATH=src python -m agim.eval.mquake_diagnostic \
  --input results/easyedit_official/current/knowedit_or_portability_run.json \
  --output results/easyedit_official/current/knowedit_or_portability_run.mquake_style.json
```

This is a local multi-hop/portability diagnostic, not an official MQuAKE dataset
score.

Tracked MQuAKE adapter output run:

```bash
PYTHONPATH=src python -m agim.eval.mquake_output_runner \
  --adapter results/external_benchmark_adapters/mquake_cf_3k_v2_first50_adapter.json \
  --device "$AGIM_DEVICE" \
  --output results/external_benchmark_runs/mquake_cf_3k_v2_first50_dual_row_outputs.json

PYTHONPATH=src python -m agim.eval.mquake_diagnostic \
  --score-adapter results/external_benchmark_adapters/mquake_cf_3k_v2_first50_adapter.json \
  --score-output results/external_benchmark_runs/mquake_cf_3k_v2_first50_dual_row_outputs.json \
  --output results/external_benchmark_runs/mquake_cf_3k_v2_first50_dual_row_scored.json
```

Current first-50 readout: `direct_rewrite_acc=100.0%`,
`multi_hop_acc=34.0%`, `composite_acc=67.0%`. This is diagnostic external
evidence for the current backend, not an official leaderboard claim.

Raw-text update proposal generation:

```bash
PYTHONPATH=src python -m agim.eval.raw_text_edit_pipeline \
  --text "The capital of France is Berlin." \
  --output results/easyedit_official/ablations/raw_text_proposal_france_capital.json
```

This only creates an edit proposal. It does not apply a patch or prove raw-text
editing quality by itself.

Raw-text scored-output layer:

```bash
PYTHONPATH=src python -m agim.eval.raw_text_edit_pipeline \
  --score-adapter results/easyedit_official/ablations/raw_text_proposal_france_capital.json \
  --score-output results/easyedit_official/ablations/raw_text_model_outputs.json \
  --output results/easyedit_official/ablations/raw_text_proposal_france_capital.scored.json
```

This scores a documented output payload. It is not an official AKEW result
unless the output payload came from a tracked external raw-text benchmark run.

Product-facing artifact diagnostic:

```bash
PYTHONPATH=src python -m agim.eval.product_diagnostic \
  --input results/easyedit_official/current/random_1000_seed_42.json \
  --output results/easyedit_official/current/random_1000_seed_42.product.json
```

This is a KnowEdit-inspired artifact score, not an external product benchmark.

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
