# Path B Hard-Gate Execution Queue

Use this queue to execute remaining completion gates in order.
All commands assume `sites/agi_personal_memory` as the active project context.

```bash
cd /mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory
export AGIM_MODEL="meta-llama/Llama-3.1-8B-Instruct"
export AGIM_DEVICE="cuda"
export AGIM_LOCAL_FILES_ONLY=0
export AGIM_EASYEDIT_ROOT="<YOUR_EASYEDIT_REPO_PATH>"
export AGIM_MQUAKE_ADAPTER="<YOUR_MQUAKE_ADAPTER_PATH>"
```

## Gate 1 — wal_memit official baseline (`n=50`)

```bash
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --n 50 --sample-policy random --seed 42 \
  --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
  --easyedit-root "$AGIM_EASYEDIT_ROOT" \
  --nt-sample-size 500 \
  --edit-backend wal_memit \
  --method-profile-id single_loc_wal_memit_n50_seed42 \
  --save-failures-only --save-neighbor-text \
  --output results/easyedit_official/current/random_50_seed_42_wal_memit.json
```

## Gate 2 — backend quality matrix (`dual_row, wal_rome, wal_memit, side_slot`)

```bash
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --dataset random --sample-policy random --n 50 --seed 42 \
  --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
  --easyedit-root "$AGIM_EASYEDIT_ROOT" \
  --nt-sample-size 500 \
  --compare-backends dual_row,wal_rome,wal_memit,side_slot \
  --method-profile-id matrix_dual_row_wal_rome_wal_memit_side_slot_random_50_seed42 \
  --output results/easyedit_official/ablations/backend_matrix_random_50_seed42.json \
  --save-failures-only \
  --save-neighbor-text
```

Expected matrix artifacts:
- `results/easyedit_official/ablations/backend_matrix_random_50_report_2026-05-18.md`
- `results/easyedit_official/ablations/backend_matrix_random_50_seed42.dual_row.json`
- `results/easyedit_official/ablations/backend_matrix_random_50_seed42.wal_rome.json`
- `results/easyedit_official/ablations/backend_matrix_random_50_seed42.wal_memit.json`
- `results/easyedit_official/ablations/backend_matrix_random_50_seed42.side_slot.json`

## Gate 3 — external consequence evidence chain

```bash
PYTHONPATH=src python -m agim.eval.ripple_diagnostic \
  --input results/easyedit_official/current/random_50_seed_42_wal_memit.json \
  --output results/external_benchmark_runs/ripple_wal_memit_n50_seed42.json

PYTHONPATH=src python -m agim.eval.mquake_output_runner \
  --adapter "$AGIM_MQUAKE_ADAPTER" \
  --output results/external_benchmark_runs/mquake_wal_memit_n50_seed42_outputs.json

PYTHONPATH=src python -m agim.eval.mquake_diagnostic \
  --score-adapter "$AGIM_MQUAKE_ADAPTER" \
  --score-output results/external_benchmark_runs/mquake_wal_memit_n50_seed42_outputs.json \
  --output results/external_benchmark_runs/mquake_wal_memit_n50_seed42_scored.json

PYTHONPATH=src python -m agim.eval.raw_text_edit_pipeline \
  --input data/raw_text_updates.jsonl \
  --model "$AGIM_MODEL" \
  --output results/external_benchmark_runs/raw_text_wal_memit_n50_seed42.json

PYTHONPATH=src python -m agim.eval.product_diagnostic \
  --input results/easyedit_official/current/random_50_seed_42_wal_memit.json \
  --benchmark-name scedit \
  --output results/external_benchmark_runs/product_scedit_wal_memit_n50_seed42.json
```

## Gate 4 — sequential side-slot retention (`n=10/50/100`, seeds `42/43/44`)

```bash
for seed in 42 43 44; do
  PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
    --n 10 --sample-policy random --seed "$seed" \
    --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
    --easyedit-root "$AGIM_EASYEDIT_ROOT" \
    --method-profile-id seq_side_slot_10_seed"${seed}" \
    --edit-backend side_slot --sequential-edit --retention-steps 10 \
    --target-token-mode contextual --use-neg-prompts \
    --save-neighbor-text --save-failures-only \
    --output "results/easyedit_official/sequential/side_slot_random_10_seed_${seed}_seq.json"

  PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
    --n 50 --sample-policy random --seed "$seed" \
    --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
    --easyedit-root "$AGIM_EASYEDIT_ROOT" \
    --method-profile-id seq_side_slot_50_seed"${seed}" \
    --edit-backend side_slot --sequential-edit --retention-steps 50 \
    --target-token-mode contextual --use-neg-prompts \
    --save-neighbor-text --save-failures-only \
    --output "results/easyedit_official/sequential/side_slot_random_50_seed_${seed}_seq.json"

  PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
    --n 100 --sample-policy random --seed "$seed" \
    --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
    --easyedit-root "$AGIM_EASYEDIT_ROOT" \
    --method-profile-id seq_side_slot_100_seed"${seed}" \
    --edit-backend side_slot --sequential-edit --retention-steps 100 \
    --target-token-mode contextual --use-neg-prompts \
    --save-neighbor-text --save-failures-only \
    --output "results/easyedit_official/sequential/side_slot_random_100_seed_${seed}_seq.json"
done
```

## Gate 5 — governance proof packet

```bash
python -m agim.model.patch_service --help
python -m agim.model.patch_governance --help
```

After all gates:
- Update `PATH_B_MAX_EXECUTION_CHECKLIST.md`
- Update `PATH_B_MAX_COMPLETION_MATRIX.md`
- Update `PATH_B_MAX_STATUS_BOARD.md`
- Update `PATH_B_MAX_PROMPT_ARTIFACT_AUDIT.md`
