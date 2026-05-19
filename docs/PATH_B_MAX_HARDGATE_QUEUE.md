# Path B Hard-Gate Execution Queue

Use this queue to execute remaining completion gates in order.
All commands assume `sites/agi_personal_memory` as the active project context.

Recommended order for the current objective:
1. `scripts/run_path_b_max_bootstrap.sh 1`
2. `scripts/run_path_b_max_bootstrap.sh 2`
3. `scripts/run_path_b_max_bootstrap.sh 4`
4. `scripts/run_path_b_max_bootstrap.sh 3`
5. `scripts/run_path_b_max_bootstrap.sh 5`

If only Gate 5 is still open, run:
1. `scripts/run_path_b_max_bootstrap.sh 5` with production external env.

```bash
cd /mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory
export AGIM_MODEL="meta-llama/Llama-3.1-8B-Instruct"
export AGIM_DEVICE="cuda:0"
export AGIM_LOCAL_FILES_ONLY=0
export AGIM_EASYEDIT_ROOT="<YOUR_EASYEDIT_REPO_PATH>"
export AGIM_RAW_TEXT_INPUT="data/raw_text_updates.jsonl"
export AGIM_MQUAKE_ADAPTER="<YOUR_MQUAKE_ADAPTER_PATH>"
```

Recommended entrypoint for all gates:

```bash
bash scripts/run_path_b_max_bootstrap.sh
```

Optional cleanup on GPU 2/3 before running gates:

```bash
AGIM_CLEAN_GPU=1 AGIM_FORCE_KILL_GPU=1 bash scripts/run_path_b_max_bootstrap.sh
AGIM_CLEAN_GPU=1 AGIM_FORCE_KILL_GPU=2 bash scripts/run_path_b_max_bootstrap.sh   # SIGTERM
AGIM_CLEAN_GPU=1 AGIM_FORCE_KILL_GPU=3 bash scripts/run_path_b_max_bootstrap.sh   # SIGKILL
```

`AGIM_FORCE_KILL_GPU=0` skips cleanup.  
`AGIM_FORCE_KILL_GPU=1` prints detected PIDs only (dry-run).

## Gate 1 — wal_memit official baseline (`n=50`)

```bash
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --n 50 --sample-policy random --seed 42 \
  --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
  --easyedit-root "$AGIM_EASYEDIT_ROOT" \
  --target-token-mode contextual \
  --nt-sample-size 500 \
  --edit-backend wal_memit \
  --method-profile-id single_loc_wal_memit_n50_seed42 \
  --save-failures-only \
  --output results/easyedit_official/current/random_50_seed_42_wal_memit.json
```

## Gate 2 — backend quality matrix (`dual_row, wal_rome, wal_memit, side_slot`)

```bash
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --sample-policy random --n 50 --seed 42 \
  --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
  --easyedit-root "$AGIM_EASYEDIT_ROOT" \
  --target-token-mode contextual \
  --nt-sample-size 500 \
  --compare-backends dual_row,wal_rome,wal_memit,side_slot \
  --method-profile-id matrix_dual_row_wal_rome_wal_memit_side_slot_random_50_seed42 \
  --output results/easyedit_official/ablations/backend_matrix_random_50_seed42.json \
  --save-failures-only
```

Expected matrix artifacts:
- `results/easyedit_official/ablations/backend_matrix_random_50_report_2026-05-18.md`
- `results/easyedit_official/ablations/backend_matrix_random_50_seed42.dual_row.json`
- `results/easyedit_official/ablations/backend_matrix_random_50_seed42.wal_rome.json`
- `results/easyedit_official/ablations/backend_matrix_random_50_seed42.wal_memit.json`
- `results/easyedit_official/ablations/backend_matrix_random_50_seed42.side_slot.json`

Legacy artifacts currently exist (for reference only, not as gate-complete):
- `results/easyedit_official/ablations/backend_matrix_random_50_seed42_dual_walrome*.json`
- `results/easyedit_official/ablations/backend_matrix_smoke_5_first42*.json`

## Gate 3 — external consequence evidence chain

This gate is hard-fail when source artifacts or required envs are missing.

```bash
PYTHONPATH=src python -m agim.eval.ripple_diagnostic \
  --input results/easyedit_official/current/random_50_seed_42_wal_memit.json \
  --output results/external_benchmark_runs/ripple_wal_memit_n50_seed42.json

PYTHONPATH=src python -m agim.eval.mquake_output_runner \
  --model "$AGIM_MODEL" \
  --device "$AGIM_DEVICE" \
  --adapter "$AGIM_MQUAKE_ADAPTER" \
  --output results/external_benchmark_runs/mquake_wal_memit_n50_seed42_outputs.json

PYTHONPATH=src python -m agim.eval.mquake_diagnostic \
  --score-adapter "$AGIM_MQUAKE_ADAPTER" \
  --score-output results/external_benchmark_runs/mquake_wal_memit_n50_seed42_outputs.json \
  --output results/external_benchmark_runs/mquake_wal_memit_n50_seed42_scored.json

PYTHONPATH=src python -m agim.eval.raw_text_edit_pipeline \
  --input "$AGIM_RAW_TEXT_INPUT" \
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
    --save-failures-only \
    --output "results/easyedit_official/sequential/side_slot_random_10_seed_${seed}_seq.json"

  PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
    --n 50 --sample-policy random --seed "$seed" \
    --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
    --easyedit-root "$AGIM_EASYEDIT_ROOT" \
    --method-profile-id seq_side_slot_50_seed"${seed}" \
    --edit-backend side_slot --sequential-edit --retention-steps 50 \
    --target-token-mode contextual --use-neg-prompts \
    --save-failures-only \
    --output "results/easyedit_official/sequential/side_slot_random_50_seed_${seed}_seq.json"

  PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
    --n 100 --sample-policy random --seed "$seed" \
    --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
    --easyedit-root "$AGIM_EASYEDIT_ROOT" \
    --method-profile-id seq_side_slot_100_seed"${seed}" \
    --edit-backend side_slot --sequential-edit --retention-steps 100 \
    --target-token-mode contextual --use-neg-prompts \
    --save-failures-only \
    --output "results/easyedit_official/sequential/side_slot_random_100_seed_${seed}_seq.json"
done
```

## Gate 5 — governance proof packet

Local packet build (если нужно пересобрать proof):

```bash
python scripts/run_path_b_max_patch_service_governance_proof.py
```

Production external execution (закрывает Gate 5 полностью):

```bash
export AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL=1
export AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER="<PRODUCTION_IMMUTABLE_PROVIDER>"
export AGIM_GATE5_TRANSPORT_IMMUTABILITY_MODE="object_lock"
export AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL="https://<PUBLIC_HOST>/api"
export AGIM_GATE5_PUBLIC_API_SMOKE=1
bash scripts/run_path_b_max_bootstrap.sh 5
```

`AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER` должен быть реальным внешним immutable backend, а не `mock-object-store`/`local_filesystem`/`local`/`filesystem`.
`AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL` должен быть не localhost/127.0.0.1/0.0.0.0/[::1].

After all gates:
- Update `PATH_B_MAX_EXECUTION_CHECKLIST.md`
- Update `PATH_B_MAX_COMPLETION_MATRIX.md`
- Update `PATH_B_MAX_STATUS_BOARD.md`
- Update `PATH_B_MAX_PROMPT_ARTIFACT_AUDIT.md`
