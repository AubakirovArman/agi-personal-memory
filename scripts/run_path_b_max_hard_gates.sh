#!/usr/bin/env bash

set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKDIR"

export AGIM_MODEL="${AGIM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export AGIM_DEVICE="${AGIM_DEVICE:-cuda}"
export AGIM_LOCAL_FILES_ONLY="${AGIM_LOCAL_FILES_ONLY:-0}"
export AGIM_EASYEDIT_ROOT="${AGIM_EASYEDIT_ROOT:?Set AGIM_EASYEDIT_ROOT}"
export AGIM_MQUAKE_ADAPTER="${AGIM_MQUAKE_ADAPTER:-}"
export PYTHONPATH=src

log() {
  echo "[PATH_B_MAX] $*"
}

run_gate_1() {
  log "Running Gate 1: wal_memit official n=50 baseline"
  python -m agim.eval.easyedit_official_runner \
    --n 50 --sample-policy random --seed 42 \
    --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
    --easyedit-root "$AGIM_EASYEDIT_ROOT" \
    --nt-sample-size 500 \
    --edit-backend wal_memit \
    --method-profile-id single_loc_wal_memit_n50_seed42 \
    --target-token-mode contextual \
    --save-failures-only --save-neighbor-text \
    --output results/easyedit_official/current/random_50_seed_42_wal_memit.json \
}

run_gate_2() {
  log "Running Gate 2: backend matrix dual_row/wal_rome/wal_memit/side_slot"
  python -m agim.eval.easyedit_official_runner \
    --dataset random --sample-policy random --n 50 --seed 42 \
    --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
    --easyedit-root "$AGIM_EASYEDIT_ROOT" \
    --nt-sample-size 500 \
    --compare-backends dual_row,wal_rome,wal_memit,side_slot \
    --method-profile-id matrix_dual_row_wal_rome_wal_memit_side_slot_random_50_seed42 \
    --output results/easyedit_official/ablations/backend_matrix_random_50_seed42.json \
    --save-failures-only \
    --save-neighbor-text
}

run_gate_3() {
  log "Running Gate 3: external consequence evidence chain"
  python -m agim.eval.ripple_diagnostic \
    --input results/easyedit_official/current/random_50_seed_42_wal_memit.json \
    --output results/external_benchmark_runs/ripple_wal_memit_n50_seed42.json

  if [[ -n "$AGIM_MQUAKE_ADAPTER" && -f "$AGIM_MQUAKE_ADAPTER" ]]; then
    python -m agim.eval.mquake_output_runner \
      --adapter "$AGIM_MQUAKE_ADAPTER" \
      --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
      --output results/external_benchmark_runs/mquake_wal_memit_n50_seed42_outputs.json
    python -m agim.eval.mquake_diagnostic \
      --score-adapter "$AGIM_MQUAKE_ADAPTER" \
      --score-output results/external_benchmark_runs/mquake_wal_memit_n50_seed42_outputs.json \
      --output results/external_benchmark_runs/mquake_wal_memit_n50_seed42_scored.json
  else
    log "Skipping MQuAKE chain: AGIM_MQUAKE_ADAPTER is expected to point to a MQuAKE adapter JSON."
  fi

  python -m agim.eval.raw_text_edit_pipeline \
    --input data/raw_text_updates.jsonl \
    --model "$AGIM_MODEL" \
    --output results/external_benchmark_runs/raw_text_wal_memit_n50_seed42.json

  python -m agim.eval.product_diagnostic \
    --input results/easyedit_official/current/random_50_seed_42_wal_memit.json \
    --benchmark-name scedit \
    --output results/external_benchmark_runs/product_scedit_wal_memit_n50_seed42.json
}

run_gate_4() {
  log "Running Gate 4: side-slot retention 10/50/100 seeds 42/43/44"
  for seed in 42 43 44; do
    python -m agim.eval.easyedit_official_runner \
      --n 10 --sample-policy random --seed "$seed" \
      --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
      --easyedit-root "$AGIM_EASYEDIT_ROOT" \
      --method-profile-id seq_side_slot_10_seed"${seed}" \
      --edit-backend side_slot --sequential-edit --retention-steps 10 \
      --target-token-mode contextual --use-neg-prompts \
      --save-neighbor-text --save-failures-only \
      --output "results/easyedit_official/sequential/side_slot_random_10_seed_${seed}_seq.json"

    python -m agim.eval.easyedit_official_runner \
      --n 50 --sample-policy random --seed "$seed" \
      --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
      --easyedit-root "$AGIM_EASYEDIT_ROOT" \
      --method-profile-id seq_side_slot_50_seed"${seed}" \
      --edit-backend side_slot --sequential-edit --retention-steps 50 \
      --target-token-mode contextual --use-neg-prompts \
      --save-neighbor-text --save-failures-only \
      --output "results/easyedit_official/sequential/side_slot_random_50_seed_${seed}_seq.json"

    python -m agim.eval.easyedit_official_runner \
      --n 100 --sample-policy random --seed "$seed" \
      --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
      --easyedit-root "$AGIM_EASYEDIT_ROOT" \
      --method-profile-id seq_side_slot_100_seed"${seed}" \
      --edit-backend side_slot --sequential-edit --retention-steps 100 \
      --target-token-mode contextual --use-neg-prompts \
      --save-neighbor-text --save-failures-only \
      --output "results/easyedit_official/sequential/side_slot_random_100_seed_${seed}_seq.json"
  done
}

run_gate_5() {
  log "Running Gate 5: printing PatchService governance command surface"
  python -m agim.model.patch_service --help
  python -m agim.model.patch_governance --help
}

main() {
  local gates=("$@")
  if (( ${#gates[@]} == 0 )); then
    gates=(1 2 3 4 5)
  fi

  for gate in "${gates[@]}"; do
    case "$gate" in
      1) run_gate_1 ;;
      2) run_gate_2 ;;
      3) run_gate_3 ;;
      4) run_gate_4 ;;
      5) run_gate_5 ;;
      *)
        echo "Unknown gate: $gate"
        ;;
    esac
  done
}

main "$@"
