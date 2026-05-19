#!/usr/bin/env bash

set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKDIR"

export AGIM_MODEL="${AGIM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export AGIM_DEVICE="${AGIM_DEVICE:-cuda:0}"
export AGIM_EASYEDIT_ROOT="${AGIM_EASYEDIT_ROOT:?Set AGIM_EASYEDIT_ROOT}"
export AGIM_LOCAL_FILES_ONLY="${AGIM_LOCAL_FILES_ONLY:-0}"
export PYTHONPATH="${PYTHONPATH:-src}"
export DRY_RUN="${DRY_RUN:-0}"
export RELATION_PROFILE_MAP="${AGIM_RELATION_PROFILE_MAP:-${RELATION_PROFILE_MAP:-}}"

usage() {
  cat <<'EOF'
Usage:
  run_path_b_psall_improvement_sweep.sh [--step name]... [--dry-run] [--all]

Known steps:
  baseline-42
  baseline-43
  baseline-44
  selective-anti
  kpos-objective
  kpos-ridge
  objective-balance
  decode-rerank
  relation-aware
  conflict-budget
  sequential-sanity-50
  all

Notes:
  --dry-run prints the commands only.
  relation-aware requires AGIM_RELATION_PROFILE_MAP to point to a JSON mapping file:
  {"P17":{"positive_profile":"w025"}} etc.
EOF
}

DRY_RUN_MODE=0
RUN_ALL_STEPS=0
SELECTED_STEPS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --step)
      if [[ $# -lt 2 ]]; then
        echo "--step requires an argument" >&2
        exit 1
      fi
      SELECTED_STEPS+=("$2")
      shift 2
      ;;
    --dry-run)
      DRY_RUN_MODE=1
      shift
      ;;
    --all)
      RUN_ALL_STEPS=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${RUN_ALL_STEPS}" == "1" ]]; then
  SELECTED_STEPS=(all)
fi

if [[ ${#SELECTED_STEPS[@]} -eq 0 ]]; then
  echo "No step selected. Use --all or one or more --step values."
  usage
  exit 1
fi

run_cmd() {
  local cmd=("$@")
  printf '[PSALL_SWEEP] %q ' "${cmd[@]}"
  printf '\n'
  if [[ "$DRY_RUN_MODE" == "1" || "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "${cmd[@]}"
}

run_baseline() {
  local seed="$1"
  local profile="random_50_seed_${seed}"
  local out="results/easyedit_official/ablations/plan_${profile}.json"
  run_cmd python -m agim.eval.easyedit_official_runner \
    --preset "random_50_seed_${seed}" \
    --output "$out" \
    --target-token-mode contextual \
    --save-failures-only
}

run_selective_anti() {
  local out="results/easyedit_official/ablations/plan_ablation_selective_anti_seed42.json"
  run_cmd python -m agim.eval.easyedit_official_runner \
    --preset "ablation_selective_anti_repetition_seed42" \
    --output "$out" \
    --save-failures-only
}

run_kpos_objective() {
  local out="results/easyedit_official/ablations/plan_ablation_kpos_positive_w025_seed42.json"
  run_cmd python -m agim.eval.easyedit_official_runner \
    --preset "ablation_kpos_positive_w025_seed42" \
    --output "$out" \
    --save-failures-only
}

run_kpos_ridge() {
  local out="results/easyedit_official/ablations/plan_positive_ridge_seed42.json"
  run_cmd python -m agim.eval.easyedit_official_runner \
    --preset "ablation_kpos_kneg_ridge_seed42" \
    --output "$out" \
    --save-failures-only
}

run_objective_balance() {
  local out="results/easyedit_official/ablations/plan_ablation_objective_balance_seed42.json"
  run_cmd python -m agim.eval.easyedit_official_runner \
    --preset "ablation_objective_balance_seed42" \
    --output "$out" \
    --save-failures-only
}

run_decode_rerank() {
  local out="results/easyedit_official/ablations/plan_ablation_decode_rerank_seed42.json"
  run_cmd python -m agim.eval.easyedit_official_runner \
    --preset "ablation_decode_rerank_seed42" \
    --output "$out" \
    --save-failures-only
}

run_relation_aware() {
  if [[ -z "$RELATION_PROFILE_MAP" ]]; then
    echo "AGIM_RELATION_PROFILE_MAP is not set; skipping relation-aware run." >&2
    echo "Set AGIM_RELATION_PROFILE_MAP=/path/to/relation_profiles.json and rerun." >&2
    return 0
  fi
  local out="results/easyedit_official/ablations/plan_ablation_relation_aware_seed42.json"
  run_cmd python -m agim.eval.easyedit_official_runner \
    --preset random_50_seed_42 \
    --relation-profile-map "$RELATION_PROFILE_MAP" \
    --target-token-mode contextual \
    --output "$out" \
    --save-failures-only
}

run_conflict_budget() {
  local out="results/easyedit_official/ablations/plan_norm_budget_no_commit_probe_seed42.json"
  run_cmd python -m agim.eval.easyedit_official_runner \
    --n 50 \
    --sample-policy random \
    --seed 42 \
    --model "$AGIM_MODEL" \
    --device "$AGIM_DEVICE" \
    --easyedit-root "$AGIM_EASYEDIT_ROOT" \
    --local-files-only "$AGIM_LOCAL_FILES_ONLY" \
    --max-row-delta-norm 0.01 \
    --target-token-mode contextual \
    --output "$out" \
    --save-failures-only
}

run_sequential_sanity_50() {
  local out="results/easyedit_official/sequential/plan_random_50_seed_42_seq.json"
  run_cmd python -m agim.eval.easyedit_official_runner \
    --n 50 \
    --sample-policy random \
    --seed 42 \
    --model "$AGIM_MODEL" \
    --device "$AGIM_DEVICE" \
    --easyedit-root "$AGIM_EASYEDIT_ROOT" \
    --local-files-only "$AGIM_LOCAL_FILES_ONLY" \
    --edit-backend side_slot \
    --sequential-edit \
    --retention-steps 10,50 \
    --target-token-mode contextual \
    --use-neg-prompts \
    --output "$out" \
    --save-failures-only
}

run_steps=0
for step in "${SELECTED_STEPS[@]}"; do
  case "$step" in
    baseline-42) run_baseline 42; run_steps=$((run_steps + 1)) ;;
    baseline-43) run_baseline 43; run_steps=$((run_steps + 1)) ;;
    baseline-44) run_baseline 44; run_steps=$((run_steps + 1)) ;;
    selective-anti) run_selective_anti; run_steps=$((run_steps + 1)) ;;
    kpos-objective) run_kpos_objective; run_steps=$((run_steps + 1)) ;;
    kpos-ridge) run_kpos_ridge; run_steps=$((run_steps + 1)) ;;
    objective-balance) run_objective_balance; run_steps=$((run_steps + 1)) ;;
    decode-rerank) run_decode_rerank; run_steps=$((run_steps + 1)) ;;
    relation-aware) run_relation_aware; run_steps=$((run_steps + 1)) ;;
    conflict-budget) run_conflict_budget; run_steps=$((run_steps + 1)) ;;
    sequential-sanity-50) run_sequential_sanity_50; run_steps=$((run_steps + 1)) ;;
    all)
      run_baseline 42; run_baseline 43; run_baseline 44
      run_selective_anti
      run_kpos_objective
      run_kpos_ridge
      run_objective_balance
      run_decode_rerank
      run_relation_aware
      run_conflict_budget
      run_sequential_sanity_50
      run_steps=9
      ;;
    *)
      echo "Unknown step: $step" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$run_steps" -eq 0 ]]; then
  echo "No commands executed. Check requested steps." >&2
  exit 1
fi

echo "[PSALL_SWEEP] done (commands executed or printed: $run_steps)"
