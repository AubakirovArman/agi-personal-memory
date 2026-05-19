#!/usr/bin/env bash

set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKDIR"

readonly PROJECT_ROOT="$WORKDIR"

log() {
  echo "[PATH_B_MAX_BOOTSTRAP] $*"
}

run_gate_runner() {
  local gate_args=("$@")
  bash scripts/run_path_b_max_hard_gates.sh "${gate_args[@]}"
}

show_gpu_usage() {
  local gpus=(2 3)
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi unavailable, skip GPU usage check"
    return
  fi
  for g in "${gpus[@]}"; do
    if nvidia-smi -i "$g" -q -d MEMORY >/dev/null 2>&1; then
      log "GPU ${g} memory snapshot"
      nvidia-smi -i "$g" --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
    else
      log "GPU ${g} is not available in this runtime"
    fi
  done
}

clean_gpu_pids() {
  if [ "${AGIM_DRY_RUN:-0}" = "1" ]; then
    log "AGIM_DRY_RUN=1, skipping GPU cleanup"
    return
  fi

  local force="${AGIM_FORCE_KILL_GPU:-0}"
  if [ "$force" = "0" ]; then
    log "GPU cleanup skipped. Set AGIM_FORCE_KILL_GPU=1 for dry-run, 2 for gentle, 3 for force kill"
    return
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi unavailable, cannot discover GPU processes"
    return
  fi

  local -a all_pids=()
  local -a gpus=(2 3)
  local g pid
  local pids
  for g in "${gpus[@]}"; do
    while IFS= read -r pid; do
      if [[ "$pid" =~ ^[0-9]+$ ]]; then
        all_pids+=("$pid")
      fi
    done < <(
      nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
      | sed 's/[[:space:]]//g'
    )
  done

  pids="$(printf '%s\n' "${all_pids[@]}" | sort -u | tr '\n' ' ')"
  if [ -z "${pids// /}" ]; then
    log "No compute processes detected for GPU 2/3"
    return
  fi

  if [ "$force" = "1" ]; then
    log "GPU process dry-run (AGIM_FORCE_KILL_GPU=1), listing PIDs for GPU 2/3:"
    echo "$pids"
    return
  fi

  log "Killing GPU compute processes (AGIM_FORCE_KILL_GPU=$force) on GPUs 2/3"
  echo "$pids"
  local signal="-9"
  if [ "$force" = "2" ]; then
    signal="-15"
  fi

  for p in $pids; do
    if [ -n "$p" ]; then
      if kill -0 "$p" >/dev/null 2>&1; then
        log "Killing pid ${p} with signal ${signal}"
        kill "$signal" "$p" || true
      else
        log "PID ${p} already exited"
      fi
    fi
  done
}

export AGIM_MODEL="${AGIM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export AGIM_DEVICE="${AGIM_DEVICE:-cuda:0}"
export AGIM_LOCAL_FILES_ONLY="${AGIM_LOCAL_FILES_ONLY:-0}"
export AGIM_EASYEDIT_ROOT="${AGIM_EASYEDIT_ROOT:?Set AGIM_EASYEDIT_ROOT}"
export AGIM_MQUAKE_ADAPTER="${AGIM_MQUAKE_ADAPTER:-}"
export AGIM_RAW_TEXT_INPUT="${AGIM_RAW_TEXT_INPUT:-data/raw_text_updates.jsonl}"
export PYTHONPATH="${PYTHONPATH:-src}"

log "Active env"
log " AGIM_MODEL=$AGIM_MODEL"
log " AGIM_DEVICE=$AGIM_DEVICE"
log " AGIM_LOCAL_FILES_ONLY=$AGIM_LOCAL_FILES_ONLY"
log " AGIM_EASYEDIT_ROOT=$AGIM_EASYEDIT_ROOT"
log " AGIM_MQUAKE_ADAPTER=${AGIM_MQUAKE_ADAPTER:-<unset>}"
log " AGIM_RAW_TEXT_INPUT=$AGIM_RAW_TEXT_INPUT"

show_gpu_usage

if [ "${AGIM_CLEAN_GPU:-0}" = "1" ]; then
  clean_gpu_pids
fi

if [ "$#" -eq 0 ]; then
  run_gate_runner 1 2 4 3 5
else
  run_gate_runner "$@"
fi
