#!/usr/bin/env bash

set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKDIR"

export AGIM_MODEL="${AGIM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export AGIM_DEVICE="${AGIM_DEVICE:-cuda:0}"
export AGIM_LOCAL_FILES_ONLY="${AGIM_LOCAL_FILES_ONLY:-0}"
export AGIM_EASYEDIT_ROOT="${AGIM_EASYEDIT_ROOT:?Set AGIM_EASYEDIT_ROOT}"
export AGIM_MQUAKE_ADAPTER="${AGIM_MQUAKE_ADAPTER:-}"
export AGIM_RAW_TEXT_INPUT="${AGIM_RAW_TEXT_INPUT:-data/raw_text_updates.jsonl}"
export PYTHONPATH=src
export AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL="${AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL:-0}"
export AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER="${AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER:-}"
export AGIM_GATE5_TRANSPORT_IMMUTABILITY_MODE="${AGIM_GATE5_TRANSPORT_IMMUTABILITY_MODE:-}"
export AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL="${AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL:-}"
export AGIM_GATE5_TRANSPORT_RETENTION_POLICY="${AGIM_GATE5_TRANSPORT_RETENTION_POLICY:-workspace-immutable}"
export AGIM_GATE5_PUBLIC_API_SMOKE="${AGIM_GATE5_PUBLIC_API_SMOKE:-0}"
export AGIM_GATE5_PUBLIC_API_PORT="${AGIM_GATE5_PUBLIC_API_PORT:-8010}"

log() {
  echo "[PATH_B_MAX] $*"
}

require_file() {
  local file="$1"
  if [ -f "$file" ]; then
    log "OK artifact: $file"
  else
    log "MISS artifact: $file"
  fi
}

require_file_if() {
  local cond="$1"
  local file="$2"
  if [ "$cond" = "1" ]; then
    require_file "$file"
  else
    log "SKIP optional artifact: $file"
  fi
}

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    log "Missing required environment variable: $name"
    return 1
  fi
}

sha256_file() {
  local path="$1"
  python - "$path" <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
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
    --save-failures-only \
    --output results/easyedit_official/current/random_50_seed_42_wal_memit.json

  require_file results/easyedit_official/current/random_50_seed_42_wal_memit.json
  require_file results/easyedit_official/current/random_50_seed_42_wal_memit.failures.json
}

run_gate_2() {
  log "Running Gate 2: backend matrix dual_row/wal_rome/wal_memit/side_slot"
  python -m agim.eval.easyedit_official_runner \
    --sample-policy random --n 50 --seed 42 \
    --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
    --easyedit-root "$AGIM_EASYEDIT_ROOT" \
    --nt-sample-size 500 \
    --compare-backends dual_row,wal_rome,wal_memit,side_slot \
    --method-profile-id matrix_dual_row_wal_rome_wal_memit_side_slot_random_50_seed42 \
    --output results/easyedit_official/ablations/backend_matrix_random_50_seed42.json \
    --save-failures-only

  require_file results/easyedit_official/ablations/backend_matrix_random_50_seed42.json
  require_file results/easyedit_official/ablations/backend_matrix_random_50_seed42.dual_row.json
  require_file results/easyedit_official/ablations/backend_matrix_random_50_seed42.wal_rome.json
  require_file results/easyedit_official/ablations/backend_matrix_random_50_seed42.wal_memit.json
  require_file results/easyedit_official/ablations/backend_matrix_random_50_seed42.side_slot.json
  require_file results/easyedit_official/ablations/backend_matrix_random_50_report_2026-05-18.md
}

run_gate_3() {
  log "Running Gate 3: external consequence evidence chain"
  require_file results/easyedit_official/current/random_50_seed_42_wal_memit.json
  require_env AGIM_MQUAKE_ADAPTER
  if [ ! -f "$AGIM_MQUAKE_ADAPTER" ]; then
    log "Missing AGIM_MQUAKE_ADAPTER file: $AGIM_MQUAKE_ADAPTER"
    return 1
  fi
  if [ ! -f "$AGIM_RAW_TEXT_INPUT" ]; then
    log "Missing AGIM_RAW_TEXT_INPUT file: $AGIM_RAW_TEXT_INPUT"
    return 1
  fi
  python -m agim.eval.ripple_diagnostic \
    --input results/easyedit_official/current/random_50_seed_42_wal_memit.json \
    --output results/external_benchmark_runs/ripple_wal_memit_n50_seed42.json

  python -m agim.eval.mquake_output_runner \
    --adapter "$AGIM_MQUAKE_ADAPTER" \
    --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
    --output results/external_benchmark_runs/mquake_wal_memit_n50_seed42_outputs.json
  python -m agim.eval.mquake_diagnostic \
    --score-adapter "$AGIM_MQUAKE_ADAPTER" \
    --score-output results/external_benchmark_runs/mquake_wal_memit_n50_seed42_outputs.json \
    --output results/external_benchmark_runs/mquake_wal_memit_n50_seed42_scored.json

  python -m agim.eval.raw_text_edit_pipeline \
    --input "$AGIM_RAW_TEXT_INPUT" \
    --output results/external_benchmark_runs/raw_text_wal_memit_n50_seed42.json

  python -m agim.eval.product_diagnostic \
    --input results/easyedit_official/current/random_50_seed_42_wal_memit.json \
    --benchmark-name scedit \
    --output results/external_benchmark_runs/product_scedit_wal_memit_n50_seed42.json

  require_file results/external_benchmark_runs/ripple_wal_memit_n50_seed42.json
  require_file results/external_benchmark_runs/mquake_wal_memit_n50_seed42_outputs.json
  require_file results/external_benchmark_runs/mquake_wal_memit_n50_seed42_scored.json
  require_file results/external_benchmark_runs/raw_text_wal_memit_n50_seed42.json
  require_file results/external_benchmark_runs/product_scedit_wal_memit_n50_seed42.json
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
      --save-failures-only \
      --output "results/easyedit_official/sequential/side_slot_random_10_seed_${seed}_seq.json"

    python -m agim.eval.easyedit_official_runner \
      --n 50 --sample-policy random --seed "$seed" \
      --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
      --easyedit-root "$AGIM_EASYEDIT_ROOT" \
      --method-profile-id seq_side_slot_50_seed"${seed}" \
      --edit-backend side_slot --sequential-edit --retention-steps 50 \
      --target-token-mode contextual --use-neg-prompts \
      --save-failures-only \
      --output "results/easyedit_official/sequential/side_slot_random_50_seed_${seed}_seq.json"

    python -m agim.eval.easyedit_official_runner \
      --n 100 --sample-policy random --seed "$seed" \
      --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
      --easyedit-root "$AGIM_EASYEDIT_ROOT" \
      --method-profile-id seq_side_slot_100_seed"${seed}" \
      --edit-backend side_slot --sequential-edit --retention-steps 100 \
      --target-token-mode contextual --use-neg-prompts \
      --save-failures-only \
      --output "results/easyedit_official/sequential/side_slot_random_100_seed_${seed}_seq.json"

    require_file "results/easyedit_official/sequential/side_slot_random_10_seed_${seed}_seq.json"
    require_file "results/easyedit_official/sequential/side_slot_random_10_seed_${seed}_seq.failures.json"
    require_file "results/easyedit_official/sequential/side_slot_random_50_seed_${seed}_seq.json"
    require_file "results/easyedit_official/sequential/side_slot_random_50_seed_${seed}_seq.failures.json"
    require_file "results/easyedit_official/sequential/side_slot_random_100_seed_${seed}_seq.json"
    require_file "results/easyedit_official/sequential/side_slot_random_100_seed_${seed}_seq.failures.json"
  done
}

run_gate_5() {
  log "Running Gate 5: generating PatchService/Governance proof packet"
  local release_sha256=""
  local receipt_sha256=""
  local bundle_sha256=""
  local transport_manifest_sha256=""

  python scripts/run_path_b_max_patch_service_governance_proof.py
  require_file results/easyedit_official/governance/path_b_max_gate5_proof.json
  python scripts/run_path_b_max_gate5_release_packet.py \
    --proof-path results/easyedit_official/governance/path_b_max_gate5_proof.json
  python scripts/run_path_b_max_verify_gate5_release.py \
    --tenant public \
    --release-path results/easyedit_official/governance/path_b_max_gate5_public_release.json
  python scripts/run_path_b_max_publish_gate5_release.py \
    --release-path results/easyedit_official/governance/path_b_max_gate5_public_release.json \
    --tenant public \
    --channel public
  python scripts/run_path_b_max_gate5_create_receipt.py \
    --index-path results/easyedit_official/governance/path_b_max_gate5_public_index.json \
    --receipt-path results/easyedit_official/governance/path_b_max_gate5_public_receipt.json \
    --tenant public \
    --channel public
  python scripts/run_path_b_max_verify_gate5_index.py \
    --tenant public \
    --channel public
  python scripts/run_path_b_max_verify_gate5_receipt.py \
    --receipt-path results/easyedit_official/governance/path_b_max_gate5_public_receipt.json
  python scripts/run_path_b_max_gate5_create_bundle.py \
    --index-path results/easyedit_official/governance/path_b_max_gate5_public_index.json \
    --receipt-path results/easyedit_official/governance/path_b_max_gate5_public_receipt.json \
    --bundle-path results/easyedit_official/governance/path_b_max_gate5_public_bundle.json \
    --tenant public \
    --channel public
  python scripts/run_path_b_max_verify_gate5_bundle.py \
    --bundle-path results/easyedit_official/governance/path_b_max_gate5_public_bundle.json
  require_file results/easyedit_official/governance/path_b_max_gate5_public_release.json
  require_file results/easyedit_official/governance/path_b_max_gate5_public_index.json
  require_file results/easyedit_official/governance/path_b_max_gate5_public_receipt.json
  require_file results/easyedit_official/governance/path_b_max_gate5_public_bundle.json
  if [[ "${AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL}" == "1" ]]; then
    require_env AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER
    require_env AGIM_GATE5_TRANSPORT_IMMUTABILITY_MODE
    require_env AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL
    python scripts/run_path_b_max_gate5_create_transport_manifest.py \
      --release-path results/easyedit_official/governance/path_b_max_gate5_public_release.json \
      --index-path results/easyedit_official/governance/path_b_max_gate5_public_index.json \
      --receipt-path results/easyedit_official/governance/path_b_max_gate5_public_receipt.json \
      --bundle-path results/easyedit_official/governance/path_b_max_gate5_public_bundle.json \
      --manifest-path results/easyedit_official/governance/path_b_max_gate5_public_transport_manifest.json \
      --tenant public \
      --channel public \
      ${AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL:+--public-base-url "$AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL"} \
      --storage-provider "${AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER}" \
      --immutability-mode "${AGIM_GATE5_TRANSPORT_IMMUTABILITY_MODE}" \
      --retention-policy "${AGIM_GATE5_TRANSPORT_RETENTION_POLICY}" \
      --require-production-external
    python scripts/run_path_b_max_verify_gate5_transport_manifest.py \
      --manifest-path results/easyedit_official/governance/path_b_max_gate5_public_transport_manifest.json \
      --require-production-external
  else
    python scripts/run_path_b_max_gate5_create_transport_manifest.py \
      --release-path results/easyedit_official/governance/path_b_max_gate5_public_release.json \
      --index-path results/easyedit_official/governance/path_b_max_gate5_public_index.json \
      --receipt-path results/easyedit_official/governance/path_b_max_gate5_public_receipt.json \
      --bundle-path results/easyedit_official/governance/path_b_max_gate5_public_bundle.json \
      --manifest-path results/easyedit_official/governance/path_b_max_gate5_public_transport_manifest.json \
      --tenant public \
      --channel public \
      ${AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL:+--public-base-url "$AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL"} \
      --storage-provider "${AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER:-local_filesystem}" \
      --immutability-mode "${AGIM_GATE5_TRANSPORT_IMMUTABILITY_MODE:-local_only}"
    python scripts/run_path_b_max_verify_gate5_transport_manifest.py \
      --manifest-path results/easyedit_official/governance/path_b_max_gate5_public_transport_manifest.json
  fi
  require_file results/easyedit_official/governance/path_b_max_gate5_public_transport_manifest.json
  release_sha256="$(sha256_file results/easyedit_official/governance/path_b_max_gate5_public_release.json)"
  receipt_sha256="$(sha256_file results/easyedit_official/governance/path_b_max_gate5_public_receipt.json)"
  bundle_sha256="$(sha256_file results/easyedit_official/governance/path_b_max_gate5_public_bundle.json)"
  transport_manifest_sha256="$(sha256_file results/easyedit_official/governance/path_b_max_gate5_public_transport_manifest.json)"
  if [[ "${AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL}" == "1" ]]; then
    python scripts/run_path_b_max_gate5_verify_publication.py \
      --release-path results/easyedit_official/governance/path_b_max_gate5_public_release.json \
      --index-path results/easyedit_official/governance/path_b_max_gate5_public_index.json \
      --receipt-path results/easyedit_official/governance/path_b_max_gate5_public_receipt.json \
      --bundle-path results/easyedit_official/governance/path_b_max_gate5_public_bundle.json \
      --transport-manifest-path results/easyedit_official/governance/path_b_max_gate5_public_transport_manifest.json \
      --expected-release-sha256 "$release_sha256" \
      --expected-receipt-sha256 "$receipt_sha256" \
      --expected-bundle-sha256 "$bundle_sha256" \
      --expected-transport-manifest-sha256 "$transport_manifest_sha256" \
      --tenant public \
      --channel public \
      --require-production-external
  else
    python scripts/run_path_b_max_gate5_verify_publication.py \
      --release-path results/easyedit_official/governance/path_b_max_gate5_public_release.json \
      --index-path results/easyedit_official/governance/path_b_max_gate5_public_index.json \
      --receipt-path results/easyedit_official/governance/path_b_max_gate5_public_receipt.json \
      --bundle-path results/easyedit_official/governance/path_b_max_gate5_public_bundle.json \
      --transport-manifest-path results/easyedit_official/governance/path_b_max_gate5_public_transport_manifest.json \
      --expected-release-sha256 "$release_sha256" \
      --expected-receipt-sha256 "$receipt_sha256" \
      --expected-bundle-sha256 "$bundle_sha256" \
      --expected-transport-manifest-sha256 "$transport_manifest_sha256" \
      --tenant public \
      --channel public
  fi
  if [[ "${AGIM_GATE5_PUBLIC_API_SMOKE}" == "1" ]]; then
    local api_base="http://127.0.0.1:${AGIM_GATE5_PUBLIC_API_PORT}"
    local api_port="${AGIM_GATE5_PUBLIC_API_PORT}"
    local api_verify_exit=0
    local public_api_args=(
      python scripts/run_path_b_max_gate5_public_api.py
      --index-path results/easyedit_official/governance/path_b_max_gate5_public_index.json
      --tenant public
      --channel public
      --receipt-path results/easyedit_official/governance/path_b_max_gate5_public_receipt.json
      --bundle-path results/easyedit_official/governance/path_b_max_gate5_public_bundle.json
      --transport-manifest-path results/easyedit_official/governance/path_b_max_gate5_public_transport_manifest.json
      --port "$api_port"
    )
    if [[ "${AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL}" == "1" ]]; then
      public_api_args+=(--require-production-external)
    fi
    "${public_api_args[@]}" &
    local api_pid=$!
    sleep 2
    if [[ "${AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL}" == "1" ]]; then
      if python scripts/run_path_b_max_gate5_verify_publication.py \
        --api-base "$api_base" \
        --release-path results/easyedit_official/governance/path_b_max_gate5_public_release.json \
        --index-path results/easyedit_official/governance/path_b_max_gate5_public_index.json \
        --receipt-path results/easyedit_official/governance/path_b_max_gate5_public_receipt.json \
        --bundle-path results/easyedit_official/governance/path_b_max_gate5_public_bundle.json \
        --transport-manifest-path results/easyedit_official/governance/path_b_max_gate5_public_transport_manifest.json \
        --expected-release-sha256 "$release_sha256" \
        --expected-receipt-sha256 "$receipt_sha256" \
        --expected-bundle-sha256 "$bundle_sha256" \
        --expected-release-schema-version path_b_max_gate5_release.v1 \
        --expected-receipt-schema-version path_b_max_gate5_receipt.v1 \
        --expected-bundle-schema-version path_b_max_gate5_bundle.v1 \
        --expected-transport-manifest-sha256 "$transport_manifest_sha256" \
        --tenant public \
        --channel public \
        --check-transport-manifest \
        --expected-transport-manifest-schema-version path_b_max_gate5_transport_manifest.v1 \
        --require-production-external; then
        :
      else
        api_verify_exit=$?
      fi
    else
      if python scripts/run_path_b_max_gate5_verify_publication.py \
        --api-base "$api_base" \
        --release-path results/easyedit_official/governance/path_b_max_gate5_public_release.json \
        --index-path results/easyedit_official/governance/path_b_max_gate5_public_index.json \
        --receipt-path results/easyedit_official/governance/path_b_max_gate5_public_receipt.json \
        --bundle-path results/easyedit_official/governance/path_b_max_gate5_public_bundle.json \
        --transport-manifest-path results/easyedit_official/governance/path_b_max_gate5_public_transport_manifest.json \
        --expected-release-sha256 "$release_sha256" \
        --expected-receipt-sha256 "$receipt_sha256" \
        --expected-bundle-sha256 "$bundle_sha256" \
        --expected-release-schema-version path_b_max_gate5_release.v1 \
        --expected-receipt-schema-version path_b_max_gate5_receipt.v1 \
        --expected-bundle-schema-version path_b_max_gate5_bundle.v1 \
        --expected-transport-manifest-sha256 "$transport_manifest_sha256" \
        --tenant public \
        --channel public \
        --check-transport-manifest \
        --expected-transport-manifest-schema-version path_b_max_gate5_transport_manifest.v1; then
        :
      else
        api_verify_exit=$?
      fi
    fi
    kill "$api_pid" 2>/dev/null || true
    if [[ "$api_verify_exit" != "0" ]]; then
      return "$api_verify_exit"
    fi
  fi
}

main() {
  local gates=("$@")
  if (( ${#gates[@]} == 0 )); then
    gates=(1 2 4 3 5)
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
