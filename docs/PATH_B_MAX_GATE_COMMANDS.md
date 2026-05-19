# Path B Gate Commands (Execution Pack)

Назначение: не запускать эксперименты автоматически из этого документа, а сохранить
фиксированный playbook с командами и ожидаемыми артефактами для hard-гейтов.

Запуск по всем gate-командам идёт через:

```bash
cd /mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory
bash scripts/run_path_b_max_bootstrap.sh 1 2 4 3 5
```

Если открыт только Gate 5:

```bash
cd /mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory
export AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL=1
export AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER="<PRODUCTION_IMMUTABLE_PROVIDER>"
export AGIM_GATE5_TRANSPORT_IMMUTABILITY_MODE="object_lock"
export AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL="https://<PUBLIC_HOST>/api"
export AGIM_GATE5_PUBLIC_API_SMOKE=1
bash scripts/run_path_b_max_bootstrap.sh 5
```

`AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER` должен быть реальным external immutable backend (не `mock-object-store`, не файловый/local provider).  
`AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL` должен быть не localhost/127.0.0.1/0.0.0.0/[::1].

Ниже — конкретные canonical команды для каждого gate.

## Gate 1 — wal_memit official n=50 baseline

```bash
export AGIM_MODEL="meta-llama/Llama-3.1-8B-Instruct"
export AGIM_EASYEDIT_ROOT="<YOUR_EASYEDIT_REPO_PATH>"
export AGIM_DEVICE="cuda:0"
export AGIM_RAW_TEXT_INPUT="data/raw_text_updates.jsonl"
export AGIM_MQUAKE_ADAPTER="<YOUR_MQUAKE_ADAPTER_PATH>"
python -m agim.eval.easyedit_official_runner \
  --n 50 \
  --sample-policy random \
  --seed 42 \
  --model "$AGIM_MODEL" \
  --edit-backend wal_memit \
  --method-profile-id single_loc_wal_memit_n50_seed42 \
  --target-token-mode contextual \
  --nt-sample-size 500 \
  --device "$AGIM_DEVICE" \
  --easyedit-root "$AGIM_EASYEDIT_ROOT" \
  --save-failures-only \
    --output results/easyedit_official/current/random_50_seed_42_wal_memit.json
```

Ожидаемый evidence:
- `results/easyedit_official/current/random_50_seed_42_wal_memit.json`
- payload fields: `artifact_schema_version`, `method_profile_id`, `base_model_digest`, `atoms_digest`

Если нужен дополнительный профиль:
- `--method-profile-id single_ps_wal_memit_n50_seed42` аналогично.

## Gate 2 — full backend matrix на одном датасете

```bash
python -m agim.eval.easyedit_official_runner \
  --sample-policy random \
  --n 50 \
  --seed 42 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --device "$AGIM_DEVICE" \
  --easyedit-root "$AGIM_EASYEDIT_ROOT" \
  --nt-sample-size "${AGIM_NT_SAMPLE_SIZE:-500}" \
  --compare-backends dual_row,wal_rome,wal_memit,side_slot \
  --target-token-mode contextual \
  --method-profile-id matrix_dual_row_wal_rome_wal_memit_side_slot_random_50_seed42 \
  --save-failures-only \
  --output results/easyedit_official/ablations/backend_matrix_random_50_seed42.json
```

Ожидаемый evidence:
- `results/easyedit_official/ablations/backend_matrix_*_n50_seed42*.json` (matrix + backend split `.dual_row.json`, `.wal_rome.json`, `.wal_memit.json`, `.side_slot.json`)
- `results/easyedit_official/ablations/backend_matrix_random_50_report_2026-05-18.md`
- сравнение backends на одном seed/policy/dataset без смешения метод-профилей

## Gate 3 — external model-output chain (minimum)

### RippleEdits

```bash
python -m agim.eval.ripple_diagnostic \
  --input results/easyedit_official/current/random_50_seed_42_wal_memit.json \
  --output results/external_benchmark_runs/ripple_wal_memit_n50_seed42.json
```

### MQuAKE

```bash
python -m agim.eval.mquake_output_runner \
  --model "$AGIM_MODEL" \
  --device "$AGIM_DEVICE" \
  --adapter "$AGIM_MQUAKE_ADAPTER" \
  --output results/external_benchmark_runs/mquake_wal_memit_n50_seed42_outputs.json

python -m agim.eval.mquake_diagnostic \
  --score-adapter "$AGIM_MQUAKE_ADAPTER" \
  --score-output results/external_benchmark_runs/mquake_wal_memit_n50_seed42_outputs.json \
  --output results/external_benchmark_runs/mquake_wal_memit_n50_seed42_scored.json
```

### Raw-text

```bash
python -m agim.eval.raw_text_edit_pipeline \
  --input "$AGIM_RAW_TEXT_INPUT" \
  --output results/external_benchmark_runs/raw_text_wal_memit_n50_seed42.json
```

### Product-like benchmark

```bash
python -m agim.eval.product_diagnostic \
  --input results/easyedit_official/current/random_50_seed_42_wal_memit.json \
  --benchmark-name scedit \
  --output results/external_benchmark_runs/product_scedit_wal_memit_n50_seed42.json
```

## Gate 4 — side-slot retention hardening (10/50/100)

```bash
python -m agim.eval.easyedit_official_runner \
  --n 10 --sample-policy random --seed 42 --model meta-llama/Llama-3.1-8B-Instruct \
  --device "$AGIM_DEVICE" \
  --method-profile-id seq_side_slot_10_seed42 \
  --sequential-edit --retention-steps 10 --save-failures-only   --output results/easyedit_official/sequential/side_slot_random_10_seed_42_seq.json

python -m agim.eval.easyedit_official_runner \
  --n 50 --sample-policy random --seed 42 --model meta-llama/Llama-3.1-8B-Instruct \
  --device "$AGIM_DEVICE" \
  --method-profile-id seq_side_slot_50_seed42 \
  --sequential-edit --retention-steps 50 --save-failures-only   --output results/easyedit_official/sequential/side_slot_random_50_seed_42_seq.json

python -m agim.eval.easyedit_official_runner \
  --n 100 --sample-policy random --seed 42 --model meta-llama/Llama-3.1-8B-Instruct \
  --device "$AGIM_DEVICE" \
  --method-profile-id seq_side_slot_100_seed42 \
  --sequential-edit --retention-steps 100 --save-failures-only   --output results/easyedit_official/sequential/side_slot_random_100_seed_42_seq.json
```

Повторить те же команды для `--seed 43` и `--seed 44`.

## Gate 5 — proof packet

- Зафиксировать пакет документации и артефактов:
  - public contract для `PatchService` lifecycle;
  - governance chain с подписью/approve/reject;
  - пример `propose→simulate→run_canaries→approve→apply→rollback→inspect/diff`.
- Обновить `docs/CLAIMS_AND_EVIDENCE.md` после получения каждого артефакта.

Local packet build (если нужно пересобрать proof):

```bash
python scripts/run_path_b_max_patch_service_governance_proof.py
```

Production external execution:

```bash
export AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL=1
export AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER="<PRODUCTION_IMMUTABLE_PROVIDER>"
export AGIM_GATE5_TRANSPORT_IMMUTABILITY_MODE="object_lock"
export AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL="https://<PUBLIC_HOST>/api"
export AGIM_GATE5_PUBLIC_API_SMOKE=1
bash scripts/run_path_b_max_bootstrap.sh 5
```
