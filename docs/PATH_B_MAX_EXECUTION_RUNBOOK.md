# Path B Maximal Execution Runbook

Назначение: пошаговый порядок закрытия критических блокеров из
`PATH_B_MAX_EXECUTION_CHECKLIST.md`.

Сопровождающий полный audit (authoritative): [PATH_B_MAX_MAX_AUDIT.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_MAX_AUDIT.md)

Карта блокеров:

- Blocker 1: baseline `wal_memit` на `n=50` (official-compatible).
- Blocker 2: честное backend-сравнение `dual_row`, `wal_rome`, `wal_memit`, `side_slot`.
- Blocker 3: внешняя валидация на уровне RippleEdits / MQuAKE / AKEW / product benchmark.
- Blocker 4: sequential/locality hardening.
- Blocker 5: публичный PatchService/governance.

## Blocker 1. WALMemit quality baseline

1. Выполнить официальный `n=50` baseline для `wal_memit` через `easyedit_official_runner`.
2. Сохранить полный JSON payload с metadata и failures в `results/easyedit_official/current/`.
3. Подтвердить ключевые поля baseline `artifact_schema_version`, `method_profile_id`, `base_model_digest`, `atoms_digest`.
4. Проверить и зафиксировать в `docs/PATH_B_MAX_EXECUTION_CHECKLIST.md`, что пункт `wal_memit n=50 baseline` закрыт.
5. Обновить соответствующий статус в `docs/PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md` (строка Hard-gate 1 / пункт 33).

Пример команды:

Предпочтительный запуск через bootstrap:

```bash
cd /mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory
bash scripts/run_path_b_max_bootstrap.sh 1
```

Ниже — команда single-gate эквивалента для ручного запуска:

```bash
export AGIM_MODEL="meta-llama/Llama-3.1-8B-Instruct"
export AGIM_EASYEDIT_ROOT="<YOUR_EASYEDIT_REPO_PATH>"
export AGIM_RAW_TEXT_INPUT="data/raw_text_updates.jsonl"
export AGIM_DEVICE="cuda:0"
```

```bash
python -m agim.eval.easyedit_official_runner \
  --n 50 --sample-policy random --seed 42 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --easyedit-root "$AGIM_EASYEDIT_ROOT" \
  --nt-sample-size "${AGIM_NT_SAMPLE_SIZE:-500}" \
  --edit-backend wal_memit \
  --target-token-mode contextual \
  --method-profile-id single_loc_wal_memit_n50_seed42 \
  --save-failures-only \
  --output results/easyedit_official/current/random_50_seed_42_wal_memit.json
```

## Blocker 2. Multi-backend quality matrix

1. Запустить matrix compare с `dual_row,wal_rome,wal_memit,side_slot` на одном фиксированном датасете и одном `method_profile_id`.
2. После прогонки обязательно проверить наличия JSON для всех 4 backend (`...dual_row.json`, `...wal_rome.json`, `...wal_memit.json`, `...side_slot.json`) и aggregate report.
3. Подтвердить наличие per-backend JSON + aggregate report в `results/easyedit_official/ablations/`.

Пример команды:

```bash
cd /mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory
bash scripts/run_path_b_max_bootstrap.sh 2
```

Вариант с явным вызовом runner для отладки:

```bash
python -m agim.eval.easyedit_official_runner \
  --sample-policy random --n 50 --seed 42 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --easyedit-root "$AGIM_EASYEDIT_ROOT" \
  --nt-sample-size "${AGIM_NT_SAMPLE_SIZE:-500}" \
  --compare-backends dual_row,wal_rome,wal_memit,side_slot \
  --target-token-mode contextual \
  --method-profile-id matrix_dual_row_wal_rome_wal_memit_side_slot_random_50_seed42 \
  --output results/easyedit_official/ablations/backend_matrix_random_50_seed42.json
```

## Blocker 3. External benchmark evidence

Gate 3 выполняется только при наличии полного набора входов (baseline artifact + adapter + raw-text input). Если обязательные входы/артефакты отсутствуют, выполнение останавливается как hard-fail.

1. RippleEdits: закрыть пост-хок диагностический проход через `ripple_diagnostic` с явно заданным tracked source.
2. MQuAKE: закрыть полный внешне отслеживаемый output + scored report через `mquake_output_runner` + `mquake_diagnostic`.
3. AKEW-style: минимум один tracked raw-text dataset run через `raw_text_edit_pipeline` и зафиксировать output payload.
4. Product benchmark: закрыть один из ScEdit/KnowEdit/UniEdit/MLaKE через `product_diagnostic` с output artifact.
5. Критерий блока: у каждого варианта есть `*.json` payload и ссылка на источник входного датасета/артефакта.

Пример команд (этапно, с реальным dataset path):

```bash
export AGIM_DATASET_RIPPLE="<YOUR_RIPPLE_DATASET_PATH>"
export AGIM_MQUAKE_ADAPTER="<YOUR_MQUAKE_ADAPTER_PATH>"
export AGIM_RAW_TEXT_INPUT="${AGIM_RAW_TEXT_INPUT:-data/raw_text_updates.jsonl}"

python -m agim.eval.ripple_diagnostic \
  --input results/easyedit_official/current/random_50_seed_42_wal_memit.json \
  --output results/external_benchmark_runs/ripple_wal_memit_n50_seed42.json

python -m agim.eval.ripple_diagnostic \
  --dataset-input "$AGIM_DATASET_RIPPLE" \
  --output results/external_benchmark_runs/ripple_adapter.json

python -m agim.eval.mquake_output_runner \
  --model "$AGIM_MODEL" \
  --device "$AGIM_DEVICE" \
  --adapter "$AGIM_MQUAKE_ADAPTER" \
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
```

## Blocker 4. Sequential/locality hardening

1. Side-slot retention для `n=10/50/100` с seed-coverage.
2. Зафиксировать соседние тексты до/после и severity-diff для локальности.
3. Подтвердить, что sequential tuned profile сохраняет traceability и no-regression в non-target budget.
4. Критерий блока: в `results/easyedit_official/sequential/` есть runs для `n=10/50/100` с failure-only breakdown по `tf/ctx_gen/prob`.

Пример запусков через bootstrap:

```bash
cd /mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory
bash scripts/run_path_b_max_bootstrap.sh 4
```

Или ручные команды по каждому шагу:

```bash
python -m agim.eval.easyedit_official_runner \
  --n 10 --sample-policy random --seed 42 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --edit-backend side_slot \
  --method-profile-id seq_side_slot_10_seed42 \
  --sequential-edit \
  --retention-steps 10 \
  --save-failures-only \
    --output results/easyedit_official/sequential/side_slot_random_10_seed_42_seq.json

python -m agim.eval.easyedit_official_runner \
  --n 50 --sample-policy random --seed 42 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --edit-backend side_slot \
  --method-profile-id seq_side_slot_50_seed42 \
  --sequential-edit \
  --retention-steps 50 \
  --save-failures-only \
    --output results/easyedit_official/sequential/side_slot_random_50_seed_42_seq.json

python -m agim.eval.easyedit_official_runner \
  --n 100 --sample-policy random --seed 42 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --edit-backend side_slot \
  --method-profile-id seq_side_slot_100_seed42 \
  --sequential-edit \
  --retention-steps 100 \
  --save-failures-only \
    --output results/easyedit_official/sequential/side_slot_random_100_seed_42_seq.json
```

## Blocker 5. Governance readiness

1. Зафиксировать PatchService lifecycle contract как обязательный public-facing path.
2. Подтянуть approvals/signature/audit checks в release gate checklist.
3. Удалить или переформулировать все незащищенные external leaderboard claims в главных docs.
4. Подготовить публичный контракт и ссылку на него из `README.md`, `BENCHMARK.md`, `docs/CLAIMS_AND_EVIDENCE.md`.
5. Обновить соответствующий статус в `docs/PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md` (строки 39-40).
5. Критерий блока: `PatchService API` и `governance trail` описаны в одном release-gate proof packet.

Пример команд для документирования готовности:

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

`AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER` должен быть реальным внешним immutable backend, а не `mock-object-store`/`local_filesystem`/`local`/`filesystem`.
`AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL` должен быть не localhost/127.0.0.1/0.0.0.0/[::1].

## Non-goal in this runbook

1. Никакие локальные smoke-подходы не меняются в публичный claim path.
2. Не заменяется текущий source-of-truth через частичный rerun без полного артефакта.
