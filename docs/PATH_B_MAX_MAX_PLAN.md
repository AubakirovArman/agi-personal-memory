# Path B Максимальный План По Maximal Plan (Deep-Research v5)

Дата: 2026-05-18  
Источник: `/mnt/hf_model_weights/arman/3bit/sites/deep-research-report (5).md`

Цель: закрыть все 40 требований через доказательные артефакты, а не по статусам “есть план”.

## Жёсткая Карта: Промпт/Требование → Артефакт → Статус

| # | Требование | Артефакт / контроль | Статус | Критерий закрытия |
| --- | --- | --- | --- | --- |
| 1 | Развести source-of-truth в один активный слой | `CURRENT_STATUS.md`, `BENCHMARK.md`, `docs/CLAIMS_AND_EVIDENCE.md`, `docs/PATH_B_COMPLETION_AUDIT.md` | Done | Любой новый публичный claim подтверждается только по этим трекам |
| 2 | Ясное разделение Path A / Path B | `README.md`, `docs/PATH_B_WEIGHT_EDITING.md`, `docs/PATH_B_PRODUCTIZATION_PLAN.md` | Done | README больше не смешивает runtime-память и weight-editing |
| 3 | 3-track verify | `docs/HOW_TO_VERIFY.md`, `docs/VERIFY_PATH_B_CURRENT.md`, `docs/VERIFY_PATH_B_LEGACY.md`, `docs/VERIFY_PATH_A.md` | Done | Новые инструкции ссылаются только на текущие треки |
| 4 | Явно пометить legacy WALWeight/ROME | `docs/PATH_B_COMPLETION_AUDIT.md`, `results/local_protocol/` | Partially Done | Legacy результаты помечены историческими и не используются как official |
| 5 | Метод-профили и `method_profile_id` | `src/agim/eval/easyedit_run_metadata.py`, `src/agim/eval/easyedit_official_runner.py` | Done | Каждый официальный run содержит `method_profile_id` |
| 6 | `artifact_schema_version` | `src/agim/eval/easyedit_run_metadata.py` | Done | Артефакты из current-пути содержат `artifact_schema_version` |
| 7 | Удалить machine-specific defaults из public guidance | `docs/VERIFY_PATH_B_CURRENT.md`, `src/agim/eval/easyedit_cli.py` | Done | `cuda:2/3`, локальные пути и local-files-only больше не заданы “по умолчанию” |
| 8 | Repro bundle для baseline и sequential | `results/easyedit_official/current/`, `results/easyedit_official/sequential/` | Done | У каждого режима есть JSON+MD пакет и reproducible seed/mode |
| 9 | `base_model_digest` + `atoms_digest` | `src/agim/eval/easyedit_payload.py`, `src/agim/eval/easyedit_run_metadata.py`, `src/agim/model/patch_artifact.py` | Partially Done | Все официальные бенчмарки проверяются на наличие digest-полей |
| 10 | Durable PatchArtifact | `src/agim/model/patch_artifact.py`, `tests/test_patch_artifact.py` | Partially Done | Нужен end-to-end сериализуемый lifecycle packet с версионными схемами и signatures |
| 11 | Save/Reload/Apply/Rollback test loop | `tests/test_patch_service.py`, `tests/test_patch_artifact.py` | Partially Done | Нужен отдельный интеграционный контур save/reload/apply/rollback |
| 12 | Metric-family selector для `failures_only` | `src/agim/eval/easyedit_run_metadata.py`, `src/agim/eval/easyedit_payload.py`, `src/agim/eval/easyedit_failures.py`, `src/agim/eval/easyedit_cli.py` | Done | `--failure-families` и persist-цепочка через `failure_summary`/`write_failures_only` |
| 13 | Покрытие random-200 (seed) | `results/easyedit_official/current/random_200_*.json` и report | Done | Имеется набор 200-facts с fail-only/dry-run метаданными |
| 14 | Покрытие random-1000 (seed) | `results/easyedit_official/current/random_1000_seed_42.json` и report | Done | Имеется 1000-facts official-compatible baseline |
| 15 | `target_token_mode` matrix | `results/easyedit_official/ablations/token_mode_matrix_report_2026-05-18.md` | Done | standalone/contextual/both сравнение выполнено |
| 16 | Component ablations | `results/easyedit_official/ablations/component_ablation_report_2026-05-18.md` | Done | Покрыты lm_head/embed/dual/no-eos/no-anti вариации |
| 17 | Exact-additive ablation | `results/easyedit_official/ablations/exact_additive_report_2026-05-18.md` | Done | Есть базовая baseline для сравнения |
| 18 | Детерминированный NT sampling | `src/agim/eval/easyedit_metrics.py`, `src/agim/eval/easyedit_eval_loop.py` | Partially Done | `nt-sample-size` и seed отражены в payload |
| 19 | Метрики growth + reuse | `results/easyedit_official/current/easyedit_agim_status_2026-05-18.md` | Partially Done | Требуется добавить cumulative norm/reuse в общий audit schema |
| 20 | Убрать EOS из persistent default | `src/agim/eval/easyedit_presets.py` | Done | Продакшн-профиль не опирается на глобальный EOS |
| 21 | Убрать global anti-repetition | `src/agim/eval/easyedit_cli.py`, `src/agim/model/wal_dual_editor.py` | Done | Глобальный anti-boost отключен по умолчанию (`clamp_anti=0`, `clamp_anti_scope=none`) |
| 22 | Fix both-mode primary selection | `src/agim/model/wal_dual_editor.py`, `src/agim/eval/easyedit_eval_loop.py` | Done (design) | Нужен dedicated both-mode official run |
| 23 | Constrained `K_pos/K_neg` solve | `src/agim/model/wal_dual_editor.py`, `src/agim/eval/easyedit_cli.py` | Done (synthetic internal/public proof) | Реализована constrained-модель и CLI-передача параметров; synthetic proof завершен |
| 24 | Relation-protected банки | `results/easyedit_official/sequential/relation_protected_bank_report_2026-05-18.md` | Done (synthetic internal/public proof) | worst-relation приоритет и protected-priority sampling зафиксированы в synthetic public proof |
| 25 | ENCORE-стиль budgets/early stop | `results/easyedit_official/ablations/norm_budget_no_commit_probe_2026-05-18.md` | Done (synthetic internal/public proof) | Runtime/patch budget + no-commit путь закрыт в synthetic public proof |
| 26 | Patch conflict detector | `src/agim/model` | Done (synthetic internal/public proof) | Проверки пересечений subject/target/EOS/control rows и risk flags закрыты в synthetic public proof |
| 27 | Runtime sparse overlay | `src/agim/model` (overlay API) | Partially Done | Перевод apply-flow в overlay-by-default |
| 28 | Namespace/session isolation | `src/agim/model`, CLI `--state-namespace` | Partially Done | Мульти-tenant isolation состояния редактора |
| 29 | Side-slot baseline | `results/easyedit_official/sequential/side_slot_random_50_report_2026-05-18.md` | Done | Stable 50 seed-coverage и clear baseline получены |
| 30 | Relation sharding in slots | `src/agim/model`, `src/agim/eval/easyedit_side_slot_loop.py` | Done (synthetic internal/public proof) | Router по relation + slot allocator |
| 31 | Side-slot retention 10/50/100 | `results/easyedit_official/sequential/` | Done | Есть n=10/50/100 для seed 42/43/44 |
| 32 | WAL-Rome n=50 baseline | `src/agim/model/wal_rome_editor.py`, `results/easyedit_official/ablations/wal_rome_smoke_5_report_2026-05-18.md` | Partially Done | Нужен n=50 quality baseline, а не только smoke |
| 33 | WALMemit consolidation и n=50 | `src/agim/model/wal_memit_batch_editor.py`, `src/agim/model/wal_memit_editor.py` | Done (synthetic internal/public proof) | Synthetic public proof по n=50 пути сформирован |
| 34 | Backend matrix (4 backends) | `src/agim/eval/easyedit_backend_matrix.py`, matrix артефакты | Done | Полная матрица `backend_matrix_random_50_seed42.*` и report подтверждены |
| 35 | RippleEdits цепочка | `src/agim/eval/ripple_diagnostic.py`, `results/external_benchmark_runs/ripple_wal_memit_n50_seed42.json` | Done | Артефакт с outputs по внешней цепочке есть |
| 36 | MQuAKE цепочка | `src/agim/eval/mquake_output_runner.py`, adapters + score | Done | `results/external_benchmark_runs/mquake_wal_memit_n50_seed42_outputs.json`, `_scored.json` |
| 37 | Raw-text AKEW-style цепочка | `src/agim/eval/raw_text_edit_pipeline.py`, `src/agim/eval/raw_text_scoring.py` | Done | `results/external_benchmark_runs/raw_text_wal_memit_n50_seed42.json` |
| 38 | Product-facing benchmark | `src/agim/eval/product_diagnostic.py`, `results/other_benchmarks/` | Done | `results/external_benchmark_runs/product_scedit_wal_memit_n50_seed42.json` |
| 39 | PatchService public lifecycle | `src/agim/model/patch_service.py`, `tests/test_patch_service.py` | Done (synthetic internal/public proof) | Базовый lifecycle реализован; доказательный payload (`propose/simulate/run_canaries/approve/apply/rollback/inspect/diff`) сформирован |
| 40 | Governance + adapter package | `src/agim/model/patch_governance.py`, `src/agim/integrations/easyedit_agimwal.py`, `docs/EASYEDIT_ADAPTER.md` | Done (synthetic internal/public proof) | approvals/signatures + immutable audit packet и публичный контракт в synthetic public proof |

## Hard gates (обязательные для completion)

1. `--edit-backend wal_memit` official n=50 baseline  
   - `results/easyedit_official/current/random_50_seed_42_wal_memit.json`  
   - `results/easyedit_official/current/random_50_seed_42_wal_memit.failures.json`  
   - Команда: см. `docs/PATH_B_MAX_HARDGATE_QUEUE.md` / `docs/PATH_B_MAX_GATE_COMMANDS.md`

2. Full backend matrix `dual_row`, `wal_rome`, `wal_memit`, `side_slot`  
   - `results/easyedit_official/ablations/backend_matrix_random_50_seed42.json`  
   - `...backend_matrix_random_50_seed42.dual_row.json`  
   - `...backend_matrix_random_50_seed42.wal_rome.json`  
   - `...backend_matrix_random_50_seed42.wal_memit.json`  
   - `...backend_matrix_random_50_seed42.side_slot.json`  
   - Отчёт: `...backend_matrix_random_50_report_2026-05-18.md`

3. External consequence chain минимум из 4 каналов  
   - `results/external_benchmark_runs/ripple_wal_memit_n50_seed42.json`  
   - `results/external_benchmark_runs/mquake_wal_memit_n50_seed42_outputs.json`  
   - `results/external_benchmark_runs/mquake_wal_memit_n50_seed42_scored.json`  
   - `results/external_benchmark_runs/raw_text_wal_memit_n50_seed42.json`  
   - `results/external_benchmark_runs/product_scedit_wal_memit_n50_seed42.json`

4. Side-slot retention hardening  
   - `results/easyedit_official/sequential/side_slot_random_10_seed_{42,43,44}_seq.json`  
   - `results/easyedit_official/sequential/side_slot_random_50_seed_{42,43,44}_seq.json`  
   - `results/easyedit_official/sequential/side_slot_random_100_seed_{42,43,44}_seq.json`  
  - `save-failures-only` артефакты для каждой точки

5. PatchService / governance proof packet  
   - `docs/CLAIMS_AND_EVIDENCE.md`, `docs/PATH_B_COMPLETION_AUDIT.md`  
   - контракт из `src/agim/model/patch_service.py`, `src/agim/model/patch_governance.py`  
   - concrete packet с `propose/simulate/run_canaries/approve/apply/rollback/inspect/diff`

## Текущее состояние completion (proof-first)

### Что подтверждено

- Источники истины и claims split: `CURRENT_STATUS.md`, `BENCHMARK.md`, `docs/CLAIMS_AND_EVIDENCE.md`, `docs/PATH_B_COMPLETION_AUDIT.md`.
- Документы-гейтс и команды подготовлены: `docs/PATH_B_MAX_MAX_AUDIT.md`, `docs/PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md`, `docs/PATH_B_MAX_PROMPT_ARTIFACT_AUDIT.md`, `docs/PATH_B_MAX_GATE_COMMANDS.md`.
  - Исполнительные скрипты и маршрут есть: `scripts/run_path_b_max_bootstrap.sh`, `docs/PATH_B_MAX_HARDGATE_QUEUE.md`.
- Метод-профили и metadata baseline уже внедрены (`method_profile_id`, `artifact_schema_version`).

### Что не закрыто для objective

- Gate 1/2/3/4 закрыты по артефактам; проверка ниже актуальна для `Gate 5`.
- Gate 5: proof packet по цепочке lifecycle (`results/easyedit_official/governance/path_b_max_gate5_proof.json`) и public release/public bundle/public receipt/public transport manifest уже сформированы, но нужен production external immutable-provider claim/adapter-level демонстрационный трек.

### Требования с partial и их status impact

- req. 9, 10, 11, 18, 19, 27, 28, 32 остаются не закрыты/частично закрыты и требуют завершения по доказательной цепочке.
- req. 23/24/25/26/30/33/39/40 закрыты на synthetic internal/public proof; для финального completion остаётся production external immutable contract Gate 5.

## Immediate max-priority execution map (from objective)

1. governance proof packet (`Gate 5`)  
   - Run: `scripts/run_path_b_max_bootstrap.sh 5`  
   - Evidence: public lifecycle release packet + immutable transport config (`public` endpoint + immutable provider)  
- Impact: synthetic proof уже закрывает reqs. 23, 24, 25, 26, 30, 33, 39, 40; final proof блокируется только production external Gate 5.

### Непосредственный нефундаментальный backlog

- req. 23/24/25/26/30/33/39/40: уже закрыты synthetic internal/public proof

## Очередь исполнения (сразу внедряемая)

1. Сначала закрыть Gate 5 в production external режиме (`AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL=1`, provider + `AGIM_GATE5_TRANSPORT_IMMUTABILITY_MODE=object_lock` + публичный base URL, API smoke через `AGIM_GATE5_PUBLIC_API_SMOKE=1`).
2. Закрыть req. 23, 24, 25, 26, 30, 33, 39, 40 на уровне доказательных public artifacts (fixture + release payload + signatures).
3. После каждого production-шага обновлять `docs/PATH_B_MAX_EXECUTION_CHECKLIST.md`, `docs/PATH_B_COMPLETION_AUDIT.md`, `docs/PATH_B_MAX_COMPLETION_MATRIX.md`, `docs/PATH_B_MAX_PROMPT_ARTIFACT_AUDIT_MATRIX.md`, `docs/PATH_B_MAX_STATUS_BOARD.md`.
4. На выходе — подготовить единый “proof packet”:
   - hard evidence matrix
   - gate summary table
   - обновлённые claims boundary (без unsafe формулировок)

## Команды-обёртки

- Полный hard-gate run: `scripts/run_path_b_max_bootstrap.sh`  
- Поочередный запуск: `scripts/run_path_b_max_bootstrap.sh 1 2 4 3 5`  
- Только подготовительный блок: `scripts/run_path_b_max_bootstrap.sh 1 2`

### Дополнительный аудит

- [PATH_B_MAX_MAX_AUDIT.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_MAX_AUDIT.md) — полный prompt→command→artifact checklist для всех 40 требований и 5 hard gates.
