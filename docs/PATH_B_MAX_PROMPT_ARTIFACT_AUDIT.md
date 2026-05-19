# Path B Prompt-to-Artifact Completion Audit

Источник: `/mnt/hf_model_weights/arman/3bit/sites/deep-research-report (5).md`

Справочный execution-backbone:
[PATH_B_MAX_EXECUTION_BLUEPRINT.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_EXECUTION_BLUEPRINT.md)
- [PATH_B_MAX_COMPLETION_MATRIX.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_COMPLETION_MATRIX.md)
- [PATH_B_MAX_EXECUTION_CHECKLIST.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_EXECUTION_CHECKLIST.md)
- [PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md)
- [PATH_B_MAX_HARDGATE_QUEUE.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_HARDGATE_QUEUE.md)

Цель:
- Для каждого явного требования из плана построить трассировку: пункт →
  команды/гейты → артефакт → текущий статус.
- Считать завершённым только при наличии явных доказательств в репозитории.
- `Done` разрешён только для evidence-backed claims.

## Deliverables Mapping

| # | Требование | Артефакт / контроль | Статус |
| --- | --- | --- | --- |
| 1 | Развести source-of-truth в один активный слой | `CURRENT_STATUS.md`, `BENCHMARK.md`, `docs/CLAIMS_AND_EVIDENCE.md`, `docs/PATH_B_COMPLETION_AUDIT.md` | Done |
| 2 | Удерживать разделение Path A и Path B в документах | `README.md`, `docs/PATH_B_WEIGHT_EDITING.md`, `docs/PATH_B_PRODUCTIZATION_PLAN.md` | Done |
| 3 | Развести current и legacy verify-подходы | `docs/HOW_TO_VERIFY.md`, `docs/VERIFY_PATH_B_CURRENT.md`, `docs/VERIFY_PATH_B_LEGACY.md`, `docs/VERIFY_PATH_A.md` | Done |
| 4 | Явно обозначить legacy-контур WALWeight/ROME | `docs/PATH_B_COMPLETION_AUDIT.md`, `results/local_protocol/README.md`, `results/local_protocol/` | Partially Done |
| 5 | Добавить method_profile_id и profiles | `src/agim/eval/easyedit_run_metadata.py`, `src/agim/eval/easyedit_official_runner.py` | Done |
| 6 | Добавить artifact_schema_version | `src/agim/eval/easyedit_run_metadata.py` | Done |
| 7 | Убрать machine-specific defaults из публичных инструкций | `docs/VERIFY_PATH_B_CURRENT.md`, `src/agim/eval/easyedit_cli.py`, `README.md` | Done |
| 8 | Обеспечить reproducibility bundle для baseline/sequential | `docs/VERIFY_PATH_B_CURRENT.md`, `results/easyedit_official/current/`, `results/easyedit_official/sequential/` | Done |
| 9 | Добавить base_model_digest + atoms_digest | `src/agim/eval/easyedit_payload.py`, `src/agim/eval/easyedit_run_metadata.py`, `src/agim/model/patch_artifact.py` | Partially Done |
| 10 | Durable PatchArtifact + сериализация | `src/agim/model/patch_artifact.py`, `tests/test_patch_artifact.py` | Partially Done |
| 11 | Тест save/reload/re-apply/rollback | `tests/test_patch_service.py`, `tests/test_patch_artifact.py` | Partially Done |
| 12 | Раздельный failures_only по метрикам | `src/agim/eval/easyedit_records.py`, `src/agim/eval/easyedit_official_runner.py` | Done |
| 13 | random-200 и random-1000 seed coverage | `results/easyedit_official/current/random_200_report_2026-05-18.md`, `results/easyedit_official/current/random_1000_report_2026-05-18.md` | Done |
| 14 | Token mode matrix (`standalone/contextual/both`) | `results/easyedit_official/ablations/token_mode_matrix_report_2026-05-18.md` | Done |
| 15 | Component ablations | `results/easyedit_official/ablations/component_ablation_report_2026-05-18.md` | Done |
| 16 | Exact-additive ablation | `results/easyedit_official/ablations/exact_additive_report_2026-05-18.md` | Done |
| 17 | Детерминированный NT mode | `src/agim/eval/easyedit_metrics.py`, `src/agim/eval/easyedit_eval_loop.py` | Partially Done |
| 18 | Метрики growth и reuse | `results/easyedit_official/current/easyedit_agim_status_2026-05-18.md` | Partially Done |
| 19 | EOS remove из persistent default | `results/easyedit_official/ablations/eos_default_report_2026-05-18.md` | Done |
| 20 | anti-repetition из глобальных rows | `results/easyedit_official/ablations/component_random_200_seed_42_dual_no_anti*.json` | Done |
| 21 | `both` primary sequence fix | `src/agim/model/wal_dual_editor.py`, `results/easyedit_official/ablations/both_primary_fixed_report_2026-05-18.md` | Done |
| 22 | Constrained K_pos/K_neg решение | `src/agim/model/wal_dual_editor.py` | Done (synthetic internal/public proof) |
| 23 | Relation-protected банки | `results/easyedit_official/sequential/relation_protected_bank_report_2026-05-18.md` | Done (synthetic internal/public proof) |
| 24 | ENCORE-like norm budgets и early stop | `results/easyedit_official/ablations/norm_budget_no_commit_probe_2026-05-18.md` | Done (synthetic internal/public proof) |
| 25 | Conflict detector | `src/agim/model/*` + `conflict_summary` | Done (synthetic internal/public proof) |
| 26 | Runtime sparse overlay | `src/agim/model/*` overlay API | Partially Done |
| 27 | Namespace/session isolation | `src/agim/model`, `--state-namespace` | Partially Done |
| 28 | Side-slot memory baseline | `results/easyedit_official/sequential/side_slot_random_50_report_2026-05-18.md` | Done |
| 29 | Relation sharding for slots | `src/agim/model`, `src/agim/eval/easyedit_side_slot_loop.py` | Done (synthetic internal/public proof) |
| 30 | side-slot retention 10/50/100 | `results/easyedit_official/sequential/` | Done |
| 31 | WALRome n=50 quality baseline | `src/agim/model/wal_rome_editor.py`, `results/easyedit_official/ablations/wal_rome_smoke_5_report_2026-05-18.md` | Partially Done |
| 32 | WALMemitBatchEditor baseline | `src/agim/model/wal_memit_batch_editor.py`, `tests/test_wal_memit_batch_editor.py` | Partially Done |
| 33 | Backend matrix comparison quality | `src/agim/eval/easyedit_backend_matrix.py`, `results/easyedit_official/ablations/backend_matrix_*` | Done |
| 34 | RippleEdits benchmark chain | `src/agim/eval/ripple_diagnostic.py`, `results/external_benchmark_*` | Done |
| 35 | MQuAKE benchmark chain | `src/agim/eval/mquake_output_runner.py`, legacy adapter `results/external_benchmark_adapters/mquake_cf_3k_v2_first50_adapter.json` | Done |
| 36 | AKEW-style raw-text chain | `src/agim/eval/raw_text_edit_pipeline.py`, `src/agim/eval/raw_text_scoring.py` | Done |
| 37 | Product-facing benchmark chain | `results/other_benchmarks/`, `src/agim/eval/product_diagnostic.py` | Done |
| 38 | PatchService public lifecycle | `src/agim/model/patch_service.py`, `tests/test_patch_service.py` | Done (synthetic internal/public proof) |
| 39 | Governance + tenant safety + signatures | `src/agim/model/patch_governance.py`, `tests/test_patch_governance.py` | Done (synthetic internal/public proof) |
| 40 | External adapter + proof package | `src/agim/integrations/easyedit_agimwal.py`, `docs/EASYEDIT_ADAPTER.md` | Done (synthetic internal/public proof) |

## Hard-gate execution contract (prompt → command → artifact)

Рекомендуемая команда:

`bash scripts/run_path_b_max_bootstrap.sh 1 2 4 3 5`

| Gate | Что закрывает | Обязательная команда | Required evidence |
| --- | --- | --- | --- |
| 1 | `wal_memit` официальный baseline на `n=50` | `python -m agim.eval.easyedit_official_runner ... --edit-backend wal_memit --n 50` | JSON+MD в `results/easyedit_official/current/` с `sample-policy`, `method_profile_id`, `base_model_digest`, `atoms_digest`, `failure_analysis`, `patch_delta` |
| 2 | Полный backend matrix (`dual_row`, `wal_rome`, `wal_memit`, `side_slot`) | `python -m agim.eval.easyedit_official_runner ... --compare-backends dual_row wal_rome wal_memit side_slot` | Один matrix run и per-backend JSON с одинаковым `sample-policy` + markdown с deltas по метрикам и families |
| 3 | External consequence chain (Ripple/MQuAKE/raw-text/product) | `ripple_diagnostic`, `mquake_output_runner`, `raw_text_edit_pipeline`, `product_diagnostic` | Для каждого канала: raw model output (`json/jsonl`), scored markdown и baseline comparison в `results/external_benchmark_runs/` |
| 4 | Side-slot retention 10/50/100 | `python -m agim.eval.easyedit_official_runner --sequential-edit --edit-backend side_slot` | Прогоны для `n=10/50/100` и `seed=42/43/44` + `save-failures-only` |
| 5 | PatchService/governance proof packet | `docs/EASYEDIT_ADAPTER.md`, `src/agim/model/patch_service.py`, `src/agim/model/patch_governance.py` | Один claims packet: `propose → simulate → run_canaries → approve → apply → rollback → inspect → diff` + immutable audit trail |

## Command source of truth

Все hard-gate и critical runs выполняются по шаблонам из [PATH_B_MAX_GATE_COMMANDS.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_GATE_COMMANDS.md) и порядку из [PATH_B_MAX_HARDGATE_QUEUE.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_HARDGATE_QUEUE.md). После каждого успешного прогона обновлять:

- [PATH_B_COMPLETION_AUDIT.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_COMPLETION_AUDIT.md)
- [PATH_B_MAX_EXECUTION_CHECKLIST.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_EXECUTION_CHECKLIST.md)
- [PATH_B_MAX_HARDGATE_QUEUE.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_HARDGATE_QUEUE.md)

## Hard blockers (must be закрыты для флага completion)

1. `AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL=1` с production immutable-provider, `AGIM_GATE5_TRANSPORT_IMMUTABILITY_MODE=object_lock` и публичным base URL.
2. Публичный Gate‑5 release packet с подтверждением public API smoke (`AGIM_GATE5_PUBLIC_API_SMOKE=1`).
3. req. 23/24/25/26/30/33/39/40 закрыты на synthetic internal/public proof; остаётся только production-external immutable boundary для Gate 5.

## Command-level references for remaining critical gaps

- Для item 1-2: `easyedit_official_runner.py --edit-backend wal_memit` на n=50; per-backend JSON/MD.
- Для item 3: `agim.eval.ripple_diagnostic`, `agim.eval.mquake_output_runner`, raw-text scorer and product benchmark runners.
- Для item 4: `easyedit_official_runner --sequential-edit` с `side_slot`.
- Для item 5: `patch_service` + `patch_governance` public contract + release-gate proof tables.

Concrete executable command set:
- [PATH_B_MAX_GATE_COMMANDS.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_GATE_COMMANDS.md)
