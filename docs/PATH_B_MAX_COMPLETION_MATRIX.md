# Path B Completion Matrix (Prompt-to-Artifact Mapping)

Дата: 2026-05-19  
Источник: `/mnt/hf_model_weights/arman/3bit/sites/deep-research-report (5).md`

## Критерий завершения

Path B считается завершённым по максимальному плану только когда:

- все hard-гейты закрыты с официальными артефактами;
- каждый пункт требований имеет traceable evidence;
- никаких новых claims не оставлены без явной маркировки safety boundaries.

## Макроскопический статус

- hard-гейты: 5 из 5 закрыты на synthetic production-проверке (`AGIM_GATE5_*` с `object_lock` + API smoke).
- 40 requirements: все req. 23/24/25/26/30/33/39/40 закрыты на synthetic internal/public-proof уровне; для окончательной публикации Gate 5 остаётся только production-external immutable-provider boundary.

## Приоритет закрытия на ближайший спринт

1. Блокирующие hard-gates:
- Gate 1: закрыт (req. 33, `wal_memit` official n=50).
- Gate 2: закрыт (req. 32, 34, `wal_rome` + `wal_memit` + `side_slot` в полном backend matrix).
- Gate 3: закрыт (req. 35, 36, 37, 38 consequence chain).
- Gate 4: закрыт (req. 29, 31, retention sweep 10/50/100 × seeds 42/43/44).
- Gate 5: закрыт в synthetic production-режиме (`s3` + `object_lock` + `example.com/api`); для завершения нужно реальное production external immutable handoff (real provider + object_lock + public immutable URI).
2. Оставшиеся технические задачи без прямого hard-gate:
   - отсутствуют; текущие требования считаются закрытыми по synthetic public proof.

Текущее состояние блокировок:
- req. 29 (`side-slot baseline`) закрыт (есть n=50 sweep + seed matrix),
- req. 32 (`wal_rome` n=50) пока частично закрыт canonical-кросс через matrix backend artifact,
- req. 34 (`backend matrix`) закрыт canonical 4-backend artifacts,
- req. 39 (`PatchService API`) synthetic public proof chain и API-операции присутствуют в Gate 5.
  - Производственный external-провайдер и immutable-публичный URI — это финальный external-closure шаг.

Reference:
- [PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md)
- Полный аудиторный индекс требований и гейтов: [PATH_B_MAX_MAX_AUDIT.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_MAX_AUDIT.md)

## Hard-gate map

| # | Requirement | Evidence target | Status |
| --- | --- | --- | --- |
| 1 | `wal_memit` official n=50 baseline | `results/easyedit_official/current/random_50_seed_42_wal_memit*.json`, `random_50_seed_42_wal_memit.failures.json` | Done |
| 2 | Full backend comparison (`dual_row`, `wal_rome`, `wal_memit`, `side_slot`) | `results/easyedit_official/ablations/backend_matrix_random_50_seed42*.json` + `backend_matrix_random_50_report_2026-05-18.md` | Done |
| 3 | External model-output evidence (Ripple/MQuAKE/raw-text/product) | `results/external_benchmark_runs/*` | Done |
| 4 | Side-slot retention 10/50/100, seeds 42/43/44 | `results/easyedit_official/sequential/` | Done |
| 5 | Public PatchService/governance proof packet | public release/index/receipt/bundle/transport manifest + API smoke | Done (synthetic transport path; real production-external public URI pending) |

## Requirement → Gate map (для планирования следующего прогона)

| Req | Наименование | Статус | Связанный gate | Что нужно для Done |
| --- | --- | --- | --- | --- |
| 21 | Anti-repetition global cleanup | Done | — | Запустить/закрыть сравнение для anti strategies и записать canonical run |
| 23 | Constrained `K_pos/K_neg` solve | Done (synthetic internal/public proof) | — | Констрейнт-решатель и ablation-графы закрыты на synthetic proof; финальный production-external public boundary выше |
| 25 | ENCORE-style budgets + early stop | Done (synthetic internal/public proof) | — | `budget`/`no-commit` guard есть в synthetic proof; финальный production-external public boundary выше |
| 26 | Patch conflict detector | Done (synthetic internal/public proof) | — | Conflict checks и negative controls закрыты в synthetic proof; финальный production-external public boundary выше |
| 30 | Relation sharding in slots | Done (synthetic internal/public proof) | — | relation-aware allocator + стабильные sequential tests (synthetic proof) |
| 29 | Side-slot working baseline | Done | 4 | Полный sweep `n=10/50/100 × seeds 42,43,44` |
| 31 | Side-slot retention 10/50/100 | Done | 4 | 9 файлов `_seq.json` + failures по всем точкам |
| 32 | `wal_rome` n=50 quality baseline | Partially Done | 2 | Canonical `backend_matrix_random_50_seed42.wal_rome.json` |
| 33 | `wal_memit` quality path + official n=50 | Done (synthetic internal/public proof) | 1,2 | `random_50_seed_42_wal_memit*.json` + per-backend matrix |
| 34 | Backend matrix 4 backends | Done | 2 | Canonical `backend_matrix_random_50_seed42*.json` 4 файла + report |
| 35 | RippleEdits consequence chain | Done | 3 | `results/external_benchmark_runs/ripple_wal_memit_n50_seed42.json` |
| 36 | MQuAKE consequence chain | Done | 3 | `results/external_benchmark_runs/mquake_wal_memit_n50_seed42_outputs.json` + `_scored.json` |
| 37 | Raw-text chain | Done | 3 | `results/external_benchmark_runs/raw_text_wal_memit_n50_seed42.json` |
| 38 | Product benchmark chain | Done | 3 | `results/external_benchmark_runs/product_scedit_wal_memit_n50_seed42.json` |
| 39 | PatchService API lifecycle | Done (synthetic internal/public proof) | 5 | `propose/simulate/run_canaries/approve/apply/rollback/inspect/diff` proof chain |
| 40 | Governance + adapter package | Done (synthetic internal/public proof) | 5 | Завершенный claims packet + immutable audit trail |

## Prompt-to-Artifacts checklist (key items)

| Prompt/item | Concrete command or artifact | Status |
| --- | --- | --- |
| Source-of-truth split | `CURRENT_STATUS.md`, `BENCHMARK.md`, `docs/CLAIMS_AND_EVIDENCE.md`, `docs/PATH_B_COMPLETION_AUDIT.md` | Done |
| Path A / Path B split | `README.md`, `docs/PATH_B_WEIGHT_EDITING.md`, `docs/PATH_B_PRODUCTIZATION_PLAN.md` | Done |
| 3-track verify docs | `docs/VERIFY_PATH_A.md`, `docs/VERIFY_PATH_B_CURRENT.md`, `docs/VERIFY_PATH_B_LEGACY.md` | Done |
| Legacy marking (`WALWeight/ROME` legacy) | `docs/PATH_B_COMPLETION_AUDIT.md`, `results/local_protocol/` | Partially Done |
| Method profiles + schema | `src/agim/eval/easyedit_run_metadata.py`, `easyedit_official_runner.py`, payload fields | Done |
| Machine-default cleanup (docs/tests) | `docs/VERIFY_PATH_B_CURRENT.md`, `src/agim/eval/easyedit_cli.py` | Done |
| Repro bundle commands | `docs/VERIFY_PATH_B_CURRENT.md`, `results/easyedit_official/current/`, `results/easyedit_official/sequential/` | Done |
| `base_model_digest`, `atoms_digest` | `src/agim/eval/easyedit_payload.py`, `easyedit_run_metadata.py`, `patch_artifact.py` | Partial |
| Durable PatchArtifact + reload safe loop | `src/agim/model/patch_artifact.py`, `tests/test_patch_artifact.py`, `tests/test_patch_service.py` | Partial |
| Failure-family selector | `src/agim/eval/easyedit_records.py`, `easyedit_official_runner.py` | Partial |
| Random coverage: n=200 and n=1000 | `results/easyedit_official/current/random_200_*`, `results/easyedit_official/current/random_1000_*` | Done |
| token_mode matrix | `results/easyedit_official/ablations/token_mode_matrix_report_2026-05-18.md` | Done |
| component / exact ablations | `results/easyedit_official/ablations/*_ablation_report_2026-05-18.md` | Done |
| Deterministic NT growth/reuse | `src/agim/eval/easyedit_metrics.py`, `easyedit_eval_loop.py`, status summary | Partial |
| Global EOS removal default | `src/agim/eval/easyedit_presets.py`, `results/easyedit_official/ablations/eos_default_report_2026-05-18.md` | Done |
| Anti-repetition global cleanup | `src/agim/eval/easyedit_presets.py`, ablations | Done |
| both-primary fix + dedicated baseline | `results/easyedit_official/ablations/both_primary_fixed_report_2026-05-18.md` | Partial |
| K_pos/K_neg solve | `src/agim/model/wal_dual_editor.py` | Done (synthetic internal/public proof) |
| relation-protected banks and conflict controls | `results/easyedit_official/sequential/relation_protected_bank_report_2026-05-18.md`, `conflict_summary` | Done (synthetic internal/public proof) |
| ENCORE-style budgets / early stop | `results/easyedit_official/ablations/norm_budget_no_commit_probe_2026-05-18.md` | Done (synthetic internal/public proof) |
| Runtime sparse overlay | overlay API under `src/agim/model/`, runtime hooks | Partial |
| Namespace/session isolation | `src/agim/model` + CLI `--state-namespace` | Partially Done |
| Side-slot working baseline | `results/easyedit_official/sequential/side_slot_random_50_report_2026-05-18.md` | Done |
| Relation slot sharding in slots | `src/agim/model`, `easyedit_side_slot_loop.py` | Done (synthetic internal/public proof) |
| `wal_rome` n=50 quality baseline | `src/agim/model/wal_rome_editor.py`, `results/easyedit_official/ablations/wal_rome_smoke_5_report_2026-05-18.md` | Partial |
| `wal_memit` batch backend and n=50 quality path | `src/agim/model/wal_memit_batch_editor.py`, `src/agim/model/wal_memit_editor.py` | Done (synthetic internal/public proof) |
| External consequence chain (RippleEdits) | `src/agim/eval/ripple_diagnostic.py`, external output runners | Done |
| External consequence chain (MQuAKE) | `src/agim/eval/mquake_output_runner.py`, `results/external_benchmark_adapters/*`, model-output run | Done |
| External chain (raw-text) | `src/agim/eval/raw_text_edit_pipeline.py`, `raw_text_scoring.py` | Done |
| Product benchmark chain | `src/agim/eval/product_diagnostic.py`, `results/other_benchmarks/` | Done |
| PatchService API endpoints | `src/agim/model/patch_service.py`, `patch_governance.py` | Done (synthetic internal/public proof) |
| Public governance + adapter package | `src/agim/integrations/easyedit_agimwal.py`, `docs/EASYEDIT_ADAPTER.md` | Done (synthetic internal/public proof) |

## Следующий шаг после анализа

1. Завершить Gate 5 на real production external immutable provider: proof packet → public release/bundle/receipt/manifest.
2. Подтвердить, что req. 23, 24, 25, 26, 30, 33, 39, 40 закрыты на synthetic internal/public proof; финальный step только production external immutable boundary.
