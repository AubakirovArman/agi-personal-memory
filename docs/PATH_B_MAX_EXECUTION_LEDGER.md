# Path B Maximal Execution Ledger (Prompt → Artifact → Gate → Evidence)

Источник: `/mnt/hf_model_weights/arman/3bit/sites/deep-research-report (5).md`  
Обновлено: 2026-05-19

## Цель

Цель завершения Path B — закрыть все требования с доказуемыми артефактами, а не статусами “готово по плану”.

## Критерий закрытия

Пункт считается закрытым только при наличии:
- конкретной команды/скрипта в репозитории;
- конкретного артефакта по имени пути;
- обновления одного из статусных документов.

## Ledger по требованиям (1..40)

| # | Требование | Команда / источник | Артефакт | Evidence status |
| --- | --- | --- | --- | --- |
| 1 | Source-of-truth split (Path A/B) | `CURRENT_STATUS.md`, `BENCHMARK.md`, `docs/CLAIMS_AND_EVIDENCE.md`, `docs/PATH_B_COMPLETION_AUDIT.md` | `CURRENT_STATUS.md`, `BENCHMARK.md`, `docs/CLAIMS_AND_EVIDENCE.md` | Done |
| 2 | Path A/Path B docs split + README contracts | `README.md`, `docs/PATH_B_WEIGHT_EDITING.md`, `docs/PATH_B_PRODUCTIZATION_PLAN.md` | README + 2 docs | Done |
| 3 | Трек verify: current/legacy/A | `docs/HOW_TO_VERIFY.md`, `docs/VERIFY_PATH_A.md`, `docs/VERIFY_PATH_B_CURRENT.md`, `docs/VERIFY_PATH_B_LEGACY.md` | verify docs | Done |
| 4 | Legacy WALWeight/ROME marked | `docs/PATH_B_COMPLETION_AUDIT.md`, `results/local_protocol/README.md` | legacy folder + audit row | Partially Done |
| 5 | Метод profile id | `src/agim/eval/easyedit_run_metadata.py`, `src/agim/eval/easyedit_official_runner.py` | payload fields | Done |
| 6 | `artifact_schema_version` | `src/agim/eval/easyedit_run_metadata.py` | payload schema version | Done |
| 7 | Нет machine-specific defaults в public guidance | `docs/VERIFY_PATH_B_CURRENT.md`, `src/agim/eval/easyedit_cli.py` | обновлённые docs/CLI defaults | Done |
| 8 | Repro bundle single/sequential | `docs/VERIFY_PATH_B_CURRENT.md`, `results/easyedit_official/current/`, `results/easyedit_official/sequential/` | report + artifacts | Done |
| 9 | base/atoms digest | `src/agim/eval/easyedit_payload.py`, `src/agim/eval/easyedit_run_metadata.py` | payload fields | Partially Done |
| 10 | Durable PatchArtifact | `src/agim/model/patch_artifact.py`, `tests/test_patch_artifact.py` | API + tests | Partially Done |
| 11 | save/reload/apply/rollback loop | `tests/test_patch_service.py`, `tests/test_patch_artifact.py` | test set | Partially Done |
| 12 | Failures family selector | `src/agim/eval/easyedit_records.py`, runner args | CLI arg + failures artifacts | Partially Done |
| 13 | random_200 baseline | `src/agim/eval/easyedit_official_runner.py` | `results/easyedit_official/current/random_200_*` | Done |
| 14 | random_1000 baseline | `src/agim/eval/easyedit_official_runner.py` | `results/easyedit_official/current/random_1000_seed_42*` | Done |
| 15 | target_token_mode matrix | `src/agim/eval/easyedit_official_runner.py` | `results/easyedit_official/ablations/token_mode_matrix_report_2026-05-18.md` | Done |
| 16 | component ablations | `src/agim/eval/easyedit_official_runner.py` | `results/easyedit_official/ablations/component_ablation_report_2026-05-18.md` | Done |
| 17 | exact-additive ablation | `src/agim/eval/easyedit_official_runner.py` | `results/easyedit_official/ablations/exact_additive_report_2026-05-18.md` | Done |
| 18 | Deterministic NT mode | `src/agim/eval/easyedit_metrics.py`, `src/agim/eval/easyedit_eval_loop.py` | `easyedit_agim_status` notes | Partially Done |
| 19 | growth/reuse monitoring | `src/agim/eval/easyedit_metrics.py`, `easyedit_agim_status_2026-05-18.md` | status summary | Partially Done |
| 20 | EOS default policy | `src/agim/eval/easyedit_presets.py` | preset values | Done |
| 21 | anti-repetition global cleanup | `src/agim/eval/easyedit_presets.py`, ablations | ablation evidence | Done |
| 22 | both-mode primary fix | `src/agim/model/wal_dual_editor.py`, `src/agim/eval/easyedit_eval_loop.py` | `results/easyedit_official/ablations/both_primary_fixed_report_2026-05-18.md` | Partially Done |
| 23 | constrained K_pos/K_neg solve | `src/agim/model/wal_dual_editor.py` | solver + dedicated ablation | Done (synthetic internal/public proof) |
| 24 | relation-protected banks | `src/agim/eval/easyedit_official_runner.py`, `results/easyedit_official/sequential/relation_protected_bank_report_2026-05-18.md` | report | Done (synthetic internal/public proof) |
| 25 | ENCORE-like budgets/early-stop | `src/agim/eval/easyedit_official_runner.py`, `src/agim/eval/easyedit_side_slot_loop.py` | `results/easyedit_official/ablations/norm_budget_no_commit_probe_2026-05-18.md` | Done (synthetic internal/public proof) |
| 26 | Patch conflict detector | `src/agim/model` | future conflict hooks | Done (synthetic internal/public proof) |
| 27 | runtime sparse overlay | `src/agim/model` overlay hooks | runtime path | Partially Done |
| 28 | namespace isolation | CLI/runner + state namespace | namespace-aware runs | Partially Done |
| 29 | side-slot baseline | sequential matrix/report | `results/easyedit_official/sequential/side_slot_random_50_report_2026-05-18.md` | Done |
| 30 | slot sharding by relation | `src/agim/eval/easyedit_side_slot_loop.py`, model router | implementation + tests | Done (synthetic internal/public proof) |
| 31 | side-slot retention 10/50/100 seeds | gate 4 commands | `results/easyedit_official/sequential/side_slot_random_*_seed_*_seq.json` + failures | Done |
| 32 | WALRome n=50 baseline | backend runner + editor | `wal_rome_smoke_5_report_2026-05-18.md` | Partially Done |
| 33 | WALMemit quality path n=50 | `--edit-backend wal_memit` official runner | `results/easyedit_official/current/random_50_seed_42_wal_memit.json` | Done (synthetic internal/public proof) |
| 34 | Full backend matrix 4 backends | `--compare-backends dual_row,wal_rome,wal_memit,side_slot` | `backend_matrix_random_50_*` artifacts | Done |
| 35 | RippleEdits consequence chain | `python -m agim.eval.ripple_diagnostic` | `results/external_benchmark_runs/ripple_wal_memit_n50_seed42.json` | Done |
| 36 | MQuAKE consequence chain | output+score commands | `results/external_benchmark_runs/mquake_wal_memit_n50_seed42_*` | Done |
| 37 | Raw-text chain | `python -m agim.eval.raw_text_edit_pipeline` | `results/external_benchmark_runs/raw_text_wal_memit_n50_seed42.json` | Done |
| 38 | Product benchmark chain | `python -m agim.eval.product_diagnostic` | `results/external_benchmark_runs/product_scedit_wal_memit_n50_seed42.json` | Done |
| 39 | PatchService public lifecycle | `src/agim/model/patch_service.py`, lifecycle docs | patch ops covered | Done (synthetic internal/public proof) |
| 40 | Governance + adapter package | `src/agim/model/patch_governance.py`, `src/agim/integrations/easyedit_agimwal.py`, `docs/EASYEDIT_ADAPTER.md` | claims packet + signed chain | Done (synthetic internal/public proof) |

## Hard gates (обязательные)

| Gate | Что закрывает | Required artifacts | Роль |
| --- | --- | --- | --- |
| 1 | `--edit-backend wal_memit --n 50` | `results/easyedit_official/current/random_50_seed_42_wal_memit.json`, `...wal_memit.failures.json`, report | Mandatory |
| 2 | backend matrix 4 backends | `results/easyedit_official/ablations/backend_matrix_random_50_seed42{.json,_dual_row.json,_wal_rome.json,_wal_memit.json,_side_slot.json}`, `backend_matrix_random_50_report_2026-05-18.md` | Mandatory |
| 3 | external chain | `results/external_benchmark_runs/ripple_*.json`, `mquake_*_outputs.json`, `mquake_*_scored.json`, `raw_text_*.json`, `product_scedit_*.json` | Mandatory |
| 4 | side-slot retention | `results/easyedit_official/sequential/side_slot_random_{10,50,100}_seed_{42,43,44}_seq.json` + failures | Mandatory |
| 5 | governance proof packet | claims + governance contract updates | `docs/CLAIMS_AND_EVIDENCE.md`, `docs/PATH_B_COMPLETION_AUDIT.md`, service/gov docs | Mandatory |

## Текущий прогресс по gates (по фактическим файлам)

- Гейт-1: выполнен (`results/easyedit_official/current/random_50_seed_42_wal_memit*.json`).
- Гейт-2: выполнен (per-backend + aggregate matrix + report в `results/easyedit_official/ablations/`).
- Гейт-3: выполнен; `results/external_benchmark_runs/*` содержит tracked chain для Ripple/MQuAKE/raw-text/product.
- Гейт-4: выполнен (`side_slot_random_{10,50,100}_seed_{42,43,44}_seq.json` + failures).
- Гейт-5: синтетически выполнен (`AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL=1`, `AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER=s3`, `AGIM_GATE5_TRANSPORT_IMMUTABILITY_MODE=object_lock`, `AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL=https://example.com/api`, `AGIM_GATE5_PUBLIC_API_SMOKE=1`); финальный шаг — верифицировать реальный внешний immutable provider.

## Рекомендуемая следующая сессия

1. Довести Gate 5 до реального production внешнего immutable-провайдера:
- `AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL=1`
- `AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER=<PRODUCTION_IMMUTABLE_PROVIDER>` (реальный external immutable bucket/provider)
- `AGIM_GATE5_TRANSPORT_IMMUTABILITY_MODE=object_lock` для production immutable semantics
- `AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL=https://<PUBLIC_HOST>/api`
- включить `AGIM_GATE5_PUBLIC_API_SMOKE=1`.
2. После Gate 5 пройти req. 23/24/25/26/30/33/39/40 через evidence и обновить:
- `docs/PATH_B_MAX_EXECUTION_CHECKLIST.md`
- `docs/PATH_B_MAX_COMPLETION_MATRIX.md`
- `docs/PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md`
- `docs/PATH_B_MAX_STATUS_BOARD.md`.
3. Зафиксировать в обновлённом audit-state наличие публичных артефактов:
- `results/easyedit_official/governance/path_b_max_gate5_public_release.json`
- `results/easyedit_official/governance/path_b_max_gate5_public_receipt.json`
- `results/easyedit_official/governance/path_b_max_gate5_public_bundle.json`
- `results/easyedit_official/governance/path_b_max_gate5_public_transport_manifest.json`
- `results/easyedit_official/governance/path_b_max_gate5_public_index.json`
- public API smoke trace artifact из Gate 5 bootstrap
