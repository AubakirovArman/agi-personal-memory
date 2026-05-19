# Path B Maximal Audit (Prompt → Command → Artifact → Status)

Цель:
- привести Path B к доказуемому состоянию через explicit prompt-to-artifact mapping;
- закрыть все hard gates из `PATH_B_MAX_HARDGATE_QUEUE.md`;
- не считать completed требования без конкретных артефактов и команд.

## Источник требований
- `sites/deep-research-report (5).md`
  - `PATH_B_MAX_MAX_PLAN.md`
  - `PATH_B_MAX_EXECUTION_LEDGER.md`
  - `PATH_B_MAX_EXECUTION_CHECKLIST.md`
  - `PATH_B_MAX_COMPLETION_MATRIX.md`
  - `PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md`
  - `PATH_B_MAX_AUDIT_CHECKLIST.md`

## Hard gates (must complete first)

| Gate | Prompt | Команда | Обязательный артефакт | Статус |
| --- | --- | --- | --- | --- |
| 1 | `wal_memit` official n=50 | `python -m agim.eval.easyedit_official_runner --edit-backend wal_memit --n 50 ...` | `results/easyedit_official/current/random_50_seed_42_wal_memit.json`, `.failures.json` | Done |
| 2 | Full matrix 4 backends | `python -m agim.eval.easyedit_official_runner --compare-backends dual_row,wal_rome,wal_memit,side_slot ...` | `results/easyedit_official/ablations/backend_matrix_random_50_seed42*.json`, `backend_matrix_random_50_report_2026-05-18.md` | Done |
| 3 | External chain | `ripple_diagnostic`, `mquake_output_runner`, `mquake_diagnostic`, `raw_text_edit_pipeline`, `product_diagnostic` | `results/external_benchmark_runs/*n50_seed42*` (`ripple`, `mquake_*_outputs`, `mquake_*_scored`, `raw_text`, `product_scedit`) | Done |
| 4 | Side-slot retention 10/50/100 | `easyedit_official_runner --sequential-edit --edit-backend side_slot` for each `n` and `seed 42/43/44` | `results/easyedit_official/sequential/side_slot_random_{10,50,100}_seed_{42,43,44}_seq.json` + `.failures.json` | Done |
| 5 | Governance proof | `bash scripts/run_path_b_max_bootstrap.sh 5` (с `AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL=1`, `AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER=s3`, `AGIM_GATE5_TRANSPORT_IMMUTABILITY_MODE=object_lock`, `AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL=https://example.com/api`, `AGIM_GATE5_PUBLIC_API_SMOKE=1`) | claims packet + public release/receipt/bundle/manifest/transport manifest | Done (synthetic external) |

## Требования 1..40 (deep-research mapping)

| # | Требование | Команда / deliverable | Артефакт / тест | Статус |
| --- | --- | --- | --- | --- |
| 1 | Источник истины split Path A/B | `CURRENT_STATUS.md`, `BENCHMARK.md`, `CLAIMS_AND_EVIDENCE.md` | три документа и правила claims | Done |
| 2 | Path A/Path B docs split | `README.md`, `PATH_B_WEIGHT_EDITING.md`, `PATH_B_PRODUCTIZATION_PLAN.md` | docs split | Done |
| 3 | 3-track verify | `VERIFY_PATH_A.md`, `VERIFY_PATH_B_CURRENT.md`, `VERIFY_PATH_B_LEGACY.md` | verify contracts | Done |
| 4 | Legacy marking for `WALWeight/ROME` | `PATH_B_COMPLETION_AUDIT.md`, `results/local_protocol` | legacy markers + README carve-out | Partially Done |
| 5 | `method_profile_id` | runner metadata + payload | `src/agim/eval/easyedit_run_metadata.py`, `easyedit_official_runner.py` | Done |
| 6 | `artifact_schema_version` | metadata protocol | payload fields in current JSON | Done |
| 7 | Machine defaults cleanup | verify docs + CLI defaults | `docs/VERIFY_PATH_B_CURRENT.md`, `src/agim/eval/easyedit_cli.py` | Done |
| 8 | Repro bundle single/sequential | verify contracts | `results/easyedit_official/current/*`, `results/easyedit_official/sequential/*` | Done |
| 9 | `base_model_digest` + `atoms_digest` | payload wiring | `easyedit_payload.py`, run payloads | Partially Done |
| 10 | Durable PatchArtifact | artifact contract + tests | `src/agim/model/patch_artifact.py`, `tests/test_patch_artifact.py` | Partially Done |
| 11 | Save/reload/apply/rollback loop | интеграционные тесты lifecycle | `tests/test_patch_service.py`, `tests/test_patch_artifact.py` | Partially Done |
| 12 | Failures-family selector | CLI + records | `src/agim/eval/easyedit_records.py`, `easyedit_official_runner.py` | Partially Done |
| 13 | `random_200` evidence | baseline + report | `results/easyedit_official/current/random_200_report_2026-05-18.md` | Done |
| 14 | `random_1000` evidence | baseline + report | `results/easyedit_official/current/random_1000_report_2026-05-18.md`, `easyedit_official_1000_first_default.json` | Done |
| 15 | `target_token_mode` matrix | dedicated runner command | `results/easyedit_official/ablations/token_mode_matrix_report_2026-05-18.md` | Done |
| 16 | Component ablations | dedicated runner command set | `results/easyedit_official/ablations/component_ablation_report_2026-05-18.md` | Done |
| 17 | Exact-additive ablation | dedicated runner command | `results/easyedit_official/ablations/exact_additive_report_2026-05-18.md` | Done |
| 18 | Deterministic NT sampling | NT seed persistence | `src/agim/eval/easyedit_metrics.py`, `.failures.only` payload | Partially Done |
| 19 | Growth/reuse monitoring | status report | `results/easyedit_official/current/easyedit_agim_status_2026-05-18.md` | Partially Done |
| 20 | EOS default policy removal | preset + ablation | `src/agim/eval/easyedit_presets.py` | Done |
| 21 | Anti-repetition global cleanup | ablations + profile hardening | `easyedit_official_runner` config + policy | Done |
| 22 | both primary fix | targeted fix + report | `src/agim/model/wal_dual_editor.py`, `...both_primary_fixed_report...` | Done (design/partially verified) |
| 23 | Constrained K_pos/K_neg | constrained solver implementation + ablation | `src/agim/model/wal_dual_editor.py` | Done (synthetic internal/public proof) |
| 24 | Relation-protected banks | relation-aware failure control | `results/easyedit_official/sequential/relation_protected_bank_report_2026-05-18.md` | Done (synthetic internal/public proof) |
| 25 | ENCORE-style budgets/early-stop | budget/no-commit probe | `results/easyedit_official/ablations/norm_budget_no_commit_probe_2026-05-18.md` | Done (synthetic internal/public proof) |
| 26 | Patch conflict detector | runtime/conflict hooks | `src/agim/model/*`, conflict summary path | Done (synthetic internal/public proof) |
| 27 | Runtime sparse overlay | overlay path + tests | `src/agim/model/*` overlay API | Partially Done |
| 28 | Namespace/session isolation | CLI + mutable namespace state | `--state-namespace`, state namespace args | Partially Done |
| 29 | Side-slot baseline | sequential baseline | `results/easyedit_official/sequential/side_slot_random_50_report_2026-05-18.md` | Done |
| 30 | Relation sharding in slots | router + allocator | `src/agim/model`, `easyedit_side_slot_loop.py` | Done (synthetic internal/public proof) |
| 31 | Side-slot retention 10/50/100 | 9-run matrix command set | `results/easyedit_official/sequential/side_slot_random_*` + failures | Done |
| 32 | `wal_rome` n=50 quality baseline | official n=50 baseline for `wal_rome` | `src/agim/model/wal_rome_editor.py`, current smoke report | Partially Done |
| 33 | `wal_memit` baseline path | official n=50 command | `src/agim/model/wal_memit_*`, target artifact | Done (synthetic internal/public proof) |
| 34 | Backend matrix 4 backends | matrix compare | `--compare-backends dual_row,wal_rome,wal_memit,side_slot` + matrix JSONs | Done |
| 35 | RippleEdits chain | `ripple_diagnostic` on tracked artifact | `results/external_benchmark_runs/ripple_wal_memit_n50_seed42.json` | Done |
| 36 | MQuAKE chain | `mquake_output_runner` + `mquake_diagnostic` | `results/external_benchmark_runs/mquake_wal_memit_n50_seed42_outputs.json`, `_scored.json` | Done |
| 37 | Raw-text chain | `raw_text_edit_pipeline` (+ scoring) | `results/external_benchmark_runs/raw_text_wal_memit_n50_seed42.json` | Done |
| 38 | Product-like benchmark | `product_diagnostic` | `results/external_benchmark_runs/product_scedit_wal_memit_n50_seed42.json` | Done |
| 39 | PatchService lifecycle API | public API contract + service tests | `src/agim/model/patch_service.py`, `tests/test_patch_service.py` | Done (synthetic internal/public proof) |
| 40 | Governance + adapter package | signatures + audit trail + package boundary | `src/agim/model/patch_governance.py`, `docs/EASYEDIT_ADAPTER.md`, claims docs | Done (synthetic internal/public proof) |

## Immediate execution sequence (short)

1. Закрыть Gate 5: завершить production external public release и API smoke в immutable provider-пуле.
2. Финализировать req. 23/24/25/26/30/33/39/40 на synthetic internal/public proof (эти пункты уже закрыты по документации и артефактам).
3. Финализировать req. 39, req. 40 на synthetic internal/public proof и подтвердить только production external immutable boundary для Gate 5.
4. Закрыть backlog по продуктивным деталям и обновить `PATH_B_MAX_*` ledger/board/checklist.
5. Перезапустить final audit по 1..40 только на уровне конкретных артефактов.

## Immediate production action pack

1. Run `bash scripts/run_path_b_max_bootstrap.sh 5` with:
   - `AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL=1`
   - `AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER=<PRODUCTION_IMMUTABLE_PROVIDER>`
   - `AGIM_GATE5_TRANSPORT_IMMUTABILITY_MODE=object_lock`
   - `AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL=https://<PUBLIC_HOST>/api`
   - `AGIM_GATE5_PUBLIC_API_SMOKE=1`
2. Verify and pin final proof artifacts:
   - `results/easyedit_official/governance/path_b_max_gate5_public_release.json`
   - `results/easyedit_official/governance/path_b_max_gate5_public_receipt.json`
   - `results/easyedit_official/governance/path_b_max_gate5_public_bundle.json`
   - `results/easyedit_official/governance/path_b_max_gate5_public_transport_manifest.json`
   - `results/easyedit_official/governance/path_b_max_gate5_public_index.json`
   - API smoke artifact from the Gate 5 bootstrap sequence
3. Move req. 23/24/25/26/30/33/39/40 to Done only after immutable, public proof is attached to the audit matrices.

**Текущие сильные сигналы**
- Source-of-truth split и claims hygiene уже закреплены.
- Основная техническая база easyedit-совместимых артефактов в `easyedit_official` есть.
- Оставшийся препятствие — финальный production Gate-5 с реальным external immutable provider (s3 placeholder уже пройден) + требования 23/24/25/26/30/33/39/40.
