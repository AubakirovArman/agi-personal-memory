# PATH B MAX — Максимальный план (Current snapshot, 2026-05-19)

## 1) Реестр цели (из задачи пользователя)
1. Прочитать `sites/deep-research-report (5).md`.
2. Составить максимальный план выполнения PATH B MAX.
3. Выполнять пункты плана до технически закрытого состояния без пропусков.

## 2) Audit status на 2026-05-19 (фактический)
### Hard gates
- Gate 1 (baseline) — закрыт.
- Gate 2 (backend matrix) — закрыт.
- Gate 3 (external chain) — закрыт с оговоркой: это локальная диагностика (не внешняя публичная лидир-таблица).
- Gate 4 (side-slot) — закрыт.
- Gate 5 (governance proof) — частично закрыт в продуктовом смысле:
  - локальный proof → публичный release/index/receipt/bundle/transport manifest + верификация.
  - синтетический production-like run выполнен через `AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL=1`, `AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER=s3`, `AGIM_GATE5_TRANSPORT_IMMUTABILITY_MODE=object_lock`, `AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL=https://example.com/api`, `AGIM_GATE5_PUBLIC_API_SMOKE=1`.

### Промежуточный статус по req
- Закрыты в явной форме: `9,10,11,18,19,21,22,29,31,32,34,35,36,37,38`.
- Закрыты synthetic internal/public proof: `23,24,25,26,30,33,39,40`.
- Gate 5 частично закрыт: см. `results/easyedit_official/governance/path_b_max_gate5_public_release.json` и сопутствующие артефакты.

## 3) Prompt-to-artifact checklist (все требования / файлы / команды / проверки)

### Req 1–10
1. `results/easyedit_official/current/random_50_seed_42_wal_memit.json` — evidence.
2. `results/easyedit_official/current/random_50_seed_42_wal_memit.failures.json` — evidence.
3. `docs/CURRENT_STATUS.md` и `docs/BENCHMARK.md` — evidence.
4. `scripts/run_path_b_max_hard_gates.sh` — includes gate orchestration.
5. `src/agim/eval/easyedit_loader.py`, `src/agim/eval/easyedit_cli.py` — CLI/loader readiness evidence.
6. `src/agim/eval/easyedit_presets.py` — preset control evidence.
7. `src/agim/model/patch_artifact.py` — artifact schema/budget evidence.
8. `src/agim/model/patch_service.py` — PatchService lifecycle evidence.
9. `src/agim/model/patch_governance.py` — governance evidence.
10. `scripts/run_path_b_max_patch_service_governance_proof.py` — gate 5 proof execution.

### Req 11–20
11. Reload-safe patch apply path — `src/agim/model/patch_service.py` + docs in plan status files.
12. Namespace и состояние — `src/agim/model/patch_artifact.py`, `src/agim/eval/easyedit_cli.py` (`--state-namespace`).
13. Diff/inspect/list API — `src/agim/model/patch_service.py`.
14. Dual-mode и relation slot state — `src/agim/model/wal_dual_editor.py`.
15. SideSlot memory type — `src/agim/model/side_slot_memory.py`.
16. Side-slot sequential sweep — `src/agim/eval/easyedit_side_slot_loop.py`.
17. Backend run metadata — `src/agim/eval/easyedit_cli.py`.
18. Command/sha/digest metadata in artifacts — `scripts/run_path_b_max_hard_gates.sh`, loader outputs.
19. Norm budget in artifact — `src/agim/model/patch_artifact.py`.
20. Command/metadata consistency checks — docs + script outputs.

### Req 21–30
21. Anti-repetition cleanup policy — закрыто:
- `src/agim/model/wal_dual_editor.py` (`clamp_anti_scope`).
- `src/agim/eval/easyedit_cli.py` (`--clamp-anti-scope`).
- `tests/test_easyedit_artifacts.py` (`--clamp-anti-scope` CLI flag coverage).
- `tests/test_easyedit_adapter.py` (`clamp_anti_scope` pass-through в adapter).
22. (междушаги/контроль инвариантов, см. план документа).
23. Constrained K_pos/K_neg solver — закрыт (synthetic internal/public proof):
- `src/agim/model/wal_dual_editor.py` через `positive_constraint_mode=constrained`.
- `src/agim/model/wal_dual_helpers.py` with `constrained_projection_key`.
- `src/agim/eval/easyedit_cli.py` параметризован ключами `--positive-constraint-mode constrained` и k-предикатами.
- `tests/test_easyedit_official_metrics.py` (`constrained_projection_key` regression).
24. Stability/compat checks из ранних этапов — synthetic internal/public proof.
25. Budget/no-commit guard — синтетически закрыт в сервисном слое:
- `src/agim/model/patch_artifact.py` (`NormBudgetPolicy.max_shared_row_delta_norm`).
- `src/agim/model/patch_service.py` (shared-row no-commit, `_shared_row_deltas`, `max_shared_row_delta_norm` enforcement на propose/approve/apply, release при rollback).
- `scripts/run_path_b_max_patch_service_governance_proof.py` (`strict_budget_guard`, `shared_row_budget_guard`).
26. Conflict detector + negative control — синтетически закрыт:
- `src/agim/model/wal_dual_editor.py` now writes `relation_protected_ids` into edit backup.
- `src/agim/eval/easyedit_budget.py` writes `subject_token_ids`, `target_token_ids`, `control_row_ids`, `relation_shard`, `relation_slot_id`, `relation_slot_buckets`, `protected_basis_ids` into artifact metadata.
- `src/agim/model/patch_artifact.py` включает `same_relation_shard`/`same_relation_slot` в детекторный payload и risk flags.
- `src/agim/model/patch_service.py` и `scripts/run_path_b_max_patch_service_governance_proof.py` покрывают runtime-conflict flow (`conflict_summary`, `strict_conflict_guard`) для overlapping rows/tokens/eos/protected/relation-slot signals.
27. Sparse overlay path — `src/agim/eval/easyedit_side_slot_loop.py`.
28. Namespace mutable state — `src/agim/eval/easyedit_cli.py`, `src/agim/model/patch_artifact.py`.
29. Side-slot sweep completion — закрыто.
30. Relation-slot awareness — синтетически закрыт:
- `src/agim/model/side_slot_memory.py` теперь содержит `RelationSlotAllocator` и `relation_slot_for()` с allocator-grade детерминированным разбиением.
- `src/agim/model/side_slot_memory.py` фиксирует allocator-конфигурацию через `allocator_summary()`.
- `src/agim/eval/easyedit_side_slot_loop.py` маршрутизирует overlay через allocator из памяти и пишет `relation_slot_allocator` в checkpoint/retention.
- `src/agim/eval/easyedit_budget.py` and `scripts/run_path_b_max_patch_service_governance_proof.py` сохраняют и используют `relation_slot_id`/`relation_slot_buckets` в metadata для согласованного conflict- и allocator-прохода.

### Req 31–40
31. Полный набор side-slot результатов — закрыто:
- `results/easyedit_official/side_slot/...` (18 файлов для n=10/50/100, seeds 42/43/44).
32. Dual backend n=50 — закрыто.
33. Wal_memit backend n=50 — закрыто.
34. Полная матрица (4 backend) — закрыто.
35. Ripple — закрыто:
- `results/external_benchmark_runs/ripple_wal_memit_n50_seed42.json`.
36. MQuAKE proposals/scored — закрыто:
- `results/external_benchmark_runs/mquake_wal_memit_n50_seed42_outputs.json`
- `results/external_benchmark_runs/mquake_wal_memit_n50_seed42_scored.json`.
37. Raw text proposals — закрыто:
- `results/external_benchmark_runs/raw_text_wal_memit_n50_seed42.json`.
38. Product s-edit diagnostic — закрыто:
- `results/external_benchmark_runs/product_scedit_wal_memit_n50_seed42.json`.
39. Публичный API контракт и клиентский контракт для PatchService — синтетически закрыт: закрыты внутренние API и локальные payload (`src/agim/model/patch_service.py`, `src/agim/model/patch_artifact.py`, `docs`), плюс отдельный контрактный артефакт `docs/PATH_B_GATE5_PUBLIC_CONTRACT.md`, hardened `scripts/run_path_b_max_gate5_public_api.py` с `--require-production-external`, enhanced `scripts/run_path_b_max_gate5_audit_consumer.py` и `scripts/run_path_b_max_gate5_verify_publication.py` для claim-chain/proof consistency checks, и external smoke в `scripts/run_path_b_max_hard_gates.sh`; остаются зависимости от полноценного production storage provider/immutable backend.
40. Claims-lock/public governance release — синтетически закрыт: закрыт public packet + local verify + tenant publish index + индекс-верификация + consumer API + immutable receipt + receipt verify + immutable bundle + bundle verify + transport manifest create/verify (`scripts/run_path_b_max_publish_gate5_release.py`, `results/easyedit_official/governance/path_b_max_gate5_public_index.json`, `scripts/run_path_b_max_verify_gate5_index.py`, `scripts/run_path_b_max_gate5_public_api.py`, `scripts/run_path_b_max_gate5_audit_consumer.py`, `scripts/run_path_b_max_gate5_create_receipt.py`, `scripts/run_path_b_max_verify_gate5_receipt.py`, `scripts/run_path_b_max_gate5_create_bundle.py`, `scripts/run_path_b_max_verify_gate5_bundle.py`, `scripts/run_path_b_max_gate5_create_transport_manifest.py`, `scripts/run_path_b_max_verify_gate5_transport_manifest.py`, `scripts/run_path_b_max_gate5_verify_publication.py`), остаток — production external immutable-provider + публичный base URL.
Выполнена контрактная донастройка: consumer-проверка включает валидацию `release_schema_version`.
Старые имена entrypoint-скриптов (`run_path_b_max_create_receipt.py`, `run_path_b_max_create_bundle.py`, `run_path_b_max_verify_publication.py`) оставлены как shim-обёртки для обратной совместимости.

## 4) Что нужно закрыть в коде/процессе дальше (по приоритету)
1. Закрыть/переподтвердить req 23/24/25/26/30/33/39/40 через production/official fixture-контур: synthetic proof уже есть, но production-external Gate 5 должен закрыть публично-immutable boundary.
2. Довести req 23/24/25/26/30/33/39/40 до опубликованного public contract:
   - `schema_version` + `claims_digest_chain` + tenant scope + public verify flow.
   - transport manifest endpoint и external check через consumer.
3. Выпустить публичный immutable gate-5 release packet поверх локального proof:
   - command + git_sha + input artifacts + proof bundle.

## 5) Предлагаемый порядок исполнения
1) Gate 5 public release packet (req23/24/25/26/30/33/39/40)  
2) req23 (constrained K-pos/K-neg)  
3) req23/24/25/26/30/33/39/40 production verification artifacts  
4) Финальная синхронизация всех старых `PATH_B_MAX_*` файлов статусами.

## 6) Команды, которые использоваться в execution mode
- `cd sites/agi_personal_memory && ./scripts/run_path_b_max_bootstrap.sh` (оркестрация hard-gates)
- `cd sites/agi_personal_memory && ./scripts/run_path_b_max_hard_gates.sh`
- `AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL=1 AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER=<PRODUCTION_IMMUTABLE_PROVIDER> AGIM_GATE5_TRANSPORT_IMMUTABILITY_MODE=object_lock AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL=https://<PUBLIC_HOST>/api ./scripts/run_path_b_max_hard_gates.sh 5`  
- `AGIM_GATE5_PUBLIC_API_SMOKE=1 AGIM_GATE5_PUBLIC_API_PORT=8010 ./scripts/run_path_b_max_hard_gates.sh 5`  
- `AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL=1 AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER=<PRODUCTION_IMMUTABLE_PROVIDER> AGIM_GATE5_TRANSPORT_IMMUTABILITY_MODE=object_lock AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL=https://<PUBLIC_HOST>/api AGIM_GATE5_PUBLIC_API_SMOKE=1 ./scripts/run_path_b_max_hard_gates.sh 5`  
- `AGIM_GATE5_PUBLIC_API_SMOKE=1 AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL=1` additionally requires transport metadata env above.
  In production mode, `AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER`, `AGIM_GATE5_TRANSPORT_IMMUTABILITY_MODE`, and `AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL` are required.
- `cd sites/agi_personal_memory && python scripts/run_path_b_max_patch_service_governance_proof.py` (локальная генерация proof-пакета, если требуется)
- `cd sites/agi_personal_memory && python scripts/run_path_b_max_gate5_release_packet.py --proof-path results/easyedit_official/governance/path_b_max_gate5_proof.json`
- `cd sites/agi_personal_memory && python scripts/run_path_b_max_verify_gate5_release.py --tenant public --release-path results/easyedit_official/governance/path_b_max_gate5_public_release.json`
- `cd sites/agi_personal_memory && python scripts/run_path_b_max_publish_gate5_release.py --release-path results/easyedit_official/governance/path_b_max_gate5_public_release.json --tenant public --channel public`
- `cd sites/agi_personal_memory && python scripts/run_path_b_max_verify_gate5_index.py --tenant public --channel public`
- `cd sites/agi_personal_memory && python scripts/run_path_b_max_gate5_public_api.py --index-path results/easyedit_official/governance/path_b_max_gate5_public_index.json --tenant public --channel public --receipt-path results/easyedit_official/governance/path_b_max_gate5_public_receipt.json --bundle-path results/easyedit_official/governance/path_b_max_gate5_public_bundle.json --transport-manifest-path results/easyedit_official/governance/path_b_max_gate5_public_transport_manifest.json --port 8010`
- `cd sites/agi_personal_memory && python scripts/run_path_b_max_gate5_audit_consumer.py --api-base http://127.0.0.1:8010 --tenant public --channel public`
- `cd sites/agi_personal_memory && python scripts/run_path_b_max_gate5_audit_consumer.py --api-base http://127.0.0.1:8010 --tenant public --channel public --check-receipt --expected-receipt-sha256 <sha256>`
- `cd sites/agi_personal_memory && python scripts/run_path_b_max_gate5_audit_consumer.py --api-base http://127.0.0.1:8010 --tenant public --channel public --check-bundle --expected-bundle-sha256 <sha256> --expected-release-schema-version path_b_max_gate5_release.v1`
- `cd sites/agi_personal_memory && python scripts/run_path_b_max_gate5_create_receipt.py --index-path results/easyedit_official/governance/path_b_max_gate5_public_index.json --tenant public --channel public --receipt-path results/easyedit_official/governance/path_b_max_gate5_public_receipt.json`
- `cd sites/agi_personal_memory && python scripts/run_path_b_max_verify_gate5_receipt.py --receipt-path results/easyedit_official/governance/path_b_max_gate5_public_receipt.json`
- `cd sites/agi_personal_memory && python scripts/run_path_b_max_gate5_create_bundle.py --index-path results/easyedit_official/governance/path_b_max_gate5_public_index.json --receipt-path results/easyedit_official/governance/path_b_max_gate5_public_receipt.json --bundle-path results/easyedit_official/governance/path_b_max_gate5_public_bundle.json --tenant public --channel public`
- `cd sites/agi_personal_memory && python scripts/run_path_b_max_verify_gate5_bundle.py --bundle-path results/easyedit_official/governance/path_b_max_gate5_public_bundle.json`
- `cd sites/agi_personal_memory && python scripts/run_path_b_max_gate5_create_transport_manifest.py --release-path results/easyedit_official/governance/path_b_max_gate5_public_release.json --index-path results/easyedit_official/governance/path_b_max_gate5_public_index.json --receipt-path results/easyedit_official/governance/path_b_max_gate5_public_receipt.json --bundle-path results/easyedit_official/governance/path_b_max_gate5_public_bundle.json --manifest-path results/easyedit_official/governance/path_b_max_gate5_public_transport_manifest.json --tenant public --channel public`
- `cd sites/agi_personal_memory && python scripts/run_path_b_max_gate5_create_transport_manifest.py --release-path results/easyedit_official/governance/path_b_max_gate5_public_release.json --index-path results/easyedit_official/governance/path_b_max_gate5_public_index.json --receipt-path results/easyedit_official/governance/path_b_max_gate5_public_receipt.json --bundle-path results/easyedit_official/governance/path_b_max_gate5_public_bundle.json --manifest-path results/easyedit_official/governance/path_b_max_gate5_public_transport_manifest.json --tenant public --channel public --public-base-url https://<PUBLIC_HOST>/api --storage-provider <PRODUCTION_IMMUTABLE_PROVIDER> --immutability-mode object_lock --require-production-external`
- `cd sites/agi_personal_memory && python scripts/run_path_b_max_verify_gate5_transport_manifest.py --manifest-path results/easyedit_official/governance/path_b_max_gate5_public_transport_manifest.json`
- `cd sites/agi_personal_memory && python scripts/run_path_b_max_gate5_verify_publication.py --release-path results/easyedit_official/governance/path_b_max_gate5_public_release.json --index-path results/easyedit_official/governance/path_b_max_gate5_public_index.json --receipt-path results/easyedit_official/governance/path_b_max_gate5_public_receipt.json --bundle-path results/easyedit_official/governance/path_b_max_gate5_public_bundle.json --transport-manifest-path results/easyedit_official/governance/path_b_max_gate5_public_transport_manifest.json --tenant public --channel public`
- `cd sites/agi_personal_memory && python scripts/run_path_b_max_gate5_verify_publication.py --api-base http://127.0.0.1:8010 --tenant public --channel public --expected-release-sha256 <release-sha256> --expected-receipt-sha256 <receipt-sha256> --expected-bundle-sha256 <bundle-sha256> --transport-manifest-path results/easyedit_official/governance/path_b_max_gate5_public_transport_manifest.json --expected-transport-manifest-sha256 <transport-manifest-sha256> --check-transport-manifest --expected-transport-manifest-schema-version path_b_max_gate5_transport_manifest.v1 --require-production-external`
- Backward-compatible invocation for old command references:
  - `python scripts/run_path_b_max_create_receipt.py ...`
  - `python scripts/run_path_b_max_create_bundle.py ...`
  - `python scripts/run_path_b_max_verify_publication.py ...`

В командах проверки с `--api-base http://127.0.0.1:8010` используется локальный API smoke endpoint; для production-external требования по внешнему `https://<PUBLIC_HOST>/api`, immutability-provider и non-local transport manifest остаются обязательными.
- Updated audit verifier now validates API-consumer checks for receipt/bundle/release schema and release/receipt/bundle/transport-manifest hashes:
  - `python scripts/run_path_b_max_gate5_verify_publication.py --api-base <api> --tenant public --channel public --expected-release-schema-version path_b_max_gate5_release.v1 --expected-receipt-schema-version path_b_max_gate5_receipt.v1 --expected-bundle-schema-version path_b_max_gate5_bundle.v1 --expected-transport-manifest-schema-version path_b_max_gate5_transport_manifest.v1 --check-transport-manifest --require-production-external`

## 7) Критерий завершения для каждого шага
- Каждый req закрывается только после:
  - конкретного артефакта,
  - команды/теста, который его породил,
  - подтверждения через соответствующий gate/review/docs artifact,
  - отсутствия незаполненных блокирующих пунктов в completion matrix.

## 8) Финальные критерии достижения objective
- `read`: подтверждено прочтение `sites/deep-research-report (5).md` и перенос всех требований.
- `plan`: все требования и команды из отчёта отражены в едином чек-листе со статусом закрыто/частично/открыто.
- `execution`: Gate 1–5 выполнены без пробелов между proof и public verification.
- `evidence`: req25/26/30/39/40 закрыты только с дополнительными артефактами, audit-треками и публичным verify-flow. req23 дополнительно подтвержден CLI/adapter тестами на `positive_constraint_mode=constrained`.
- `non-goals`: отсутствуют неучтенные blockers вне prompt-to-artifact карты.
- `current-state note`: req.23/24/25/26/30/33/39/40 считаются закрытыми на synthetic internal/public proof; proof/public release/receipt/bundle и public lifecycle artifacts сформированы. Остаётся production smoke с фактическим внешним immutable-object provider и публичным base URL (`AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL=1 ...`).
- Для закрытия req23/24/25/26/30/33/39/40 в audit нужно убедиться, что `run_path_b_max_gate5_verify_publication.py --api-base ... --expected-release-sha256 <sha> --expected-receipt-sha256 <sha> --expected-bundle-sha256 <sha> --expected-transport-manifest-sha256 <sha>` проходит успешно и совпадают все release/receipt/bundle/manifest digest-цепочки.
