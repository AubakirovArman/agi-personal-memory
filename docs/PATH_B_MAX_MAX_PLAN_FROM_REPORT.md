# Path B максимальный план (по deep-research-report (5))

## Цель

Закрыть все требования от 1 до 40 из отчёта в reproducible виде: каждый пункт должен иметь явный источник запуска и подтверждающий артефакт, а hard-gate claims — только после production external Gate 5 с внешним immutable-объектным провайдером и public API smoke.

## Prompt → Artifact чек-лист

1. Source-of-truth split (Path A/B): `CURRENT_STATUS.md`, `BENCHMARK.md`, `docs/CLAIMS_AND_EVIDENCE.md` — синхронизировано, закрыто.
2. README/Path docs split + контракт: `README.md`, `docs/PATH_B_WEIGHT_EDITING.md`, `docs/PATH_B_PRODUCTIZATION_PLAN.md` — закрыто.
3. Verify-tracks: `docs/HOW_TO_VERIFY.md`, `docs/VERIFY_PATH_B_CURRENT.md`, `docs/VERIFY_PATH_B_LEGACY.md` — частично пересобрать после final Gate 5.
4. Legacy payload demotion: `src/agim/eval/easyedit_official_runner.py`, local legacy references — частично.
5. Метод профилирования: runner + payload (`method_profile_id`) — закрыто.
6. `artifact_schema_version` — закрыто.
7. Удаление machine-specific defaults из public guidance — закрыто (`agim/eval/easyedit_cli.py`, docs).
8. Репродуцируемые bundle’ы single/current и sequential — закрыто.
9. `base_model_digest` / `atoms_digest` в payload — частично (должно пройти финальную проверку всех пайплайнов).
10. Durable `PatchArtifact` — частично.
11. save/reload/apply/rollback интеграционный контракт — частично.
12. failures families (`tf`, `ctx_gen`, `prob`, `vanilla_gen`) — частично.
13. `random_200` baseline — закрыто.
14. `random_1000` baseline — закрыто.
15. `target_token_mode` матрица — закрыто.
16. component ablations — закрыто.
17. exact-additive ablation — закрыто.
18. детерминированный NT режим — частично.
19. growth/reuse мониторинг (`edit_row_norm`, `edit basis reuse`) — частично.
20. EOS default policy — закрыто.
21. anti-repetition global cleanup — закрыто (`clamp_anti*` defaults переведены).
22. both-mode fix (`primary`/`contextual`) — частично.
23. constrained solve `K_pos/K_neg` — Done (synthetic internal/public proof).
24. relation-protected banks — Done (synthetic internal/public proof).
25. ENCORE-like budget/early-stop — Done (synthetic internal/public proof).
26. conflict detector — Done (synthetic internal/public proof).
27. runtime sparse overlay path — частично.
28. namespace isolation для patch state — частично.
29. side-slot baseline — закрыто.
30. side-slot relation-aware slot sharding — Done (synthetic internal/public proof).
31. side-slot retention 10/50/100 × 42/43/44 — закрыто.
32. WALRome n=50 baseline — частично.
33. WALMemit n=50 quality path — Done (synthetic internal/public proof).
34. Backend matrix 4 backends — закрыто.
35. RippleEdits consequence chain — закрыто.
36. MQuAKE consequence chain — закрыто.
37. raw-text chain — закрыто.
38. product-like benchmark chain — частично.
39. PatchService public lifecycle (`propose/simulate/canaries/approve/apply/rollback`) — Done (synthetic internal/public proof).
40. Governance + adapter package + public claims — Done (synthetic internal/public proof).

## Hard-gate status

1 — выполнен (`results/easyedit_official/current/random_50_seed_42_wal_memit*.json` + report).  
2 — выполнен (`results/easyedit_official/ablations/backend_matrix_random_50_seed42*`).  
3 — выполнен (`results/external_benchmark_runs/*`).  
4 — выполнен (`results/easyedit_official/sequential/side_slot_random_{10,50,100}_seed_{42,43,44}_seq.json`).  
5 — частично: локальный proof и публичные release artifacts есть, но production external immutable-провайдер с реальным `https://<PUBLIC_HOST>/api` + API smoke ещё не исполнены.

## Непосредственный порядок после локальной ревизии

1. Выполнить Gate 5 строго production external:
   - `AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL=1`
   - `AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER=<PRODUCTION_IMMUTABLE_PROVIDER>`
   - `AGIM_GATE5_TRANSPORT_IMMUTABILITY_MODE=object_lock`
   - `AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL=https://<PUBLIC_HOST>/api`
   - `AGIM_GATE5_PUBLIC_API_SMOKE=1`
   - `bash scripts/run_path_b_max_bootstrap.sh 5`
2. Обновить `docs/PATH_B_MAX_COMPLETION_MATRIX.md`, `PATH_B_MAX_EXECUTION_CHECKLIST.md`, `PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md`, `docs/PATH_B_COMPLETION_AUDIT.md`.
3. Финализировать req 23/24/25/26/30/33/39/40 в `docs/PATH_B_MAX_PROMPT_ARTIFACT_AUDIT.md` как synthetic internal/public proof, затем закрыть финальную production-external Gate 5.
4. Закрыть residual risk controls req 25/26/30 после подтверждения в публичном chain.
5. Запустить full re-audit всех 40 пунктов и зафиксировать финальный proof-state.

## Concrete next execution block (immediate)

1. Run production external Gate 5 command from `PATH_B_MAX_GATE_COMMANDS.md` and `PATH_B_MAX_HARDGATE_QUEUE.md`.
2. Validate all five governance artifacts:
   - `results/easyedit_official/governance/path_b_max_gate5_public_release.json`
   - `results/easyedit_official/governance/path_b_max_gate5_public_receipt.json`
   - `results/easyedit_official/governance/path_b_max_gate5_public_bundle.json`
   - `results/easyedit_official/governance/path_b_max_gate5_public_transport_manifest.json`
   - `results/easyedit_official/governance/path_b_max_gate5_public_index.json`
3. Capture public API smoke output/trace artifact and add it to governance chain docs.
4. Keep req. 23 / 24 / 25 / 26 / 30 / 33 / 39 / 40 as synthetic internal/public proof while executing final production-external Gate 5 verification.
