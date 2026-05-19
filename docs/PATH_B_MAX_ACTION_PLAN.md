# Path B Max Action Plan (Objective Executor)

Цель: довести репозиторий до доказательного состояния по каждому пункту из `sites/deep-research-report (5).md`.

Максимальная трассировка лежит в:
- [PATH_B_MAX_MAX_PLAN.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_MAX_PLAN.md)
- [PATH_B_MAX_MAX_AUDIT.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_MAX_AUDIT.md)

Источник прогресса:
- `PATH_B_MAX_EXECUTION_CHECKLIST.md`
- `PATH_B_MAX_COMPLETION_MATRIX.md`
- `PATH_B_MAX_STATUS_BOARD.md`
- `PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md`
- `PATH_B_MAX_PROMPT_ARTIFACT_AUDIT.md`

## Hard gates

1. Gate 1: `wal_memit` official n=50
   - Evidence: `results/easyedit_official/current/random_50_seed_42_wal_memit*.json`
   - Status: done

2. Gate 2: full backend matrix (`dual_row`, `wal_rome`, `wal_memit`, `side_slot`)
   - Evidence: `results/easyedit_official/ablations/backend_matrix_random_50_seed42*.json`, `backend_matrix_random_50_report_2026-05-18.md`
   - Status: done

3. Gate 3: external consequence chain (Ripple, MQuAKE, raw-text, product-like)
   - Evidence: `results/external_benchmark_runs/*n50_seed42*`
   - Status: done

4. Gate 4: side-slot retention hardening (`10/50/100 × 42/43/44`)
   - Evidence: `results/easyedit_official/sequential/side_slot_random_{10,50,100}_seed_{42,43,44}_seq.json` + failures
   - Status: done

5. Gate 5: PatchService/governance proof packet
   - Evidence: public immutable proof packet with API lifecycle surface and audit trail
   - Status: synthetic done (production external boundary still pending)

## Операционный порядок на ближайшие этапы

1. Закрыть Gate 5 в production external режиме:
   - `AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL=1`
   - `AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER=<production_immutable_provider>`
   - `AGIM_GATE5_TRANSPORT_IMMUTABILITY_MODE=object_lock`
   - `AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL="https://<PUBLIC_HOST>/api"`
   - `AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL` must not use `localhost/127.0.0.1/0.0.0.0/[::1]`
   - `AGIM_GATE5_PUBLIC_API_SMOKE=1`
   - `bash scripts/run_path_b_max_bootstrap.sh 5`

2. После Gate 5 закрыть публичными доказательствами:
   - req. 23/24/25/26/30/33/39/40 (`K_pos/K_neg`, `relation-protected`, `no-commit` / budget guard, conflict detector, relation-aware slot sharding, `wal_memit` quality path, PatchService lifecycle, governance package) уже закрыты на synthetic internal/public proof; необходимо подтверждение production-external immutable-провайдера и публичного API smoke.

3. Финализировать residual risk controls, если они ещё не в public chain:
   - req. 25 (`no-commit`/budget guard)
   - req. 26 (`conflict detector`)
   - req. 30 (`relation-aware` slot sharding)

## Что обновлять после каждого этапа

- `docs/PATH_B_MAX_EXECUTION_CHECKLIST.md`
- `docs/PATH_B_MAX_COMPLETION_MATRIX.md`
- `docs/PATH_B_MAX_STATUS_BOARD.md`
- `docs/PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md`
- `docs/PATH_B_COMPLETION_AUDIT.md`
- `docs/CLAIMS_AND_EVIDENCE.md`

## Completion rule

Ни одна строка не переводится в `Done` без конкретного официального `json/md` артефакта и обновления audit matrix.
