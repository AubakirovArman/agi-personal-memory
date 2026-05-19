# Path B Maximum Execution Blueprint

Source objective: `/mnt/hf_model_weights/arman/3bit/sites/deep-research-report (5).md`  
Date: 2026-05-18

## Objective in deliverables

- Build a claim-safe Path B with reproducible evidence.
- Keep Path A and Path B split across official artifacts.
- Run hard gates and then perform artifact-complete audit of all 40 requirements.

## Hard acceptance gates (must all be done for completion)

1. `wal_memit` official baseline at `n=50`.
   - Status: done
   - Artifact: `results/easyedit_official/current/random_50_seed_42_wal_memit*.json`
   - Command: `python -m agim.eval.easyedit_official_runner --edit-backend wal_memit ...`

2. Full backend comparison with `dual_row`, `wal_rome`, `wal_memit`, `side_slot`.
   - Status: done
   - Artifacts:
     - `results/easyedit_official/ablations/backend_matrix_random_50_seed42.json`
     - `results/easyedit_official/ablations/backend_matrix_random_50_seed42.dual_row.json`
     - `results/easyedit_official/ablations/backend_matrix_random_50_seed42.wal_rome.json`
     - `results/easyedit_official/ablations/backend_matrix_random_50_seed42.wal_memit.json`
     - `results/easyedit_official/ablations/backend_matrix_random_50_seed42.side_slot.json`
     - `results/easyedit_official/ablations/backend_matrix_random_50_report_2026-05-18.md`

3. External chain coverage (Ripple / MQuAKE / raw-text / product-like).
   - Status: done
   - Artifacts: `results/external_benchmark_runs/*n50_seed42*`

4. Side-slot retention hardening (`n=10/50/100`, `seed 42/43/44`).
   - Status: done
   - Artifacts: `results/easyedit_official/sequential/side_slot_random_{10,50,100}_seed_{42,43,44}_seq.json` and failures

5. PatchService/governance proof packet.
   - Status: Done (synthetic internal/public proof)
   - Requirement: one public immutable proof packet with `propose/simulate/run_canaries/approve/apply/rollback/inspect/diff` and tenant claims/audit chain

## Execution cadence

- S1: confirm Gate 5 synthetic internal/public proof path (already done) and close remaining production-external immutable boundary.
- S2: close req. 23/24/25/26/30/33/39/40 in synthetic public proof context (done), then validate final public provider boundary for Gate 5.
- S3: close req. 25/26/30 residual risk controls (technical, non-blocking for final objective statement).
- S4: re-audit all 40 requirements and synchronize completion matrices.

## Current top priorities now

1. Confirm production external immutable Gate 5 (`AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL=1`) and run API smoke (`AGIM_GATE5_PUBLIC_API_SMOKE=1`) against real provider endpoint.
2. Reconfirm req. 23, 24, 25, 26, 30, 33, 39, 40 evidence was generated in synthetic internal/public proof path; keep final proof on production external immutable boundary.
3. Finalize req. 25, 26, 30 residual risk controls in public audit notes (non-blocking once Gate 5 is production-verified).
4. Sync status docs after completion:
   - `docs/PATH_B_MAX_EXECUTION_CHECKLIST.md`
   - `docs/PATH_B_MAX_COMPLETION_MATRIX.md`
   - `docs/PATH_B_MAX_STATUS_BOARD.md`

## Canonical one-command execution pack

Для production-Gate-5:

```bash
cd /mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory
export AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL=1
export AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER=<PRODUCTION_IMMUTABLE_PROVIDER>
export AGIM_GATE5_TRANSPORT_IMMUTABILITY_MODE=object_lock
export AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL=https://<PUBLIC_HOST>/api
export AGIM_GATE5_PUBLIC_API_SMOKE=1
bash scripts/run_path_b_max_bootstrap.sh 5
```

`AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER` должен быть внешним immutable backend, а не `mock-object-store`/`local_filesystem`/`local`.
`AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL` должен быть `https://<PUBLIC_HOST>/api` (не localhost/127.0.0.1/0.0.0.0/[::1]).

Для full-coverage replay:

```bash
bash scripts/run_path_b_max_bootstrap.sh 1 2 4 3 5
```

После каждого этапа обновлять:
- `docs/PATH_B_MAX_COMPLETION_MATRIX.md`
- `docs/PATH_B_MAX_EXECUTION_CHECKLIST.md`
- `docs/PATH_B_MAX_STATUS_BOARD.md`
- `docs/PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md`
- `docs/PATH_B_COMPLETION_AUDIT.md`
