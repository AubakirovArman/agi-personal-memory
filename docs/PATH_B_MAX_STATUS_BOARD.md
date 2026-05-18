# Path B Status Board (Current vs Next)

Last updated: 2026-05-18

## What is complete right now

- Source-of-truth separation between Path A/Path B is in place.
  - `CURRENT_STATUS.md`, `BENCHMARK.md`, `docs/CLAIMS_AND_EVIDENCE.md`.
- Public reproducibility path is documented under `results/easyedit_official/*`.
  - `docs/VERIFY_PATH_B_CURRENT.md`, `docs/PATH_B_MAX_EXECUTION_CHECKLIST.md`.
- EasyEdit metadata protocol and schema are in use.
  - `src/agim/eval/easyedit_run_metadata.py`, `src/agim/eval/easyedit_official_runner.py`.
- Core hardening scaffolding is present for PatchArtifact, PatchService, and governance.
  - `src/agim/model/patch_artifact.py`, `tests/test_patch_artifact.py`,
    `src/agim/model/patch_service.py`, `src/agim/model/patch_governance.py`.
- Core evidence for 40-item plan is mapped; remaining points are all explicitly partial/not started in the completion matrix.
  - `docs/PATH_B_MAX_COMPLETION_MATRIX.md`.

## Current blockers (must close for completion)

1. `wal_memit` official n=50 baseline: no dedicated quality artifact in official bundle yet.
2. Backend matrix: need comparable quality matrix for `dual_row`, `wal_rome`, `wal_memit`, `side_slot`.
3. External model-output chain: RippleEdits / MQuAKE / raw-text / product benchmark missing as tracked model-output + report in one evidence cycle.
4. Sequential locality: side-slot retention still needs seeds/length matrix `10/50/100`.
5. Governance proof packet: full lifecycle contract (`propose/simulate/run_canaries/approve/apply/rollback/inspect/diff`) still requires one public claim packet.

### Artifact presence check (as of 2026-05-18)

- `results/easyedit_official/current/random_50_seed_42_wal_memit.json`: missing.
- `results/easyedit_official/ablations/backend_matrix_random_50_seed42.json`: missing.
- `results/easyedit_official/ablations/backend_matrix_random_50_seed42.{dual_row,wal_rome,wal_memit,side_slot}.json`: missing.
- `results/easyedit_official/ablations/backend_matrix_random_50_report_2026-05-18.md`: present.
- `results/external_benchmark_runs/*n50_seed42*` for Ripple/MQuAKE/raw-text/product: missing required files.
- `results/easyedit_official/sequential/side_slot_random_10/50/100_seed_{42,43,44}_seq.json`: missing.

## Completion audit against source objective

- Official compatibility hard proof: 5 hard gates from `PATH_B_MAX_HARDGATE_QUEUE.md` remain open until corresponding JSON+MD artifacts are in `results/easyedit_official/*` and `results/external_benchmark_runs/*`.
- 40-point evidence mapping: all items are mapped in `PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md`, but several critical rows remain `Not Started` or `Partial`.
- Proxy signals status: passing tests and document existence are not sufficient; artifact evidence still required for Gates 1-5 and rows 21, 23, 26, 30, 38.

## Evidence-backed status by group

- `easyedit_official` single-edit quality: Done for current baselines and ablations, but not yet complete for all hard backends.
- Sequential behavior: partial. Side-slot has baseline evidence at 50; 10/100 + multi-seed sweep still pending.
- External consequence metrics: partial. Diagnostic scripts and adapters exist, but tracked model-output/score package missing for at least one channel.
- Claims safety boundary: correct and enforced; no leaderboard or “solved sequential” claims should be treated as proven.

## Next 5 actions (ordered)

1. Run `--edit-backend wal_memit --n 50` official baseline, store JSON+MD in `results/easyedit_official/current/`.
2. Run full backend matrix including `dual_row`, `wal_rome`, `wal_memit`, `side_slot` on one sample-policy profile.
3. Close external chain by running and scoring at least one full dataset-level output run for RippleEdits and MQuAKE each.
4. Finish side-slot retention matrix for `n=10/50/100` with seeds `42/43/44` in `results/easyedit_official/sequential/`.
5. Create a single release-gate proof packet for PatchService/governance with all lifecycle ops and immutable audit trail references.

## Current command source

- `docs/PATH_B_MAX_GATE_COMMANDS.md`
- `docs/PATH_B_MAX_EXECUTION_RUNBOOK.md`
- `docs/PATH_B_MAX_EXECUTION_CHECKLIST.md`
- `docs/PATH_B_MAX_PROMPT_ARTIFACT_AUDIT.md`
- `docs/PATH_B_MAX_EXECUTION_BLUEPRINT.md`
- `docs/PATH_B_MAX_HARDGATE_QUEUE.md`
- `docs/PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md`
