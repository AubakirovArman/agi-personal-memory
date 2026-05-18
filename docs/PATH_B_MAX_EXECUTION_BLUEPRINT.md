# Path B Maximum Execution Blueprint

Source objective: `/mnt/hf_model_weights/arman/3bit/sites/deep-research-report (5).md`
Date: 2026-05-18

## Objective in deliverables

- Build a trustworthy Path B that is claim-safe.
- Separate Path A and Path B evaluation and production claims.
- Run evidence-first hard gates with official and tracked artifacts.
- Upgrade the architecture toward interference-safe editing (backends + governance + service).

## Hard acceptance gates (must all be done for completion)

1. `wal_memit` official baseline at `n=50`.
   - Command: `python -m agim.eval.easyedit_official_runner --edit-backend wal_memit ...`
   - Artifact: `results/easyedit_official/current/random_50_seed_42_wal_memit*.json` + markdown report.
   - Evidence update files: `PATH_B_MAX_COMPLETION_MATRIX.md`, `PATH_B_MAX_EXECUTION_CHECKLIST.md`.
2. Backend quality matrix with `dual_row`, `wal_rome`, `wal_memit`, `side_slot`.
   - Command: `python -m agim.eval.easyedit_official_runner --compare-backends dual_row,wal_rome,wal_memit,side_slot ...`
   - Artifact: shared dataset run + one matrix report per backend family in `results/easyedit_official/ablations/`.
3. External output chain coverage.
   - Ripple: `ripple_diagnostic`/`ripple output run`.
   - MQuAKE: `mquake_output_runner` + scorer pipeline.
   - Raw-text: `raw_text_edit_pipeline` + scoring.
   - Product-like: `product_diagnostic` (ScEdit / KnowEdit / UniEdit / MLaKE selection).
   - Artifact: each run must produce both `*.json` model output and `*.md` scored report in `results/external_benchmark_runs/`.
4. Sequential hardening in side-slot.
   - Command family: `--sequential-edit --edit-backend side_slot --n 10/50/100 --seed 42/43/44`
   - Required outputs: `seed-coverage`, retention table, failures split by family.
5. PatchService/governance proof packet.
   - Artifact: one auditable packet with `propose → simulate → run_canaries → approve → apply → rollback → inspect → diff`.
   - Evidence: public claim table + docs update + immutable operation trail.

## Prompt-to-artifact mapping by requirement groups

### A. Truth source and evidence hygiene

- 1) Source-of-truth split across docs and status.
  - `CURRENT_STATUS.md`, `BENCHMARK.md`, `docs/CLAIMS_AND_EVIDENCE.md`, `docs/PATH_B_COMPLETION_AUDIT.md`.
- 2) Path A vs Path B split in docs and quick-start.
  - `README.md`, `docs/PATH_B_WEIGHT_EDITING.md`, `docs/PATH_B_PRODUCTIZATION_PLAN.md`.
- 3) 3-track verification.
  - `docs/VERIFY_PATH_A.md`, `docs/VERIFY_PATH_B_CURRENT.md`, `docs/VERIFY_PATH_B_LEGACY.md`.
- 4) Legacy marking for `WALWeight/ROME` substrate.
  - `docs/PATH_B_COMPLETION_AUDIT.md`, `results/local_protocol/README.md`.
- 5) Method profiles and `method_profile_id` in metadata.
  - `src/agim/eval/easyedit_run_metadata.py`, `src/agim/eval/easyedit_official_runner.py`.
- 6) `artifact_schema_version`.
  - `src/agim/eval/easyedit_run_metadata.py`.

### B. Current single-edit and ablation stack

- 7) Machine-default cleanup in public path.
  - `docs/VERIFY_PATH_B_CURRENT.md`, `src/agim/eval/easyedit_cli.py`.
- 8) Reproducibility bundle.
  - `docs/VERIFY_PATH_B_CURRENT.md`, `results/easyedit_official/current/`, `results/easyedit_official/sequential/`.
- 9) `base_model_digest`, `atoms_digest`.
  - `src/agim/eval/easyedit_payload.py`, `src/agim/eval/easyedit_run_metadata.py`, `src/agim/model/patch_artifact.py`.
- 10) Durable `PatchArtifact` + reload flow.
  - `src/agim/model/patch_artifact.py`, `tests/test_patch_artifact.py`.
- 11) Patch cycle integration (`save → reload → apply → rollback`).
  - `tests/test_patch_service.py`, `tests/test_patch_artifact.py`.
- 12) Metric-family failure split.
  - `src/agim/eval/easyedit_records.py`, `src/agim/eval/easyedit_official_runner.py`.
- 13) `random_200` and `random_1000` seed evidence.
  - `results/easyedit_official/current/random_200_report_2026-05-18.md`,
    `results/easyedit_official/current/random_1000_report_2026-05-18.md`.
- 14) Token-mode matrix.
  - `results/easyedit_official/ablations/token_mode_matrix_report_2026-05-18.md`.
- 15) Component ablations.
  - `results/easyedit_official/ablations/component_ablation_report_2026-05-18.md`.
- 16) Exact-additive ablation.
  - `results/easyedit_official/ablations/exact_additive_report_2026-05-18.md`.
- 17) Deterministic NT snapshot handling.
  - `src/agim/eval/easyedit_metrics.py`, `src/agim/eval/easyedit_eval_loop.py`.
- 18) Norm growth and reuse metrics.
  - `results/easyedit_official/current/easyedit_agim_status_2026-05-18.md`.
- 19) EOS removal from default persistence.
  - `src/agim/eval/easyedit_presets.py`.
- 20) Global anti-repetition hardening.
  - Ablation and replacement policy in edit application path.

### C. Interference and locality controls

- 21) `both` primary sequence selection fixed.
  - `src/agim/model/wal_dual_editor.py`, `results/easyedit_official/ablations/both_primary_fixed_report_2026-05-18.md`.
- 22) Constrained `K_pos/K_neg` solution.
  - `src/agim/model/wal_dual_editor.py` + dedicated constrained-solver experiments.
- 23) Relation-protected banking.
  - `results/easyedit_official/sequential/relation_protected_bank_report_2026-05-18.md`.
- 24) ENCORE-style budget and early stop.
  - `results/easyedit_official/ablations/norm_budget_no_commit_probe_2026-05-18.md`.
- 25) Conflict detector.
  - `src/agim/model/*` (conflict summary hooks).
- 26) Runtime sparse overlay.
  - `src/agim/model/*` overlay API and patch hooks.
- 27) Namespace/session isolation.
  - `--state-namespace`, mutable-state API.
- 28) Side-slot baseline.
  - `results/easyedit_official/sequential/side_slot_random_50_report_2026-05-18.md`.
- 29) Relation sharding for side slots.
  - `src/agim/model`, `src/agim/eval/easyedit_side_slot_loop.py`.
- 30) Side-slot retention schedule.
  - Sequential runs in `results/easyedit_official/sequential/`.

### D. Backend and external productization

- 31) `wal_rome` n=50 quality baseline.
  - `src/agim/model/wal_rome_editor.py`, `results/easyedit_official/ablations/wal_rome_smoke_5_report_2026-05-18.md`.
- 32) `wal_memit` consolidation + quality path.
  - `src/agim/model/wal_memit_batch_editor.py`, `src/agim/model/wal_memit_editor.py`.
- 33) Full backend matrix in official format.
  - `src/agim/eval/easyedit_backend_matrix.py`, `results/easyedit_official/ablations/*backend_matrix*.md`.
- 34) RippleEdits consequence chain.
  - `src/agim/eval/ripple_diagnostic.py` + external run payload/score.
- 35) MQuAKE consequence chain.
  - `src/agim/eval/mquake_output_runner.py` + scored output.
- 36) Raw-text chain.
  - `src/agim/eval/raw_text_edit_pipeline.py`, `src/agim/eval/raw_text_scoring.py`.
- 37) Product-facing benchmark chain.
  - `src/agim/eval/product_diagnostic.py` and corresponding benchmark artifacts.
- 38) PatchService public lifecycle APIs.
  - `src/agim/model/patch_service.py`, docs references in `docs/EASYEDIT_ADAPTER.md`.
- 39) Governance and tenant-safe signatures.
  - `src/agim/model/patch_governance.py`, `tests/test_patch_governance.py`.
- 40) External adapter and proof packet.
  - `src/agim/integrations/easyedit_agimwal.py`, `docs/EASYEDIT_ADAPTER.md`, and claim boundary docs.

## Execution cadence

- S1: close hard-gates 1 and 2.
- S2: close hard-gates 3 and 4.
- S3: close hard-gate 5 and harden governance/protocol claims.
- S4: re-audit all 40 items for final completion matrix.

## Current top priorities now

1. `wal_memit` baseline official bundle for n=50 with failure family summaries.
2. Full backend matrix with `side_slot` included.
3. One tracked Ripple + one tracked MQuAKE + one tracked raw-text + one product-like benchmark.
4. Side-slot retention matrix across 10/50/100 and seeds 42/43/44.
5. Publish proof packet for PatchService and governance trail.
