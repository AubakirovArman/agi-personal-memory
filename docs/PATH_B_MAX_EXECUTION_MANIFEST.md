# Path B Maximal Execution Manifest

Источник требований: `/mnt/hf_model_weights/arman/3bit/sites/deep-research-report (5).md`  
Формальная трассировка: `docs/PATH_B_MAX_MAX_PLAN.md`, `docs/PATH_B_MAX_MAX_AUDIT.md`, `docs/PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md`

## Acceptance criteria

- Все 40 требований из deep-research source закрыты по `Done` только при наличии явного артефакта.
- Все 5 hard-gates закрыты with canonical JSON/MD payloads and command outputs.
- `README/CURRENT_STATUS/BENCHMARK/CLAIMS_AND_EVIDENCE` отражают только официальные EasyEdit-compatible claims.
- Legacy `local_protocol` и debug-скрипты не считаются completion proof.

## Hard gates (non-negotiable)

1. `wal_memit` официальная n=50 baseline
- Command target: `scripts/run_path_b_max_bootstrap.sh 1` (или эквивалентный `python -m agim.eval.easyedit_official_runner --edit-backend wal_memit ...`)
- Required artifacts:
  - `results/easyedit_official/current/random_50_seed_42_wal_memit.json`
  - `results/easyedit_official/current/random_50_seed_42_wal_memit.failures.json`
  - related summary md

2. Полный backend matrix 4 backends
- Command target: `scripts/run_path_b_max_bootstrap.sh 2`
- Required artifacts:
  - `results/easyedit_official/ablations/backend_matrix_random_50_seed42.json`
  - `...backend_matrix_random_50_seed42.dual_row.json`
  - `...backend_matrix_random_50_seed42.wal_rome.json`
  - `...backend_matrix_random_50_seed42.wal_memit.json`
  - `...backend_matrix_random_50_seed42.side_slot.json`
  - `...backend_matrix_random_50_report_2026-05-18.md`

3. External consequence chain
- Command target: `scripts/run_path_b_max_bootstrap.sh 3`
- Required artifacts:
  - `results/external_benchmark_runs/ripple_wal_memit_n50_seed42.json`
  - `results/external_benchmark_runs/mquake_wal_memit_n50_seed42_outputs.json`
  - `results/external_benchmark_runs/mquake_wal_memit_n50_seed42_scored.json`
  - `results/external_benchmark_runs/raw_text_wal_memit_n50_seed42.json`
  - `results/external_benchmark_runs/product_scedit_wal_memit_n50_seed42.json`

4. Side-slot retention hardening
- Command target: `scripts/run_path_b_max_bootstrap.sh 4`
- Required artifacts:
  - `results/easyedit_official/sequential/side_slot_random_10_seed_{42,43,44}_seq.json`
  - `results/easyedit_official/sequential/side_slot_random_10_seed_{42,43,44}_seq.failures.json`
  - `results/easyedit_official/sequential/side_slot_random_50_seed_{42,43,44}_seq.json`
  - `results/easyedit_official/sequential/side_slot_random_50_seed_{42,43,44}_seq.failures.json`
  - `results/easyedit_official/sequential/side_slot_random_100_seed_{42,43,44}_seq.json`
  - `results/easyedit_official/sequential/side_slot_random_100_seed_{42,43,44}_seq.failures.json`

5. PatchService / governance proof packet
- Command target: `scripts/run_path_b_max_bootstrap.sh 5`
- Required evidence:
  - claims-safe public lifecycle for `propose/simulate/run_canaries/approve/apply/rollback/inspect/diff`
  - signatures + immutable audit chain for one concrete patch path
  - claims/docs update removing unsafe Path B leaderboard statements

## Canonical execution order

```bash
cd /mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory
export AGIM_MODEL="meta-llama/Llama-3.1-8B-Instruct"
export AGIM_DEVICE="cuda:0"
export AGIM_LOCAL_FILES_ONLY=0
export AGIM_RAW_TEXT_INPUT="data/raw_text_updates.jsonl"
export AGIM_EASYEDIT_ROOT="<YOUR_EASYEDIT_REPO_PATH>"
export AGIM_MQUAKE_ADAPTER="<YOUR_MQUAKE_ADAPTER_PATH>"
bash scripts/run_path_b_max_bootstrap.sh 1 2 4 3 5
```

## Evidence update after each gate

- `docs/PATH_B_MAX_COMPLETION_MATRIX.md`
- `docs/PATH_B_MAX_EXECUTION_CHECKLIST.md`
- `docs/PATH_B_MAX_STATUS_BOARD.md`
- `docs/PATH_B_MAX_PROMPT_ARTIFACT_AUDIT_MATRIX.md`
- `docs/PATH_B_MAX_PROMPT_ARTIFACT_AUDIT.md`
- `docs/PATH_B_COMPLETION_AUDIT.md`

## Current residual gap set

- Gate 1 baseline and Gate 2 matrix artifacts are done.
- Gate 3 external chain is completed through 4 channels (Ripple/MQuAKE/raw-text/product), score payload closure remains evidence-audited in-flight.
- Gate 4 side-slot sequence artifacts are done.
- Gate 5 synthetic internal/public proof path is done (`example.com/api`, `object_lock`, `AGIM_GATE5_PUBLIC_API_SMOKE=1`);
  remaining step is real external immutable provider/provider-level verification in production.
- Open non-gate technical backlog is reduced to technical follow-ups; req. 23/24/25/26/30/33/39/40 are synthetic proof-closed.
