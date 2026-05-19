# Path B Prompt-to-Artifact Audit Matrix (Maximal Plan v1.0)

Source objective: `/mnt/hf_model_weights/arman/3bit/sites/deep-research-report (5).md`  
Date: 2026-05-19  
Scope: all Path B hardening points, including 40 roadmap requirements, hard gates, commands, and evidence chains.

## Success criteria

Path B is complete only when every row below is `Done` with concrete artifact evidence and, where applicable, closed by executed hard-gate commands.

## Maximal execution map (Requirements 1..40)

| # | Requirement | Command / artifact | Evidence | Status |
| --- | --- | --- | --- | --- |
| 1 | Source-of-truth split between Path A / Path B | `CURRENT_STATUS.md`, `BENCHMARK.md`, `docs/CLAIMS_AND_EVIDENCE.md`, `docs/PATH_B_COMPLETION_AUDIT.md` | `CURRENT_STATUS.md`, `BENCHMARK.md`, `docs/CLAIMS_AND_EVIDENCE.md`, `docs/PATH_B_COMPLETION_AUDIT.md` | Done |
| 2 | Separate Path A and Path B docs + quick-start split | `README.md` + `docs/PATH_B_WEIGHT_EDITING.md` + `docs/PATH_B_PRODUCTIZATION_PLAN.md` | `README.md`, `docs/PATH_B_WEIGHT_EDITING.md`, `docs/PATH_B_PRODUCTIZATION_PLAN.md` | Done |
| 3 | Three-track verify docs | `docs/VERIFY_PATH_A.md`, `docs/VERIFY_PATH_B_CURRENT.md`, `docs/VERIFY_PATH_B_LEGACY.md` | `docs/VERIFY_PATH_A.md`, `docs/VERIFY_PATH_B_CURRENT.md`, `docs/VERIFY_PATH_B_LEGACY.md` | Done |
| 4 | Legacy `WALWeight/ROME` marking as historical | Dedicated legacy docs/paths and explicit markers | `docs/PATH_B_COMPLETION_AUDIT.md`, `results/local_protocol/` | Partially Done |
| 5 | Method profile IDs in protocol | `--method-profile-id` in official runner payload | `src/agim/eval/easyedit_run_metadata.py`, `src/agim/eval/easyedit_official_runner.py` | Done |
| 6 | `artifact_schema_version` | Metadata schema field in payload | `src/agim/eval/easyedit_run_metadata.py` | Done |
| 7 | Machine-specific defaults removed from public guidance | env-driven examples for model/device/root/local-files-only | `docs/VERIFY_PATH_B_CURRENT.md`, `README.md`, `src/agim/eval/easyedit_cli.py` | Done |
| 8 | Reproducible bundle for current/sequential runs | single entry docs + result folders | `docs/VERIFY_PATH_B_CURRENT.md`, `results/easyedit_official/current/`, `results/easyedit_official/sequential/` | Done |
| 9 | `base_model_digest` + `atoms_digest` in payload | required metadata fields | `src/agim/eval/easyedit_payload.py`, `src/agim/eval/easyedit_run_metadata.py`, `src/agim/model/patch_artifact.py` | Partially Done |
| 10 | Durable, reloadable PatchArtifact | save/load/inspect contract | `src/agim/model/patch_artifact.py`, `tests/test_patch_artifact.py` | Partially Done |
| 11 | Patch cycle save → reload → apply → rollback | test for persistence + parity behavior | `tests/test_patch_service.py`, `tests/test_patch_artifact.py` | Partially Done |
| 12 | Metric-family `failures_only` selector | `tf/ctx_gen/prob/vanilla_gen` split | `src/agim/eval/easyedit_cli.py`, `src/agim/eval/easyedit_run_metadata.py`, `src/agim/eval/easyedit_failures.py`, `src/agim/eval/easyedit_official_runner.py`, `docs/VERIFY_PATH_B_CURRENT.md` | Done |
| 13 | Seeded random_200 coverage | official random_200 bundle with reports | `results/easyedit_official/current/random_200_report_2026-05-18.md` | Done |
| 14 | Seeded random_1000 coverage | first/random/seeded stability | `results/easyedit_official/current/random_1000_report_2026-05-18.md` | Done |
| 15 | `target_token_mode` matrix | standalone/contextual/both comparison | `results/easyedit_official/ablations/token_mode_matrix_report_2026-05-18.md` | Done |
| 16 | Component ablations | lm_head/embed/dual/no_eos/no_anti combinations | `results/easyedit_official/ablations/component_ablation_report_2026-05-18.md` | Done |
| 17 | Exact-additive ablation | official-compatible dual comparison | `results/easyedit_official/ablations/exact_additive_report_2026-05-18.md` | Done |
| 18 | Deterministic NT snapshot mode | fixed NT sample + snapshot reproducibility | `src/agim/eval/easyedit_metrics.py`, `src/agim/eval/easyedit_eval_loop.py` | Partially Done |
| 19 | Norm growth + reuse monitoring | patched row growth / cumulative norm trend | `results/easyedit_official/current/easyedit_agim_status_2026-05-18.md` | Partially Done |
| 20 | Default EOS handling explicit | non-eos default persisted in profile | `src/agim/eval/easyedit_presets.py`, `results/easyedit_official/ablations/eos_default_report_2026-05-18.md` | Done |
| 21 | Anti-repetition from global shared rows removed from defaults | de-globalized anti strategy by default (`clamp_anti=0`, `clamp_anti_scope=none`) | `src/agim/eval/easyedit_cli.py`, `src/agim/model/wal_dual_editor.py` | Done |
| 22 | `target_token_mode="both"` primary selection fix | deterministic continuation-aligned logic | `src/agim/model/wal_dual_editor.py`, `src/agim/eval/easyedit_eval_loop.py`, ablation report | Partially Done (design evidence) |
| 23 | Constrained `K_pos/K_neg` solve | constrained objective for update | `src/agim/model/wal_dual_editor.py`, `src/agim/eval/easyedit_cli.py`, `tests/test_easyedit_artifacts.py`, `tests/test_easyedit_official_metrics.py` | Done (synthetic internal/public proof) |
| 24 | Relation-protected banks | protected relation buckets / failure pool | `results/easyedit_official/sequential/relation_protected_bank_report_2026-05-18.md` | Done (synthetic internal/public proof) |
| 25 | ENCORE-like budgets + early stop | norm budget and no-commit guard | `results/easyedit_official/ablations/norm_budget_no_commit_probe_2026-05-18.md`, `src/agim/model/patch_artifact.py`, `src/agim/model/patch_service.py`, `scripts/run_path_b_max_patch_service_governance_proof.py` | Done (synthetic internal/public proof) |
| 26 | Patch conflict detector | subject/target/EOS/control overlap checks | `src/agim/model/wal_dual_editor.py`, `src/agim/eval/easyedit_budget.py`, `src/agim/model/patch_artifact.py`, `src/agim/model/patch_service.py` | Done (synthetic internal/public proof) |
| 27 | Runtime sparse overlay | overlay apply path for lm_head/embed | `src/agim/model` overlay/runtime hooks | Partially Done |
| 28 | Namespace/session mutable state isolation | `--state-namespace` and namespace-scoped mutable state | `src/agim/model`, eval glue | Partially Done |
| 29 | Side-slot baseline with sequential evidence | side-slot n=50 baseline artifact | `results/easyedit_official/sequential/side_slot_random_50_report_2026-05-18.md` | Done |
| 30 | Side-slot relation sharding | slot allocator with relation-aware isolation | `src/agim/model`, `src/agim/eval/easyedit_side_slot_loop.py` | Done (synthetic internal/public proof) |
| 31 | Side-slot retention matrix 10/50/100, seeds 42/43/44 | explicit command runs | `results/easyedit_official/sequential/` (all n/seeds + outputs) | Done |
| 32 | `WALRomeEditor` FFN backend | implementation + n=50 quality baseline | `src/agim/model/wal_rome_editor.py`, `results/easyedit_official/ablations/wal_rome_smoke_5_report_2026-05-18.md` | Partially Done |
| 33 | `WALMemitBatchEditor` backend and consolidation path | editor module + compatibility path | `src/agim/model/wal_memit_batch_editor.py`, `src/agim/model/wal_memit_editor.py` | Done (synthetic internal/public proof) |
| 34 | Backend comparison matrix (`dual_row`, `wal_rome`, `wal_memit`, `side_slot`) | `easyedit_official_runner --compare-backends` execution | `src/agim/eval/easyedit_backend_matrix.py`, matrix artifacts | Done |
| 35 | RippleEdits consequence chain | output JSON + scored report | `src/agim/eval/ripple_diagnostic.py` and external benchmark adapter outputs | Done |
| 36 | MQuAKE consequence chain | output JSON + scored report | `src/agim/eval/mquake_output_runner.py`, `src/agim/eval/mquake_diagnostic.py`, `results/external_benchmark_runs/mquake_wal_memit_n50_seed42_outputs.json`, `results/external_benchmark_runs/mquake_wal_memit_n50_seed42_scored.json` | Done |
| 37 | AKEW-style raw-text chain | raw-text adapter + scoring | `src/agim/eval/raw_text_edit_pipeline.py`, `src/agim/eval/raw_text_scoring.py` | Done |
| 38 | Product benchmark chain (ScEdit / KnowEdit / UniEdit / MLaKE) | at least one model-output benchmark chain | `src/agim/eval/product_diagnostic.py` | Done |
| 39 | PatchService API lifecycle | propose/simulate/run_canaries/approve/apply/rollback/inspect/diff | `src/agim/model/patch_service.py`, `src/agim/model/patch_governance.py` | Done (synthetic internal/public proof) |
| 40 | Governance + external adapter package | signature, approvals, audit trail, package boundary | `src/agim/model/patch_governance.py`, `src/agim/integrations/easyedit_agimwal.py`, `docs/EASYEDIT_ADAPTER.md`, `scripts/run_path_b_max_gate5_*.py` | Done (synthetic internal/public proof) |

## Hard gates execution mapping

Recommended hard-gate run:

`bash scripts/run_path_b_max_bootstrap.sh 1 2 4 3 5`

1. `wal_memit` official n=50 baseline  
   - Source: `docs/PATH_B_MAX_HARDGATE_QUEUE.md` gate 1  
   - Expected artifacts: `results/easyedit_official/current/random_50_seed_42_wal_memit.json`, `results/easyedit_official/current/random_50_seed_42_wal_memit.failures.json`  

2. Backend matrix (`dual_row`, `wal_rome`, `wal_memit`, `side_slot`)  
   - Source: `docs/PATH_B_MAX_HARDGATE_QUEUE.md` gate 2  
   - Expected artifacts: `results/easyedit_official/ablations/backend_matrix_random_50_report_2026-05-18.md` + 4 per-backend JSON files  

3. External consequence chain  
   - Source: `docs/PATH_B_MAX_HARDGATE_QUEUE.md` gate 3  
   - Required artifacts: `results/external_benchmark_runs/*` for Ripple, MQuAKE, raw-text, product  

4. Side-slot retention hardening  
   - Source: `docs/PATH_B_MAX_HARDGATE_QUEUE.md` gate 4  
   - Required artifacts: runs for n=10/50/100 for seeds 42/43/44  

5. Governance proof packet  
   - Source: `docs/PATH_B_MAX_HARDGATE_QUEUE.md` gate 5  
   - Required artifacts: `docs/CLAIMS_AND_EVIDENCE.md`, `docs/PATH_B_COMPLETION_AUDIT.md`, public lifecycle docs + audit trail

## Immediate completion order

1. Gate 5 (public external proof packet and immutable API smoke) — требуется real provider + immutable public URI.
2. req. 23/24/25/26/30/33/39/40 подтверждены на synthetic internal/public proof; закрыть финальную external proof-boundary для Gate 5.

## Примечание по evidence-классификации

Для учета completion все required артефакты должны быть в формате:
- `backend_matrix_random_50_seed42.json`
- `backend_matrix_random_50_seed42.dual_row.json`
- `backend_matrix_random_50_seed42.wal_rome.json`
- `backend_matrix_random_50_seed42.wal_memit.json`
- `backend_matrix_random_50_seed42.side_slot.json`

Найденные на диске `backend_matrix_random_50_seed42_dual_walrome*.json`, `backend_matrix_smoke_5*.json`, и прочие legacy/переходные файлы полезны для анализа, но не закрывают hard gate 2 как официальные completion артефакты.
