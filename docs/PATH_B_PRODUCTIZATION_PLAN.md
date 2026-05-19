# Path B Productization Plan

Source: `/mnt/hf_model_weights/arman/3bit/sites/deep-research-report (5).md`

This is the maximum Path B plan. It turns AGIM WAL from a lab-only dual-row
editor into an audited model-hotfix product line.

Execution authority lives in `docs/PATH_B_MAX_EXECUTION_CHECKLIST.md`. Use status
there for completion gates; this file tracks directional milestones.

## Operating Principle

Path B should be positioned as audited model hotfixing first, not solved
lifelong parametric memory. WAL remains the control plane: provenance,
rollback, patch lifecycle, canaries, and reproducibility. Editing backends can
evolve from `dual_row` to side slots, located internal-layer patches, and batch
consolidation.

## Status Legend

| Status | Meaning |
| --- | --- |
| Done | Implemented and backed by tracked files or artifacts |
| Partially Done | Implemented in part, but one or more required gates/artifacts are still missing |
| In progress | Partially implemented; usable but incomplete |
| Queued | Planned and documented; not implemented |

## Roadmap Checklist

| # | Work item | Status | Evidence / next artifact |
| ---: | --- | --- | --- |
| 1 | Remove source-of-truth drift | Done | `CURRENT_STATUS.md`, `BENCHMARK.md`, docs updated after n=1000 |
| 2 | Split README quick starts for Path A and Path B | Done | `README.md` |
| 3 | Split verification docs into Path A, Path B current, Path B legacy | Done | `docs/VERIFY_PATH_A.md`, `docs/VERIFY_PATH_B_CURRENT.md`, `docs/VERIFY_PATH_B_LEGACY.md` |
| 4 | Mark heavy `WALWeightEditor` / `ROMEEditor` E2E tests as substrate tests | Done | `pytest.mark.substrate`, `docs/VERIFY_PATH_B_LEGACY.md` |
| 5 | Add formal `method_profile_id` | Done | `easyedit_run_metadata.py`, payload metadata |
| 6 | Add `artifact_schema_version` | Done | `easyedit_run_metadata.py`, payload metadata |
| 7 | Move machine-specific defaults out of public docs | Done | verify docs use placeholders/env vars |
| 8 | Add reproducibility command bundle | Done | `docs/VERIFY_PATH_B_CURRENT.md` |
| 9 | Add `base_model_digest` and `atoms_digest` | Partially Done | new EasyEdit payload fields |
| 10 | Add serializable Path B `PatchArtifact` | Partially Done | `src/agim/model/patch_artifact.py`, reload-safe test |
| 11 | Add reload-safe patch integration test | Done | `test_patch_artifact_reload_apply_and_rollback` |
| 12 | Make failures-only metric families configurable | In progress | `--failure-families` declared; selected-family outputs and happy-path tests still incomplete |
| 13 | Run random-200 seeds 42/43/44 | Done | `results/easyedit_official/current/random_200_report_2026-05-18.md` |
| 14 | Run random-1000 seed 42 | Done | `results/easyedit_official/current/random_1000_report_2026-05-18.md` |
| 15 | Run `standalone/contextual/both` token-mode matrix | Done | `results/easyedit_official/ablations/token_mode_matrix_report_2026-05-18.md` |
| 16 | Run component ablations | Done | `results/easyedit_official/ablations/component_ablation_report_2026-05-18.md` |
| 17 | Add exact-additive update ablation | Done | `results/easyedit_official/ablations/exact_additive_report_2026-05-18.md` |
| 18 | Add deterministic NT mode | Partially Done | `NT.*_sampled_row_ids`, `--nt-sample-size`; strict snapshot determinism not required for every workflow yet |
| 19 | Collect norm-growth metrics | Done | `NT.edited_*_delta_l2_*`, `PatchArtifact.norm_summary()` |
| 20 | Remove global EOS row from persistent default | Done | `results/easyedit_official/ablations/eos_default_report_2026-05-18.md` |
| 21 | Remove anti-repetition from global rows | Queued | component matrix shows anti protects locality today |
| 22 | Fix `target_token_mode=both` primary sequence selection | Done | `results/easyedit_official/ablations/both_primary_fixed_report_2026-05-18.md` |
| 23 | Replace positive-key averaging with constrained solve | Not Started | ridge/ablation exists, but constrained solver is not merged to main path |
| 24 | Add relation-specific protected banks | In progress | report exists; worst-relation driven prioritization still incomplete |
| 25 | Add ENCORE-style norm budgets and early stop | In progress | probe run exists; production enforce-path still pending |
| 26 | Add patch conflict detector | In progress | `conflict_summary()` row/metadata risk flags |
| 27 | Add runtime sparse overlay mode | Done | `RuntimeSparseOverlay.add_patch_artifact()` |
| 28 | Namespace mutable history state | Done | `--state-namespace`, `WALDualMutableState` |
| 29 | Build true side-slot memory | Done | EasyEdit `--edit-backend side_slot` |
| 30 | Lift relation sharding to patch-slot isolation | In progress | `SideSlotMemory.relation_slot_summary()` |
| 31 | Re-run sequential retention in side-slot mode | In progress | `results/easyedit_official/sequential/side_slot_random_50_report_2026-05-18.md` |
| 32 | Add `WALRomeEditor` backend | Done | `src/agim/model/wal_rome_editor.py`, `--edit-backend wal_rome` |
| 33 | Add `WALMemitBatchEditor` backend | Done | `src/agim/model/wal_memit_batch_editor.py` |
| 34 | Let runner compare multiple backends | In progress | n=50 direct and sequential matrices; `wal_memit` now in compatibility backend matrix path, n=50 tuning run still pending |
| 35 | Add RippleEdits diagnostic benchmark | In progress | `agim.eval.ripple_diagnostic` post-hoc diagnostic and dataset adapter |
| 36 | Add MQuAKE benchmark | In progress | first-50 MQuAKE adapter plus tracked dual-row model-output run |
| 37 | Add AKEW-style raw-text edit pipeline | In progress | parser, PatchService drafts, service materialization hook, and scored-output layer |
| 38 | Add product-facing benchmark | In progress | `agim.eval.product_diagnostic` KnowEdit-inspired score and dataset adapter |
| 39 | Build `PatchService API` | In progress | `PatchService` implemented in-process, public contract hardening pending |
| 40 | Add governance layer and external adapter package | In progress | `PatchGovernance`, `agim.integrations.easyedit_agimwal`, `docs/EASYEDIT_ADAPTER.md`; full public service packaging not complete |

## Beta Gate

Path B can be described as beta-product ready when all of these are true:

- durable patch artifacts exist;
- reload-safe rollback exists;
- `single_loc` is stable on random-50, random-200, and first-1000;
- at least one stronger backend exists beyond `dual_row`;
- MQuAKE/Ripple/AKEW-style diagnostics exist;
- service-layer approvals and audit exist.
