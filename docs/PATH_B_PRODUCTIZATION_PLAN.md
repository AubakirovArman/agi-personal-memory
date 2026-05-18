# Path B Productization Plan

Source: `/mnt/hf_model_weights/arman/3bit/sites/deep-research-report (5).md`

This is the maximum Path B plan. It turns AGIM WAL from a lab-only dual-row
editor into an audited model-hotfix product line.

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
| In progress | Partially implemented; usable but incomplete |
| Queued | Planned and documented; not implemented |

## Roadmap Checklist

| # | Work item | Status | Evidence / next artifact |
| ---: | --- | --- | --- |
| 1 | Remove source-of-truth drift | In progress | `CURRENT_STATUS.md`, `BENCHMARK.md`, docs updated after n=1000 |
| 2 | Split README quick starts for Path A and Path B | In progress | `README.md` |
| 3 | Split verification docs into Path A, Path B current, Path B legacy | In progress | `docs/VERIFY_PATH_A.md`, `docs/VERIFY_PATH_B_CURRENT.md`, `docs/VERIFY_PATH_B_LEGACY.md` |
| 4 | Mark heavy `WALWeightEditor` / `ROMEEditor` E2E tests as substrate tests | Queued | pytest markers/docs |
| 5 | Add formal `method_profile_id` | Done | `easyedit_run_metadata.py`, payload metadata |
| 6 | Add `artifact_schema_version` | Done | `easyedit_run_metadata.py`, payload metadata |
| 7 | Move machine-specific defaults out of public docs | In progress | verify docs use placeholders/env vars |
| 8 | Add reproducibility command bundle | In progress | `docs/VERIFY_PATH_B_CURRENT.md` |
| 9 | Add `base_model_digest` and `atoms_digest` | Done | new EasyEdit payload fields |
| 10 | Add serializable Path B `PatchArtifact` | In progress | `src/agim/model/patch_artifact.py` |
| 11 | Add reload-safe patch integration test | Queued | future model-level test |
| 12 | Make failures-only metric families configurable | Done | `--failure-families`; default excludes `vanilla_gen` |
| 13 | Run random-200 seeds 42/43/44 | Queued | future GPU artifacts |
| 14 | Run random-1000 seed 42 | Queued | future GPU artifact |
| 15 | Run `standalone/contextual/both` token-mode matrix | Queued | future GPU matrix report |
| 16 | Run component ablations | Queued | `lm_head-only`, `embed-only`, `dual`, no-EOS/no-anti |
| 17 | Add exact-additive update ablation | Queued | compare WAL quantization vs editing locus |
| 18 | Add deterministic NT mode | Queued | store sampled row ids in artifacts |
| 19 | Collect norm-growth metrics | In progress | `PatchArtifact.norm_summary()` foundation |
| 20 | Remove global EOS row from persistent default | Queued | profile change after ablation |
| 21 | Remove anti-repetition from global rows | Queued | decode-time or patch-local policy |
| 22 | Fix `target_token_mode=both` primary sequence selection | Queued | runner/editor change |
| 23 | Replace positive-key averaging with constrained solve | Queued | `K_pos/K_neg` solve |
| 24 | Add relation-specific protected banks | Queued | relation failure artifacts |
| 25 | Add ENCORE-style norm budgets and early stop | Queued | patch no-commit policy |
| 26 | Add patch conflict detector | In progress | `conflict_summary()` foundation |
| 27 | Add runtime sparse overlay mode | Queued | frozen base plus row overlays |
| 28 | Namespace mutable history state | Queued | tenant/batch state |
| 29 | Build true side-slot memory | Queued | routed sequential serving |
| 30 | Lift relation sharding to patch-slot isolation | Queued | beyond projection-only relation mode |
| 31 | Re-run sequential retention in side-slot mode | Queued | future GPU artifacts |
| 32 | Add `WALRomeEditor` backend | Queued | located FFN patch backend |
| 33 | Add `WALMemitBatchEditor` backend | Queued | batch consolidation backend |
| 34 | Let runner compare multiple backends | Queued | `dual_row`, `wal_rome`, `wal_memit`, `side_slot` |
| 35 | Add RippleEdits diagnostic benchmark | Queued | related-fact effects |
| 36 | Add MQuAKE benchmark | Queued | multi-hop consequences |
| 37 | Add AKEW-style raw-text edit pipeline | Queued | text update to patch proposal |
| 38 | Add product-facing benchmark | Queued | ScEdit, UniEdit/KnowEdit, or MLaKE |
| 39 | Build `PatchService API` | Queued | propose/simulate/canary/approve/apply/rollback |
| 40 | Add governance layer and external adapter package | Queued | signatures, ACL, audit trail, EasyEdit adapter |

## Beta Gate

Path B can be described as beta-product ready when all of these are true:

- durable patch artifacts exist;
- reload-safe rollback exists;
- `single_loc` is stable on random-50, random-200, and first-1000;
- at least one stronger backend exists beyond `dual_row`;
- MQuAKE/Ripple/AKEW-style diagnostics exist;
- service-layer approvals and audit exist.
