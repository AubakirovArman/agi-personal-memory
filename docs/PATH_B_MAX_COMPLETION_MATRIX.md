# Path B Completion Matrix (Prompt-to-Artifact Mapping)

Дата: 2026-05-18  
Источник: `/mnt/hf_model_weights/arman/3bit/sites/deep-research-report (5).md`

## Критерий завершения

Path B считается завершённым по максимальному плану только когда:

- все hard-гейты закрыты с официальными артефактами;
- каждый пункт требований имеет traceable evidence;
- никаких новых claims не оставлены без явной маркировки safety boundaries.

## Макроскопический статус

- hard-гейты: 5 из 5 не закрыты полностью (только partially/частично).
- 40 requirements: многие закрыты частично, часть остаётся open.

Reference:
- [PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md)

## Hard-gate map

| # | Requirement | Evidence target | Status |
| --- | --- | --- | --- |
| 1 | `wal_memit` official n=50 baseline | `results/easyedit_official/current/random_50_seed_42_wal_memit*.json`, `*md` | Not complete |
| 2 | Full backend comparison (`dual_row`, `wal_rome`, `wal_memit`, `side_slot`) | `results/easyedit_official/ablations/*backend_matrix*.md` | Partial |
| 3 | External model-output evidence (Ripple/MQuAKE/raw-text/product) | `results/external_benchmark_runs/*` | Partial |
| 4 | Side-slot retention 10/50/100, seeds 42/43/44 | `results/easyedit_official/sequential/` | Not complete |
| 5 | Public PatchService/governance proof packet | docs + API contract + audit trail artifacts | Partial |

## Prompt-to-Artifacts checklist (key items)

| Prompt/item | Concrete command or artifact | Status |
| --- | --- | --- |
| Source-of-truth split | `CURRENT_STATUS.md`, `BENCHMARK.md`, `docs/CLAIMS_AND_EVIDENCE.md`, `docs/PATH_B_COMPLETION_AUDIT.md` | Done |
| Path A / Path B split | `README.md`, `docs/PATH_B_WEIGHT_EDITING.md`, `docs/PATH_B_PRODUCTIZATION_PLAN.md` | Done |
| 3-track verify docs | `docs/VERIFY_PATH_A.md`, `docs/VERIFY_PATH_B_CURRENT.md`, `docs/VERIFY_PATH_B_LEGACY.md` | Done |
| Legacy marking (`WALWeight/ROME` legacy) | `docs/PATH_B_COMPLETION_AUDIT.md`, `results/local_protocol/` | Partial |
| Method profiles + schema | `src/agim/eval/easyedit_run_metadata.py`, `easyedit_official_runner.py`, payload fields | Done |
| Machine-default cleanup (docs/tests) | `docs/VERIFY_PATH_B_CURRENT.md`, `src/agim/eval/easyedit_cli.py` | Done |
| Repro bundle commands | `docs/VERIFY_PATH_B_CURRENT.md`, `results/easyedit_official/current/`, `results/easyedit_official/sequential/` | Done |
| `base_model_digest`, `atoms_digest` | `src/agim/eval/easyedit_payload.py`, `easyedit_run_metadata.py`, `patch_artifact.py` | Partial |
| Durable PatchArtifact + reload safe loop | `src/agim/model/patch_artifact.py`, `tests/test_patch_artifact.py`, `tests/test_patch_service.py` | Partial |
| Failure-family selector | `src/agim/eval/easyedit_records.py`, `easyedit_official_runner.py` | Partial |
| Random coverage: n=200 and n=1000 | `results/easyedit_official/current/random_200_*`, `results/easyedit_official/current/random_1000_*` | Done |
| token_mode matrix | `results/easyedit_official/ablations/token_mode_matrix_report_2026-05-18.md` | Done |
| component / exact ablations | `results/easyedit_official/ablations/*_ablation_report_2026-05-18.md` | Done |
| Deterministic NT growth/reuse | `src/agim/eval/easyedit_metrics.py`, `easyedit_eval_loop.py`, status summary | Partial |
| Global EOS removal default | `src/agim/eval/easyedit_presets.py`, `results/easyedit_official/ablations/eos_default_report_2026-05-18.md` | Done |
| Anti-repetition global cleanup | `src/agim/eval/easyedit_presets.py`, ablations | Not started |
| both-primary fix + dedicated baseline | `results/easyedit_official/ablations/both_primary_fixed_report_2026-05-18.md` | Partial |
| K_pos/K_neg solve | `src/agim/model/wal_dual_editor.py` | Not started |
| relation-protected banks and conflict controls | `results/easyedit_official/sequential/relation_protected_bank_report_2026-05-18.md`, `conflict_summary` | Partial |
| ENCORE-style budgets / early stop | `results/easyedit_official/ablations/norm_budget_no_commit_probe_2026-05-18.md` | Partial |
| Runtime sparse overlay | overlay API under `src/agim/model/`, runtime hooks | Partial |
| Namespace/session isolation | `src/agim/model` + CLI `--state-namespace` | Partial/Done-in-parts |
| Side-slot working baseline | `results/easyedit_official/sequential/side_slot_random_50_report_2026-05-18.md` | In progress |
| Relation slot sharding in slots | `src/agim/model`, `easyedit_side_slot_loop.py` | Partial |
| `wal_rome` n=50 quality baseline | `src/agim/model/wal_rome_editor.py`, `results/easyedit_official/ablations/wal_rome_smoke_5_report_2026-05-18.md` | Partial |
| `wal_memit` batch backend and n=50 quality path | `src/agim/model/wal_memit_batch_editor.py`, `src/agim/model/wal_memit_editor.py` | Partial |
| External consequence chain (RippleEdits) | `src/agim/eval/ripple_diagnostic.py`, external output runners | Partial |
| External consequence chain (MQuAKE) | `src/agim/eval/mquake_output_runner.py`, `results/external_benchmark_adapters/*`, model-output run | Partial |
| External chain (raw-text) | `src/agim/eval/raw_text_edit_pipeline.py`, `raw_text_scoring.py` | Partial |
| Product benchmark chain | `src/agim/eval/product_diagnostic.py`, `results/other_benchmarks/` | Not started |
| PatchService API endpoints | `src/agim/model/patch_service.py`, `patch_governance.py` | Partial |
| Public governance + adapter package | `src/agim/integrations/easyedit_agimwal.py`, `docs/EASYEDIT_ADAPTER.md` | In progress |

## Следующий шаг после анализа

1. Закрыть Gate 1 и Gate 2, чтобы дать hard evidence для backend parity.
2. Параллельно закрыть Gate 3 (Ripple/MQuAKE/AKEW/product) с tracked output chain.
3. Закрыть Gate 4 (side-slot retention 10/50/100 + seeds).
4. Закрыть Gate 5 (proof packet).
