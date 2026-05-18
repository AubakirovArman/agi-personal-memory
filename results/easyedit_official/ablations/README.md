# EasyEdit-Compatible Ablations

This folder contains tuning and protocol ablation artifacts from the
EasyEdit-compatible runner.

These files are useful for audit and ablation history, but they are not the
current headline result. Use `../current/` for current single-edit claims and
`../sequential/` for sequential claims.

Fresh PS@All ablation:

- `easyedit_official_50_first42_psall_positive_prompts.json`

Readout: positive prompts improve single-edit paraphrase / PS@All, but reduce
locality, so this is not the default method setting.

Fresh token-mode matrix:

- `token_mode_matrix_report_2026-05-18.md`
- `token_mode_random_200_seed_42_{standalone,contextual,both}.json`

Readout: `contextual` is the current correct default for Llama continuation
editing; `standalone` mostly measures a different generation target, and
`both` needs the planned primary-sequence fix before it can be considered as a
default profile.

Fresh component ablation:

- `component_ablation_report_2026-05-18.md`
- `component_random_200_seed_42_{lm_head_only,embed_only,dual,dual_no_eos,dual_no_anti,dual_no_eos_anti}.json`

Readout: `lm_head` performs the rewrite, embeddings alone do not. The current
`dual` profile is locality-protected mostly because of `clamp_anti`; removing
anti improves PS@All but drops locality. EOS does not materially improve the
single-edit n=200 result.

Fresh exact-additive ablation:

- `exact_additive_report_2026-05-18.md`
- `exact_additive_random_200_seed_42_dual.json`

Readout: disabling WAL re-encoding gives only a small n=200 gain, so the
current ceiling is mainly the edit locus/control policy rather than WAL
quantization.

Deterministic NT probe:

- `nt_deterministic_probe_2026-05-18.md`
- `nt_deterministic_probe_5_seed_42.json`

Readout: new artifacts store deterministic `lm_head_sampled_row_ids` and
`embed_sampled_row_ids` in each NT metrics row. This is a diagnostic schema
probe, not a benchmark result.

Norm metrics probe:

- `norm_metrics_probe_2026-05-18.md`
- `norm_metrics_probe_5_seed_42.json`

Readout: new artifacts store edited-row delta L2 norm metrics under `NT`.
This is a diagnostic schema probe, not a benchmark result.

EOS default validation:

- `eos_default_report_2026-05-18.md`
- `eos_default_random_200_seed_{42,43,44}_no_eos.json`

Readout: removing the global EOS row leaves random-200 single-edit metrics
effectively unchanged and avoids a persistent global control-row mutation.

Both primary-sequence fix:

- `both_primary_fixed_report_2026-05-18.md`
- `both_primary_fixed_random_200_seed_42.json`

Readout: `target_token_mode=both` now uses contextual continuation ids as the
primary sequence, so it no longer relies on standalone ids for stop/key control.

Positive ridge constraint:

- `positive_ridge_report_2026-05-18.md`
- `positive_ridge_50_seed_42.json`
- `positive_ridge_50_seed_42.failures.json`

Readout: the ridge constrained positive/protected key solve restores locality
relative to naive positive prompts (`TF locality=99.5%`, `Prob locality=93.2%`)
but drops exact rewrite to `80.0%`, so it is an ablation path rather than the
default profile.

Norm budget no-commit probe:

- `norm_budget_no_commit_probe_2026-05-18.md`
- `norm_budget_no_commit_probe_5_seed_42.json`
- `norm_budget_no_commit_probe_5_seed_42.failures.json`

Readout: a deliberately strict row-delta budget rejects 5/5 proposed edits and
marks them as `edit_status=no_commit`, with post-edit locality preserved at
100.0%. This validates the runtime rollback gate, not an operating profile.

Sequential backend matrix:

- `backend_matrix_sequential_random_50_report_2026-05-18.md`
- `backend_matrix_sequential_random_50_seed42.{dual_row,side_slot,wal_rome}.json`

Readout: `side_slot` is the strongest current sequential backend on seed-42
n=50 (`TF rewrite=98.0%`, `TF locality=97.2%`). In-place `dual_row` reaches
`TF rewrite=88.0%` but drops to `TF locality=59.9%`; `wal_rome` preserves
locality but does not rewrite with the current configuration.
