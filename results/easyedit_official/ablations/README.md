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
