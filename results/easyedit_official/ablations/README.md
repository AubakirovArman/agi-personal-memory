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
