# Sequential EasyEdit-Compatible Artifacts

This folder contains n=50 sequential editing runs where edits are accumulated
before final evaluation.

Current readout: sequential editing is partial and weak. The best balanced
profile improves over the initial sequential collapse, but exact-token locality
and rephrase remain below the level needed for a lifelong editing claim.

Fresh PS@All artifacts:

- `easyedit_official_50_first42_psall_seq_lm015_negx05_noeosanti_retention.json`
- `easyedit_official_50_first42_psall_seq_lm015_negx05_noeosanti_positive_prompts_retention.json`
- `easyedit_official_50_first42_psall_seq_lm015_negx05_orthogonal_noeosanti_retention.json`

Random-seed retention report:

- `sequential_random_50_report_2026-05-18.md`
- `random_50_seed_42_seq_lm015_negx05_noeosanti_retention.json`
- `random_50_seed_43_seq_lm015_negx05_noeosanti_retention.json`
- `random_50_seed_44_seq_lm015_negx05_noeosanti_retention.json`
- `random_50_seed_42_seq_lm015_negx05_relation_slots_noeosanti_retention.json`
- `random_50_seed_42_seq_lm015_negx05_projected_positive_noeosanti_retention.json`

Use these artifacts to discuss sequential failure modes, tuning progress, and
next research steps. Do not cite them as solved sequential editing.
