# Current EasyEdit-Compatible Artifacts

This folder contains the current single-edit source of truth for AGIM WAL
weight-editing claims.

Primary files:

- `easyedit_agim_status_2026-05-18.md`: human-readable status and caveats.
- `easyedit_psall_report_2026-05-18.md`: fresh PS@All comparison report.
- `easyedit_1000_first_default_report_2026-05-18.md`: official-compatible
  n=1000 scale check for the current default single-edit profile.
- `random_50_report_2026-05-18.md`: random-seed n=50 validation report.
- `random_200_report_2026-05-18.md`: random-seed n=200 validation report.
- `random_1000_report_2026-05-18.md`: random-seed n=1000 validation
  report.
- `relation_failure_notes_2026-05-18.md`: relation-level failure triage.
- `easyedit_official_50_first42_psall_baseline.json`: current n=50
  single-edit baseline artifact.
- `random_50_seed_{42,43,44}.json`: current random-seed n=50 single-edit
  artifacts.
- `random_200_seed_{42,43,44}.json`: current random-seed n=200 single-edit
  artifacts.
- `random_1000_seed_42.json`: current random-seed n=1000 single-edit
  artifact.
- `random_50_seed_{42,43,44}.dry_run.json`: selected case/relation inspection
  artifacts for the same presets.
- `random_200_seed_{42,43,44}.dry_run.json`: selected case/relation inspection
  artifacts for the random n=200 presets.
- `random_50_seed_{42,43,44}.failures.json`: compact failed-case triage for
  the same presets.
- `random_200_seed_{42,43,44}.failures.json`: compact failed-case triage for
  the random n=200 presets.
- `easyedit_official_1000_first_default.json`: first 1000 CounterFact facts
  with the current default single-edit profile.
- `easyedit_official_1000_first_default.dry_run.json`: selected case/relation
  inspection for the n=1000 run.
- `easyedit_official_1000_first_default.failures.json`: compact failed-case
  triage for the n=1000 run.
- `random_1000_seed_42.dry_run.json`: selected case/relation inspection for
  the random n=1000 run.
- `random_1000_seed_42.failures.json`: compact failed-case triage for the
  random n=1000 run.
- `easyedit_official_50_contextual_neg4_ctxgen_nt.json`: older n=50
  single-edit artifact kept for audit.

Do not mix these numbers with `../sequential/` or `../ablations/` without
stating the protocol difference.
