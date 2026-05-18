# Other Benchmark Artifacts

This folder contains early KnowEdit, MQuAKE, and WikiBio diagnostic outputs.
They are not the current source of truth for EasyEdit/CounterFact claims.

Use them only to understand historical experiments and future benchmark work.

## Current Status

These artifacts are legacy diagnostics, not current external benchmark claims:

| File | Status | Notes |
| --- | --- | --- |
| `mquake_wal.json` | Legacy local diagnostic | Small custom MQuAKE-style run; direct edits often repeat target strings and multi-hop remains weak |
| `mquake_rome.json` | Legacy local diagnostic | Same custom task family as `mquake_wal.json`; not an official MQuAKE dataset score |
| `mquake_real_wal_wal.json` | Legacy local diagnostic | Older real-MQuAKE experiment summary without current EasyEdit artifact metadata |
| `knowedit_wal_wal.json` | Legacy local diagnostic | Older KnowEdit-style summary, not a current KnowEdit/UniEdit/ScEdit leaderboard result |
| `wikibio_wal.json` | Legacy local diagnostic | Small WikiBio-style correction check; repetition artifacts are visible in outputs |
| `wikibio_rome.json` | Legacy local diagnostic | Small WikiBio-style correction check; not part of current Path B claims |

Current Path B claims should cite:

- `results/easyedit_official/current/` for single-edit EasyEdit-compatible
  evidence;
- `results/easyedit_official/sequential/` for sequential limitations and
  side-slot experiments;
- `results/easyedit_official/ablations/` for component/backend diagnostics;
- `docs/PATH_B_COMPLETION_AUDIT.md` for remaining gaps.
