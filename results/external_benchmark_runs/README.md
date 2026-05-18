# External Benchmark Runs

This folder contains tracked model-output runs for external-style benchmark
adapters. These artifacts are not EasyEdit CounterFact artifacts and must not
be compared directly to `results/easyedit_official/current/`.

Current runs:

| Artifact | Benchmark | Readout |
| --- | --- | --- |
| `mquake_cf_3k_v2_first50_dual_row_report_2026-05-18.md` | MQuAKE-CF-3k-v2 first 50 | `direct_rewrite_acc=100.0%`, `multi_hop_acc=34.0%`, `composite_acc=67.0%` |

Use these as diagnostic external benchmark evidence unless and until the run is
matched to the exact protocol and model family used by an external leaderboard.
