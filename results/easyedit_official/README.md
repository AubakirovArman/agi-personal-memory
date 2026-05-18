# EasyEdit-Compatible Results

These artifacts are the current source of truth for AGIM WAL weight-editing
claims against CounterFact on `meta-llama/Llama-3.1-8B-Instruct`.

They are produced by:

```bash
PYTHONPATH=src python -m agim.eval.easyedit_official_runner ...
```

The runner keeps AGIM's editor local, but computes pre/post metrics using the
local EasyEdit source tree:

- `easyeditor.evaluate.evaluate.compute_edit_quality`
- `easyeditor.evaluate.evaluate_utils.test_prediction_acc`
- EasyEdit BaseEditor-style locality comparison: pre-edit neighbor output vs
  post-edit neighbor output

## Directory Layout

| Folder | Purpose |
| --- | --- |
| `current/` | Human status summary and the current best single-edit artifact |
| `sequential/` | Sequential n=50 runs, including tuned projection profiles |
| `ablations/` | Older single-edit/tuning artifacts kept for audit |
| `smoke/` | Ignored local smoke outputs, not committed |

## Primary Artifacts

| Artifact | Mode | Use |
| --- | --- | --- |
| `current/easyedit_official_50_first42_psall_baseline.json` | Single edit with rollback after each fact | Current baseline with PS@All |
| `current/random_50_report_2026-05-18.md` | Single-edit random-seed report | Stability check for seeds 42/43/44 |
| `current/easyedit_1000_first_default_report_2026-05-18.md` | Single-edit n=1000 report | Correct current EasyEdit-compatible scale check |
| `ablations/easyedit_official_50_first42_psall_positive_prompts.json` | Single edit with positive prompts | PS@All ablation, locality tradeoff |
| `sequential/easyedit_official_50_first42_psall_seq_lm015_negx05_noeosanti_retention.json` | Sequential, tuned projection | Current balanced sequential profile with retention |
| `sequential/sequential_random_50_report_2026-05-18.md` | Sequential random-seed report | Retention check for seeds 42/43/44 and seed 42 ablations |
| `sequential/easyedit_official_50_first42_psall_seq_lm015_negx05_noeosanti_positive_prompts_retention.json` | Sequential, tuned projection plus positive prompts | Sequential PS@All ablation |
| `sequential/easyedit_official_50_first42_psall_seq_lm015_negx05_orthogonal_noeosanti_retention.json` | Sequential, orthogonal projection | Negative locality ablation |
| `sequential/easyedit_official_50_contextual_neg4x05_seq_lm012_noeosanti.json` | Sequential, tuned projection | Best exact-token locality among tuned n=50 runs |
| `sequential/easyedit_official_50_contextual_neg4x05_seq_lm020_noeosanti.json` | Sequential, tuned projection | Best exact-token rewrite, weak locality |
| `sequential/easyedit_official_50_contextual_neg4x07_seq_lm020_noeosanti.json` | Sequential, stronger projection | Strong probability locality, weak rephrase |
| `current/easyedit_agim_status_2026-05-18.md` | Human summary | Read first |
| `current/easyedit_psall_report_2026-05-18.md` | Human PS@All report | Fresh n=50 comparison table |
| `current/relation_failure_notes_2026-05-18.md` | Human relation triage | Worst relation ids by locality |

Smoke and failed/tuning runs live in `smoke/` and are intentionally ignored by
git.
