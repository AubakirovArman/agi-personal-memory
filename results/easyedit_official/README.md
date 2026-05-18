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

## Primary Artifacts

| Artifact | Mode | Use |
| --- | --- | --- |
| `easyedit_official_50_contextual_neg4_ctxgen_nt.json` | Single edit with rollback after each fact | Best current single-edit result |
| `easyedit_official_50_contextual_neg4x05_seq_lm015_noeosanti.json` | Sequential, tuned projection | Best balanced sequential profile |
| `easyedit_official_50_contextual_neg4x05_seq_lm012_noeosanti.json` | Sequential, tuned projection | Best exact-token locality among tuned n=50 runs |
| `easyedit_official_50_contextual_neg4x05_seq_lm020_noeosanti.json` | Sequential, tuned projection | Best exact-token rewrite, weak locality |
| `easyedit_official_50_contextual_neg4x07_seq_lm020_noeosanti.json` | Sequential, stronger projection | Strong probability locality, weak rephrase |
| `easyedit_agim_status_2026-05-18.md` | Human summary | Read first |

Smoke and failed/tuning runs live in `smoke/` and are intentionally ignored by
git.

