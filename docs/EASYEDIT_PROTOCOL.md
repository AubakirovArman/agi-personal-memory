# EasyEdit-Compatible Protocol Notes

AGIM keeps the editor local, but the current runner evaluates with local
EasyEdit metric functions where possible.

## Source Of Truth

- Runner: `agim.eval.easyedit_official_runner`
- Current artifacts: `results/easyedit_official/current/`
- Sequential artifacts: `results/easyedit_official/sequential/`
- Summary: `BENCHMARK.md`

## Metric Groups

| Metric group | Meaning | Headline use |
| --- | --- | --- |
| Teacher-forcing token EM | EasyEdit-style target token equality under teacher forcing | Main internal metric |
| Official vanilla generation | EasyEdit generation equality against `tok.encode(target_new)` | Report separately |
| Contextual generation | Greedy equality against `prompt + space + target` suffix ids | Diagnostic for Llama tokenization |
| Probability compare | Checks whether `P(target_new) > P(target_true)` | Diagnostic |
| Locality | Pre/post neighbor output consistency | Main weakness today |
| `PS@All` / `rephrase_all_acc` | Rephrase aggregate across all CounterFact paraphrases | Diagnostic for cherry-pick risk |
| Sequential retention | Metrics after selected accumulated-edit checkpoints | Diagnostic for collapse timing |
| Multi-positive key ablation | Optional edit-key averaging with paraphrase prompts | Experimental method knob |
| Orthogonal projection ablation | Optional protected-subspace projection mode | Negative result so far |
| NT diff | Non-edited `lm_head`/embedding row drift | Diagnostic |

## Why Vanilla Generation Is Separate

For Llama tokenization, EasyEdit's vanilla generation target ids can differ from
the teacher-forcing continuation ids. Example: standalone `English` can tokenize
differently from the suffix in `The language is English`.

Because of this, reports must show:

```text
Teacher-forcing token EM
Official vanilla generation
Contextual generation diagnostic
Probability diagnostic
```

Do not merge those into one headline number.

## Completed Random-Seed Runs

The required random n=50 single-edit presets have been run with the current
default locality-protected profile:

```text
results/easyedit_official/current/random_50_seed_42.json
results/easyedit_official/current/random_50_seed_43.json
results/easyedit_official/current/random_50_seed_44.json
```

Mean readout over seeds 42/43/44: `TF rewrite=92.7%`,
`TF PS@All=23.3%`, `TF locality=97.1%`, `Prob locality=87.8%`.

This validates rewrite/locality stability for the default profile, while also
showing weak paraphrase transfer. The detailed report is
`results/easyedit_official/current/random_50_report_2026-05-18.md`.

## Required Future Runs

For sequential runs, add retention reporting:

```text
after edit 1: evaluate edits 1..1
after edit 10: evaluate edits 1..10
after edit 50: evaluate edits 1..50
```

## Current Gaps

- `PS@All`, `metrics_by_relation_id`, and sequential retention summaries are
  emitted for new artifacts; older artifacts do not contain them.
- `--dry-run-summary` writes the selected case ids, relation counts, and prompt
  counts without loading the model or EasyEdit.
- `--save-failures-only` writes a compact failure triage artifact next to the
  main run output.
- `--history-slot-mode relation` is a relation-sharded sequential history
  ablation. It keeps edit-key history slots by CounterFact `relation_id`.
- `--positive-constraint-mode projected` is a constrained positive/protected
  key ablation. It projects paraphrase-positive keys away from protected
  locality keys before mixing them into the edit key.
- Sequential retention is currently summary-only. Full retention metrics should
  be stored only when needed because they can make artifacts large.
- `--use-positive-prompts` is backed by n=50 GPU artifacts. It improves
  PS@All but currently hurts locality, so it is an ablation knob rather than the
  default headline setting.
- `--projection-mode orthogonal` is backed by an n=50 GPU artifact. It is worse
  on exact-token locality than the tuned sequential baseline, so it is not a
  recommended default.
