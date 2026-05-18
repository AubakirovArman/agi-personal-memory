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

## Required Future Runs

Before making stronger external claims, run:

```text
first_50
random_50_seed_42
random_50_seed_43
first_200
random_200_seed_42
```

For sequential runs, add retention reporting:

```text
after edit 1: evaluate edits 1..1
after edit 10: evaluate edits 1..10
after edit 50: evaluate edits 1..50
```

## Current Gaps

- `PS@All` should be added alongside first-rephrase reporting.
- `metrics_by_relation_id` is emitted for new artifacts; older artifacts do not
  contain it.
- Sequential retention curves should be emitted as first-class artifacts.
