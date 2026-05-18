# EasyEdit PS@All Report - 2026-05-18

Model: `meta-llama/Llama-3.1-8B-Instruct`

Dataset: CounterFact first 50 facts, seed 42

Runner commits:

- `cadd6f82f4e5b979610caf0441e6790028ed62ab` for baseline,
  positive-prompt, and tuned sequential artifacts.
- `0d21721c1918225f9071b7ee852d8d4cff2e050d` for the orthogonal projection
  artifact.

GPU used: `cuda:3`

## Fresh Artifacts

| Artifact | Mode |
| --- | --- |
| `current/easyedit_official_50_first42_psall_baseline.json` | Single edit, rollback after each fact |
| `ablations/easyedit_official_50_first42_psall_positive_prompts.json` | Single edit with `--use-positive-prompts` |
| `sequential/easyedit_official_50_first42_psall_seq_lm015_negx05_noeosanti_retention.json` | Sequential tuned profile with retention |
| `sequential/easyedit_official_50_first42_psall_seq_lm015_negx05_noeosanti_positive_prompts_retention.json` | Sequential tuned profile with positive prompts and retention |
| `sequential/easyedit_official_50_first42_psall_seq_lm015_negx05_orthogonal_noeosanti_retention.json` | Sequential tuned profile with orthogonal projection and retention |

## Headline Metrics

| Run | TF rewrite | TF rephrase | TF PS@All | TF locality | Prob rewrite | Prob rephrase | Prob PS@All | Prob locality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single baseline | 100.0% | 71.0% | 67.0% | 58.4% | 100.0% | 88.0% | 89.0% | 37.4% |
| Single positive prompts | 100.0% | 96.0% | 95.0% | 45.2% | 100.0% | 96.0% | 98.0% | 25.2% |
| Sequential tuned | 73.0% | 21.0% | 20.0% | 25.4% | 86.0% | 62.0% | 61.0% | 61.4% |
| Sequential positive prompts | 47.0% | 31.0% | 29.5% | 16.2% | 84.0% | 78.0% | 79.0% | 56.0% |
| Sequential orthogonal projection | 58.0% | 25.0% | 25.0% | 6.6% | 90.0% | 82.0% | 81.0% | 30.4% |

## Sequential Retention

| Run | Checkpoint | TF rewrite | TF rephrase | TF PS@All | TF locality | Prob PS@All | Prob locality |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sequential tuned | after 1 | 100.0% | 0.0% | 0.0% | 60.0% | 50.0% | 60.0% |
| Sequential tuned | after 10 | 100.0% | 50.0% | 50.0% | 59.0% | 90.0% | 39.0% |
| Sequential tuned | after 50 | 73.0% | 21.0% | 20.0% | 25.4% | 61.0% | 61.4% |
| Sequential positive prompts | after 1 | 100.0% | 100.0% | 100.0% | 90.0% | 100.0% | 90.0% |
| Sequential positive prompts | after 10 | 100.0% | 100.0% | 100.0% | 51.0% | 100.0% | 29.0% |
| Sequential positive prompts | after 50 | 47.0% | 31.0% | 29.5% | 16.2% | 79.0% | 56.0% |
| Sequential orthogonal projection | after 1 | 100.0% | 100.0% | 100.0% | 30.0% | 100.0% | 0.0% |
| Sequential orthogonal projection | after 10 | 90.0% | 60.0% | 65.0% | 10.0% | 100.0% | 10.0% |
| Sequential orthogonal projection | after 50 | 58.0% | 25.0% | 25.0% | 6.6% | 81.0% | 30.4% |

## Readout

`--use-positive-prompts` is a real single-edit generalization win: TF PS@All
improves from 67.0% to 95.0%. It is not a global quality win because locality
drops from 58.4% to 45.2% under teacher forcing and from 37.4% to 25.2% under
probability locality.

Sequential editing remains the blocker. The tuned sequential profile retains
good rewrite through 10 accumulated edits, then drops by edit 50. Positive
prompts improve sequential PS@All but hurt exact rewrite and locality.
Orthogonal protected projection is also not enough: it improves probability
rewrite/PS@All but sharply hurts exact-token locality. This points to
accumulated edit interference and row sharing, not only a missing paraphrase key
or weak protected-key projection.

The next method work should prioritize protected-key/null-space projection,
per-edit isolation, or relation sharding before making stronger EasyEdit or
lifelong-editing claims.
