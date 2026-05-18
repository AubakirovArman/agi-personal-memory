# Random 50 EasyEdit-Compatible Report - 2026-05-18

Model: `meta-llama/Llama-3.1-8B-Instruct`

Runner: `agim.eval.easyedit_official_runner`

Code commit in artifacts: `8d3f69991fbc6611d30691c00e3ef181d0a5ac05`

Dataset: CounterFact from `https://rome.baulab.info/data/dsets/counterfact.json`

Command shape:

```text
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --preset random_50_seed_<seed> --device cuda:<2|3> --save-failures-only
```

## Artifacts

| Seed | Main artifact | Dry run | Failures-only |
| ---: | --- | --- | --- |
| 42 | `random_50_seed_42.json` | `random_50_seed_42.dry_run.json` | `random_50_seed_42.failures.json` |
| 43 | `random_50_seed_43.json` | `random_50_seed_43.dry_run.json` | `random_50_seed_43.failures.json` |
| 44 | `random_50_seed_44.json` | `random_50_seed_44.dry_run.json` | `random_50_seed_44.failures.json` |

## Summary

| Seed | TF rewrite | TF rephrase | TF PS@All | TF locality | CTX rewrite | CTX PS@All | Prob rewrite | Prob PS@All | Prob locality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 98.0% | 30.0% | 25.0% | 97.2% | 98.0% | 25.0% | 100.0% | 52.0% | 87.4% |
| 43 | 86.0% | 18.0% | 22.0% | 96.8% | 86.0% | 20.0% | 92.0% | 38.0% | 88.4% |
| 44 | 94.0% | 26.0% | 23.0% | 97.4% | 92.0% | 23.0% | 100.0% | 43.0% | 87.6% |
| Mean | 92.7% | 24.7% | 23.3% | 97.1% | 92.0% | 22.7% | 97.3% | 44.3% | 87.8% |

Official vanilla generation is 0.0% on all three runs. Keep it reported, but do
not collapse it with teacher-forcing or contextual generation because Llama
target tokenization differs between standalone target strings and prompt
continuations.

## Readout

These random-seed n=50 runs validate that the current default single-edit path
is not a one-sample accident: exact rewrite stays high across seeds and
teacher-forcing locality is strong.

They also show the current tradeoff clearly. The default locality-protected
profile does not generalize well to paraphrases: teacher-forcing PS@All is only
22-25%. The older first-50 PS@All baseline and positive-prompt ablation show
higher paraphrase generalization, but at much weaker locality. Treat these as
different operating points, not one combined score.

Failure-only artifacts are intentionally broad: every case is marked failed
because official vanilla generation fails under the EasyEdit standalone target
tokenization check. For locality triage, inspect `tf_locality` counts instead:
10, 12, and 11 failed cases for seeds 42, 43, and 44.
