# Random 200 EasyEdit-Compatible Report - 2026-05-18

Model: `meta-llama/Llama-3.1-8B-Instruct`

Runner: `agim.eval.easyedit_official_runner`

Code commit in artifacts: `6c88dadd2dc4d8b1375579a9619264db6dfbdf6b`

Artifact schema: `easyedit_official.v2`

Method profile: `single_loc`

Dataset: CounterFact from `https://rome.baulab.info/data/dsets/counterfact.json`

Command shape:

```text
AGIM_EASYEDIT_ROOT=/path/to/EasyEdit PYTHONPATH=src \
python -m agim.eval.easyedit_official_runner \
  --n 200 --sample-policy random --seed <seed> --device cuda:<2|3> \
  --output results/easyedit_official/current/random_200_seed_<seed>.json \
  --save-failures-only
```

## Artifacts

| Seed | Main artifact | Dry run | Failures-only |
| ---: | --- | --- | --- |
| 42 | `random_200_seed_42.json` | `random_200_seed_42.dry_run.json` | `random_200_seed_42.failures.json` |
| 43 | `random_200_seed_43.json` | `random_200_seed_43.dry_run.json` | `random_200_seed_43.failures.json` |
| 44 | `random_200_seed_44.json` | `random_200_seed_44.dry_run.json` | `random_200_seed_44.failures.json` |

## Summary

| Seed | TF rewrite | TF rephrase | TF PS@All | TF locality | CTX rewrite | CTX PS@All | Prob rewrite | Prob PS@All | Prob locality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 97.0% | 28.5% | 27.0% | 95.9% | 96.5% | 26.5% | 99.5% | 44.5% | 85.9% |
| 43 | 89.0% | 20.8% | 22.2% | 96.9% | 89.0% | 21.5% | 94.5% | 38.0% | 88.6% |
| 44 | 93.5% | 26.5% | 27.5% | 96.4% | 93.5% | 26.5% | 98.0% | 46.5% | 85.2% |
| Mean | 93.2% | 25.3% | 25.6% | 96.4% | 93.0% | 24.8% | 97.3% | 43.0% | 86.6% |

Failures-only artifacts use `failure_families=tf,ctx_gen,prob`, so EasyEdit
standalone vanilla generation mismatch does not dominate triage by default.

| Seed | Failed cases | Main failure modes |
| ---: | ---: | --- |
| 42 | 190 | `tf_ps_all=173`, `tf_rephrase=144`, `prob_locality=120`, `tf_locality=55`, `tf_rewrite=6` |
| 43 | 193 | `tf_ps_all=180`, `tf_rephrase=160`, `prob_locality=104`, `tf_locality=46`, `tf_rewrite=22` |
| 44 | 196 | `tf_ps_all=167`, `tf_rephrase=149`, `prob_locality=123`, `tf_locality=53`, `tf_rewrite=13` |

## Readout

The random-200 layer supports the same conclusion as random-50 and first-1000:
`single_loc` is a stable exact rewrite/locality profile, not a paraphrase
generalization solution.

Compared with random-50, the mean rewrite and locality remain stable
(`TF rewrite=93.2%`, `TF locality=96.4%`). PS@All remains weak
(`TF PS@All=25.6%`). This is useful evidence for an audited hotfix profile and
not enough for a lifelong-editing or broad semantic-update claim.
