# EasyEdit 1000 First Default Report - 2026-05-18

Model: `meta-llama/Llama-3.1-8B-Instruct`

Runner: `agim.eval.easyedit_official_runner`

Code commit in artifact: `f712fbc0d0946058633a2426b203edd7fc7c620d`

Dataset: CounterFact from `https://rome.baulab.info/data/dsets/counterfact.json`

Dataset sha256: `d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f`

Selection: first 1000 records, case ids `0..999`

Command:

```text
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --n 1000 --sample-policy first --device cuda:2 \
  --output results/easyedit_official/current/easyedit_official_1000_first_default.json \
  --save-failures-only
```

## Artifacts

- `easyedit_official_1000_first_default.json`
- `easyedit_official_1000_first_default.dry_run.json`
- `easyedit_official_1000_first_default.failures.json`

## Metrics

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing | 91.1% | 25.4% | 24.7% | 96.2% |
| Official vanilla generation | 0.1% | 0.1% | 0.1% | n/a |
| Contextual generation | 91.0% | 24.8% | 24.1% | n/a |
| Probability compare | 96.3% | 43.5% | 43.6% | 87.5% |

NT diagnostics:

| Metric | Value |
| --- | ---: |
| `lm_head_non_edited_max` | 0.0 |
| `embed_non_edited_max` | 0.0 |
| `edited_lm_rows_avg` | 2.012 |
| `edited_embed_rows_avg` | 4.140 |
| `eos_row_changed_rate` | 100.0% |

Runtime: `1490.07s`, `1.4901s/edit`.

## Failure Triage

The failures-only artifact marks every case as failed because official vanilla
generation fails almost every standalone-target tokenization check. For method
triage, use the specific failure modes:

| Failure mode | Count |
| --- | ---: |
| `tf_rewrite` | 91 |
| `tf_rephrase` | 751 |
| `tf_ps_all` | 868 |
| `tf_locality` | 260 |
| `ctx_gen_rewrite` | 90 |
| `prob_locality` | 557 |
| `gen_rewrite` | 1000 |
| `gen_rephrase` | 1000 |
| `gen_ps_all` | 1000 |

Worst relation ids by teacher-forcing locality:

| Relation | n | TF rewrite | TF PS@All | TF locality | Prob locality |
| --- | ---: | ---: | ---: | ---: | ---: |
| P1303 | 17 | 100.0% | 29.4% | 87.1% | 74.1% |
| P190 | 32 | 100.0% | 18.8% | 89.1% | 70.0% |
| P641 | 22 | 100.0% | 18.2% | 89.1% | 82.7% |
| P407 | 12 | 100.0% | 29.2% | 90.4% | 75.8% |
| P364 | 33 | 90.9% | 40.9% | 91.5% | 84.2% |

Worst relation ids by PS@All:

| Relation | n | TF rewrite | TF PS@All | TF locality | Prob PS@All |
| --- | ---: | ---: | ---: | ---: | ---: |
| P1412 | 40 | 87.5% | 1.2% | 94.5% | 13.8% |
| P136 | 33 | 84.8% | 7.6% | 99.1% | 36.4% |
| P20 | 42 | 85.7% | 8.3% | 96.3% | 32.1% |
| P413 | 46 | 87.0% | 8.7% | 97.6% | 19.6% |
| P127 | 16 | 100.0% | 9.4% | 98.1% | 21.9% |

## Readout

This is the correct current EasyEdit-compatible 1000-fact scale check for the
default locality-protected single-edit profile. It is not the legacy local
1000-fact protocol.

The result is good for exact rewrite and locality at scale: teacher-forcing
rewrite stays above 91% and teacher-forcing locality stays above 96% across the
first 1000 CounterFact facts.

The same result is weak for paraphrase transfer: PS@All is 24.7% under
teacher-forcing and 43.6% under the probability diagnostic. This confirms the
n=50 conclusion at larger scale: the current default profile is a
rewrite/locality profile, not a solved generalization profile.
