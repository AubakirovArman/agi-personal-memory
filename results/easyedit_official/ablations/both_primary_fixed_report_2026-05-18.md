# Both Primary Sequence Fix Report - 2026-05-18

This report validates the `target_token_mode=both` primary-sequence fix.

Before the fix, `both` edited both standalone and contextual target token
sequences but used the first sequence as primary, which meant the stop key and
embedding direction were aligned to standalone ids. After the fix, `both`
still edits both sequences but uses the contextual continuation sequence as
primary.

Model: `meta-llama/Llama-3.1-8B-Instruct`

Runner: `agim.eval.easyedit_official_runner`

Code commit in fixed artifact: `92de9da9a8d2cf4a0ef1f5946f50ab48b78b95a0`

Dataset: CounterFact random n=200, seed `42`

Common settings: `target_token_mode=both`, `clamp_lm=0.20`,
`clamp_embed=0.06`, `clamp_eos=0.16`, `clamp_anti=0.06`,
`nt_sample_size=32`.

## Artifacts

- `both_primary_fixed_random_200_seed_42.json`
- `both_primary_fixed_random_200_seed_42.dry_run.json`
- `both_primary_fixed_random_200_seed_42.failures.json`

Comparison baseline:

- `token_mode_random_200_seed_42_both.json`

## Metrics

| `both` behavior | TF rewrite | TF PS@All | TF locality | CTX rewrite | CTX PS@All | Prob rewrite | Prob PS@All | Prob locality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Before fix | 97.5% | 28.2% | 94.8% | 97.5% | 27.8% | 99.5% | 45.5% | 84.4% |
| Contextual-primary fix | 96.0% | 27.0% | 95.9% | 96.0% | 26.5% | 99.5% | 44.2% | 85.8% |

## Failure Triage

The fixed artifact uses `failure_families=tf,ctx_gen,prob`.

| Failure mode | Count |
| --- | ---: |
| `tf_ps_all` | 173 |
| `tf_rephrase` | 144 |
| `prob_locality` | 121 |
| `tf_locality` | 55 |
| `tf_rewrite` | 8 |
| `ctx_gen_rewrite` | 8 |

## Readout

The fix behaves as intended: `both` no longer gets primary-key behavior from
standalone ids. It now looks much closer to the contextual profile, with better
locality than the old `both` result and without the previous small rewrite/PS
boost.

`both` should remain an ablation knob, not a new default. It is useful for
auditing tokenization effects, while the current default remains
`target_token_mode=contextual`.
