# Component Ablation Report - 2026-05-18

Model: `meta-llama/Llama-3.1-8B-Instruct`

Runner: `agim.eval.easyedit_official_runner`

Code commit in artifacts: `1448ad24b07e1825a55735fcf6b472cc1cd357b5`

Artifact schema: `easyedit_official.v2`

Method family: `single_loc`

Dataset: CounterFact from `https://rome.baulab.info/data/dsets/counterfact.json`

Selection: random 200 records with seed `42`

Selected case-id hash: `2517f71c21db2952484dd83aa87afce592a2ceba1c81b8bacffa943b7a26439e`

All runs use `target_token_mode=contextual`.

## Artifacts

| Ablation | Main artifact | Dry run | Failures-only |
| --- | --- | --- | --- |
| `lm_head_only` | `component_random_200_seed_42_lm_head_only.json` | `component_random_200_seed_42_lm_head_only.dry_run.json` | `component_random_200_seed_42_lm_head_only.failures.json` |
| `embed_only` | `component_random_200_seed_42_embed_only.json` | `component_random_200_seed_42_embed_only.dry_run.json` | `component_random_200_seed_42_embed_only.failures.json` |
| `dual` | `component_random_200_seed_42_dual.json` | `component_random_200_seed_42_dual.dry_run.json` | `component_random_200_seed_42_dual.failures.json` |
| `dual_no_eos` | `component_random_200_seed_42_dual_no_eos.json` | `component_random_200_seed_42_dual_no_eos.dry_run.json` | `component_random_200_seed_42_dual_no_eos.failures.json` |
| `dual_no_anti` | `component_random_200_seed_42_dual_no_anti.json` | `component_random_200_seed_42_dual_no_anti.dry_run.json` | `component_random_200_seed_42_dual_no_anti.failures.json` |
| `dual_no_eos_anti` | `component_random_200_seed_42_dual_no_eos_anti.json` | `component_random_200_seed_42_dual_no_eos_anti.dry_run.json` | `component_random_200_seed_42_dual_no_eos_anti.failures.json` |

## Hyperparameters

| Ablation | `clamp_lm` | `clamp_embed` | `clamp_eos` | `clamp_anti` |
| --- | ---: | ---: | ---: | ---: |
| `lm_head_only` | 0.20 | 0.00 | 0.00 | 0.00 |
| `embed_only` | 0.00 | 0.06 | 0.00 | 0.00 |
| `dual` | 0.20 | 0.06 | 0.16 | 0.06 |
| `dual_no_eos` | 0.20 | 0.06 | 0.00 | 0.06 |
| `dual_no_anti` | 0.20 | 0.06 | 0.16 | 0.00 |
| `dual_no_eos_anti` | 0.20 | 0.06 | 0.00 | 0.00 |

## Metrics

| Ablation | TF rewrite | TF rephrase | TF PS@All | TF locality | CTX rewrite | CTX PS@All | Prob rewrite | Prob PS@All | Prob locality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lm_head_only` | 100.0% | 48.5% | 47.0% | 88.9% | 100.0% | 46.2% | 100.0% | 68.5% | 71.4% |
| `embed_only` | 0.5% | 0.8% | 0.9% | 99.9% | 0.0% | 0.5% | 6.5% | 8.5% | 88.5% |
| `dual` | 96.5% | 28.5% | 26.8% | 95.8% | 96.5% | 26.2% | 99.5% | 44.2% | 86.0% |
| `dual_no_eos` | 96.5% | 28.5% | 27.0% | 95.8% | 96.5% | 26.5% | 99.5% | 44.8% | 85.9% |
| `dual_no_anti` | 100.0% | 48.5% | 46.5% | 88.8% | 100.0% | 46.0% | 100.0% | 68.0% | 71.4% |
| `dual_no_eos_anti` | 100.0% | 48.5% | 46.5% | 88.8% | 100.0% | 46.0% | 100.0% | 68.0% | 71.4% |

NT diagnostics:

| Ablation | Non-edited lm_head max | Non-edited embed max | Edited lm rows avg | Edited embed rows avg | EOS changed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lm_head_only` | 0.0 | 0.0 | 1.010 | 0.000 | 0.0% |
| `embed_only` | 0.0 | 0.0 | 0.000 | 4.230 | 0.0% |
| `dual` | 0.0 | 0.0 | 2.010 | 4.230 | 100.0% |
| `dual_no_eos` | 0.0 | 0.0 | 1.010 | 4.230 | 0.0% |
| `dual_no_anti` | 0.0 | 0.0 | 2.010 | 4.230 | 100.0% |
| `dual_no_eos_anti` | 0.0 | 0.0 | 1.010 | 4.230 | 0.0% |

## Failure Triage

Failures-only artifacts use `failure_families=tf,ctx_gen,prob`.

| Ablation | Failed cases | Main failure modes |
| --- | ---: | --- |
| `lm_head_only` | 189 | `prob_locality=160`, `tf_locality=113`, `tf_ps_all=136`, `tf_rephrase=104` |
| `embed_only` | 200 | `tf_rewrite=200`, `tf_rephrase=199`, `tf_ps_all=200`, `ctx_gen_rewrite=200`, `prob_locality=84` |
| `dual` | 190 | `tf_ps_all=173`, `tf_rephrase=144`, `prob_locality=120`, `tf_locality=56`, `tf_rewrite=7` |
| `dual_no_eos` | 190 | `tf_ps_all=173`, `tf_rephrase=144`, `prob_locality=120`, `tf_locality=55`, `tf_rewrite=7` |
| `dual_no_anti` | 189 | `prob_locality=160`, `tf_locality=114`, `tf_ps_all=137`, `tf_rephrase=104` |
| `dual_no_eos_anti` | 189 | `prob_locality=160`, `tf_locality=114`, `tf_ps_all=137`, `tf_rephrase=104` |

## Readout

`lm_head` is the component that actually performs the current rewrite.
`lm_head_only` reaches `100.0%` TF rewrite and `47.0%` TF PS@All, while
`embed_only` reaches only `0.5%` TF rewrite. Subject embeddings alone are not a
usable knowledge editor in this setup.

The current default `dual` profile is a locality-protected operating point. It
gives up paraphrase transfer versus `lm_head_only` and `dual_no_anti`, but it
improves TF locality from about `88.8-88.9%` to `95.8%` and probability
locality from `71.4%` to `86.0%`.

`clamp_anti` is the main locality-preserving knob in this matrix. Removing it
makes `dual_no_anti` behave almost exactly like `lm_head_only`: better rewrite
and PS@All, much worse locality.

`clamp_eos` does not materially improve the single-edit n=200 result. `dual`
and `dual_no_eos` are effectively identical on headline metrics, while
`dual_no_eos` avoids changing the global EOS row. This supports the roadmap
direction to move EOS control out of the persistent default, but the default
profile should be changed only after a follow-up random-seed check.
