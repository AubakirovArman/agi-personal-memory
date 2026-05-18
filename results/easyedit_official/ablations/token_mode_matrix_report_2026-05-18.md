# Token Mode Matrix Report - 2026-05-18

Model: `meta-llama/Llama-3.1-8B-Instruct`

Runner: `agim.eval.easyedit_official_runner`

Code commit in artifacts: `278562b227d52694d6cd977629f3e789a6a6edc2`

Artifact schema: `easyedit_official.v2`

Method profile: `single_loc`

Dataset: CounterFact from `https://rome.baulab.info/data/dsets/counterfact.json`

Selection: random 200 records with seed `42`

Selected case-id hash: `2517f71c21db2952484dd83aa87afce592a2ceba1c81b8bacffa943b7a26439e`

## Command Shape

```text
AGIM_EASYEDIT_ROOT=/mnt/hf_model_weights/arman/3bit/sites/EasyEdit PYTHONPATH=src \
python -m agim.eval.easyedit_official_runner \
  --n 200 --sample-policy random --seed 42 --device cuda:<2|3> \
  --target-token-mode <standalone|contextual|both> \
  --output results/easyedit_official/ablations/token_mode_random_200_seed_42_<mode>.json \
  --save-failures-only
```

## Artifacts

| Mode | Main artifact | Dry run | Failures-only |
| --- | --- | --- | --- |
| `standalone` | `token_mode_random_200_seed_42_standalone.json` | `token_mode_random_200_seed_42_standalone.dry_run.json` | `token_mode_random_200_seed_42_standalone.failures.json` |
| `contextual` | `token_mode_random_200_seed_42_contextual.json` | `token_mode_random_200_seed_42_contextual.dry_run.json` | `token_mode_random_200_seed_42_contextual.failures.json` |
| `both` | `token_mode_random_200_seed_42_both.json` | `token_mode_random_200_seed_42_both.dry_run.json` | `token_mode_random_200_seed_42_both.failures.json` |

## Metrics

| Target token mode | TF rewrite | TF rephrase | TF PS@All | TF locality | Vanilla rewrite | Vanilla PS@All | CTX rewrite | CTX PS@All | Prob rewrite | Prob PS@All | Prob locality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `standalone` | 0.2% | 0.2% | 0.5% | 99.6% | 57.5% | 4.8% | 0.0% | 0.2% | 7.0% | 8.8% | 88.5% |
| `contextual` | 96.0% | 28.5% | 27.0% | 95.9% | 0.0% | 0.0% | 96.0% | 26.5% | 99.5% | 44.5% | 86.0% |
| `both` | 97.5% | 30.5% | 28.2% | 94.8% | 0.5% | 0.3% | 97.5% | 27.8% | 99.5% | 45.5% | 84.4% |

NT diagnostics:

| Target token mode | Non-edited lm_head max | Non-edited embed max | Edited lm rows avg | Edited embed rows avg | EOS changed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `standalone` | 0.0 | 0.0 | 2.665 | 4.230 | 100.0% |
| `contextual` | 0.0 | 0.0 | 2.010 | 4.230 | 100.0% |
| `both` | 0.0 | 0.0 | 3.670 | 4.230 | 100.0% |

Runtime:

| Target token mode | Time | Seconds/edit |
| --- | ---: | ---: |
| `standalone` | 309.09s | 1.55 |
| `contextual` | 305.21s | 1.53 |
| `both` | 309.12s | 1.55 |

## Failure Triage

Failures-only artifacts use `failure_families=tf,ctx_gen,prob`.

| Target token mode | Failed cases | Main failure modes |
| --- | ---: | --- |
| `standalone` | 200 | `tf_rewrite=200`, `tf_rephrase=200`, `tf_ps_all=200`, `ctx_gen_rewrite=200`, `prob_locality=84` |
| `contextual` | 190 | `tf_ps_all=173`, `tf_rephrase=144`, `prob_locality=120`, `tf_locality=55`, `tf_rewrite=8` |
| `both` | 190 | `tf_ps_all=171`, `tf_rephrase=140`, `prob_locality=123`, `tf_locality=65`, `tf_rewrite=5` |

## Readout

This matrix confirms that target-token alignment is a first-order protocol
choice for Llama-3.1-8B-Instruct.

`standalone` improves official vanilla generation rewrite (`57.5%`) but almost
fully breaks teacher-forcing and contextual-generation metrics because the edit
is aligned to standalone target ids rather than prompt-continuation suffix ids.

`contextual` is the correct default for current AGIM WAL claims: it preserves
high teacher-forcing rewrite (`96.0%`) and contextual generation rewrite
(`96.0%`) on the same 200 facts while keeping locality close to the previously
tracked random-200 result.

`both` slightly improves rewrite and PS@All versus `contextual`, but locality
drops (`TF locality=94.8%`, `Prob locality=84.4%`) and edited lm_head rows
increase. Because the current implementation chooses the first target sequence
as primary, `both` still needs the planned primary-sequence fix before it can
become a default profile.

Follow-up: `both_primary_fixed_report_2026-05-18.md` validates the contextual
primary-sequence fix. After the fix, `both` behaves much closer to
`contextual` and remains an ablation knob rather than a default profile.
