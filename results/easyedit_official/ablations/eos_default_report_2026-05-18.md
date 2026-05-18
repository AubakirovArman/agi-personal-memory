# EOS Default Report - 2026-05-18

This report validates removing the global EOS row from the persistent
single-edit default.

Model: `meta-llama/Llama-3.1-8B-Instruct`

Runner: `agim.eval.easyedit_official_runner`

Code commit in artifacts: `206adf4f10721a3f2c89f9dfd8d332f453364a5d`

Dataset: CounterFact random n=200, seeds `42`, `43`, and `44`

Common settings: `target_token_mode=contextual`, `clamp_lm=0.20`,
`clamp_embed=0.06`, `clamp_eos=0`, `clamp_anti=0.06`,
`nt_sample_size=32`.

## Artifacts

| Seed | Main artifact | Dry run | Failures-only |
| ---: | --- | --- | --- |
| 42 | `eos_default_random_200_seed_42_no_eos.json` | `eos_default_random_200_seed_42_no_eos.dry_run.json` | `eos_default_random_200_seed_42_no_eos.failures.json` |
| 43 | `eos_default_random_200_seed_43_no_eos.json` | `eos_default_random_200_seed_43_no_eos.dry_run.json` | `eos_default_random_200_seed_43_no_eos.failures.json` |
| 44 | `eos_default_random_200_seed_44_no_eos.json` | `eos_default_random_200_seed_44_no_eos.dry_run.json` | `eos_default_random_200_seed_44_no_eos.failures.json` |

## Metrics

| Seed | TF rewrite | TF rephrase | TF PS@All | TF locality | CTX rewrite | CTX PS@All | Prob rewrite | Prob PS@All | Prob locality | EOS changed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 96.5% | 28.5% | 27.0% | 95.8% | 96.0% | 26.5% | 99.5% | 44.5% | 86.0% | 0.0% |
| 43 | 89.0% | 20.8% | 22.2% | 96.8% | 89.0% | 21.5% | 94.5% | 37.8% | 88.6% | 0.0% |
| 44 | 93.0% | 26.0% | 27.5% | 96.5% | 93.0% | 26.5% | 98.0% | 46.2% | 85.2% | 0.0% |
| Mean | 92.8% | 25.1% | 25.6% | 96.4% | 92.7% | 24.8% | 97.3% | 42.8% | 86.6% | 0.0% |

Comparison to old random-200 default mean:

| Profile | TF rewrite | TF PS@All | TF locality | Prob PS@All | Prob locality |
| --- | ---: | ---: | ---: | ---: | ---: |
| Old default, EOS enabled | 93.2% | 25.6% | 96.4% | 43.0% | 86.6% |
| New no-EOS default candidate | 92.8% | 25.6% | 96.4% | 42.8% | 86.6% |

## Readout

Removing the global EOS row does not materially change random-200 single-edit
quality for the current locality-protected profile, and it eliminates a global
control-row mutation from the persistent default.

The repository default is therefore changed to `clamp_eos=0.0`. Historical
artifacts that used `clamp_eos=0.16` remain valid audit artifacts, but new
default runs should report `EOS_changed=0%` unless an ablation explicitly
enables EOS control.
