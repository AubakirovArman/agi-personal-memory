# Exact Additive Update Report - 2026-05-18

Model: `meta-llama/Llama-3.1-8B-Instruct`

Runner: `agim.eval.easyedit_official_runner`

Code commit in exact-additive artifact: `d767dfce92f3fa73c22ead688bdbbbbf789057b7`

Artifact schema: `easyedit_official.v2`

Method profile: `single_exact_additive`

Dataset: CounterFact from `https://rome.baulab.info/data/dsets/counterfact.json`

Selection: random 200 records with seed `42`

Selected case-id hash: `2517f71c21db2952484dd83aa87afce592a2ceba1c81b8bacffa943b7a26439e`

All runs use `target_token_mode=contextual` and the default dual clamps:
`clamp_lm=0.20`, `clamp_embed=0.06`, `clamp_eos=0.16`,
`clamp_anti=0.06`.

## Command

```text
AGIM_EASYEDIT_ROOT=/mnt/hf_model_weights/arman/3bit/sites/EasyEdit PYTHONPATH=src \
python -m agim.eval.easyedit_official_runner \
  --n 200 --sample-policy random --seed 42 --device cuda:2 \
  --target-token-mode contextual \
  --clamp_lm 0.20 --clamp_embed 0.06 --clamp_eos 0.16 --clamp_anti 0.06 \
  --no-wal-encode-updates \
  --output results/easyedit_official/ablations/exact_additive_random_200_seed_42_dual.json \
  --save-failures-only
```

## Artifacts

- `exact_additive_random_200_seed_42_dual.json`
- `exact_additive_random_200_seed_42_dual.dry_run.json`
- `exact_additive_random_200_seed_42_dual.failures.json`

Comparison baseline:

- `component_random_200_seed_42_dual.json`

The baseline artifact was produced before `wal_encode_updates` was added to the
payload, but it uses the normal WAL-encoded row update path.

## Metrics

| Update path | TF rewrite | TF rephrase | TF PS@All | TF locality | CTX rewrite | CTX PS@All | Prob rewrite | Prob PS@All | Prob locality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WAL-encoded dual | 96.5% | 28.5% | 26.8% | 95.8% | 96.5% | 26.2% | 99.5% | 44.2% | 86.0% |
| Exact-additive dual | 97.5% | 28.5% | 27.0% | 95.8% | 97.5% | 26.5% | 99.5% | 44.5% | 85.9% |

NT diagnostics for exact-additive dual:

| Metric | Value |
| --- | ---: |
| `lm_head_non_edited_max` | 0.0 |
| `embed_non_edited_max` | 0.0 |
| `edited_lm_rows_avg` | 2.010 |
| `edited_embed_rows_avg` | 4.230 |
| `eos_row_changed_rate` | 100.0% |

Runtime:

| Update path | Time | Seconds/edit |
| --- | ---: | ---: |
| WAL-encoded dual | 308.06s | 1.54 |
| Exact-additive dual | 296.56s | 1.48 |

## Failure Triage

The exact-additive failures-only artifact uses `failure_families=tf,ctx_gen,prob`.

| Failure mode | Count |
| --- | ---: |
| `tf_ps_all` | 173 |
| `tf_rephrase` | 144 |
| `prob_locality` | 120 |
| `tf_locality` | 56 |
| `tf_rewrite` | 5 |
| `ctx_gen_rewrite` | 5 |

## Readout

Skipping WAL re-encoding produces only a small gain on this n=200 sample:
`TF rewrite` improves by 1 point and `TF PS@All` by 0.25 point, while locality
is effectively unchanged.

This means WAL quantization is not the main current bottleneck for the default
single-edit profile. The larger bottleneck remains the editing locus and
control policy: output-row edits, subject embedding edits, and global
EOS/anti-control rows. Future quality work should prioritize constrained
updates, non-global control policies, internal-layer factual patches, or
side-slot/routed serving over further WAL reconstruction tuning.
