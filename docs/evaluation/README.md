# Evaluation Protocols

AGIM currently has three different evaluation families. Keeping them separate is
mandatory for honest reporting.

## 1. Real EasyEdit-Compatible CounterFact

Source of truth:

- Runner: `src/agim/eval/easyedit_official_runner.py`
- Artifacts: `results/easyedit_official/`
- Summary: `results/easyedit_official/easyedit_agim_status_2026-05-18.md`

Current single-edit n=50 result on `meta-llama/Llama-3.1-8B-Instruct`:

| Metric group | Rewrite | Rephrase | Locality |
| --- | ---: | ---: | ---: |
| Teacher-forcing | 100.0% | 71.0% | 58.4% |
| Contextual generation | 100.0% | 70.0% | n/a |
| Probability compare | 100.0% | 88.0% | 37.6% |

Current sequential n=50 tuned profile (`clamp_lm=0.15`,
`neg_projection_strength=0.50`, no EOS/anti rows):

| Metric group | Rewrite | Rephrase | Locality |
| --- | ---: | ---: | ---: |
| Teacher-forcing | 71.0% | 21.0% | 25.4% |
| Contextual generation | 70.0% | 20.0% | n/a |
| Probability compare | 86.0% | 62.0% | 61.2% |

Interpretation: single-edit is strong; sequential/locality is still weak and
does not support a lifelong-editing claim.

## 2. Legacy Local CounterFact Protocol

Source:

- Artifacts: `results/local_protocol/`
- Example: `results/local_protocol/official_eval_1000.json`

The historical 1000-fact diagnostic reports `ES=91.4%`,
`NS_absence=76.2%`, `Composite=67.8%`, `rep_rate=0%`, and `NT≈0`.
This is a useful local stress result, but not the same as the current
EasyEdit-compatible runner.

## 3. Memory/Retrieval Benchmarks

Source:

- Artifacts: `results/memory_retrieval/`

These runs measure storage/retrieval behavior, not weight editing. They are
valid for Path A memory claims, not for EasyEdit claims.

