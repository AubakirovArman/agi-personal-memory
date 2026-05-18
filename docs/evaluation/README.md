# Evaluation Protocols

AGIM currently has three different evaluation families. Keeping them separate is
mandatory for honest reporting.

Related docs:

- `../PATH_A_MEMORY.md`
- `../PATH_B_WEIGHT_EDITING.md`
- `../EASYEDIT_PROTOCOL.md`
- `../CLAIMS_AND_EVIDENCE.md`
- `../ROADMAP_REALISTIC.md`

## 1. Real EasyEdit-Compatible CounterFact

Source of truth:

- Runner: `src/agim/eval/easyedit_official_runner.py`
- Artifacts: `results/easyedit_official/`
- Summary: `results/easyedit_official/current/easyedit_agim_status_2026-05-18.md`

Current single-edit n=50 baseline on `meta-llama/Llama-3.1-8B-Instruct`:

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing | 100.0% | 71.0% | 67.0% | 58.4% |
| Contextual generation | 100.0% | 70.0% | 66.0% | n/a |
| Probability compare | 100.0% | 88.0% | 89.0% | 37.4% |

Positive-prompt ablation improves PS@All at a locality cost:

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing | 100.0% | 96.0% | 95.0% | 45.2% |
| Probability compare | 100.0% | 96.0% | 98.0% | 25.2% |

Random-seed n=50 validation for the current default profile:

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing mean over seeds 42/43/44 | 92.7% | 24.7% | 23.3% | 97.1% |
| Contextual generation mean over seeds 42/43/44 | 92.0% | n/a | 22.7% | n/a |
| Probability compare mean over seeds 42/43/44 | 97.3% | n/a | 44.3% | 87.8% |

Readout: the current default profile keeps locality high on random samples, but
paraphrase transfer is weak. The detailed report is
`results/easyedit_official/current/random_50_report_2026-05-18.md`.

Current sequential n=50 tuned profile (`clamp_lm=0.15`,
`neg_projection_strength=0.50`, no EOS/anti rows):

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing | 73.0% | 21.0% | 20.0% | 25.4% |
| Contextual generation | 70.0% | 20.0% | 19.0% | n/a |
| Probability compare | 86.0% | 62.0% | 61.0% | 61.4% |

Random-seed validation of the same sequential tuned profile:

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing mean over seeds 42/43/44 | 78.7% | 26.7% | 25.3% | 33.9% |
| Probability compare mean over seeds 42/43/44 | 96.0% | n/a | 65.7% | 65.1% |

Interpretation: single-edit is strong; sequential/locality is still weak and
does not support a lifelong-editing claim.

The fresh orthogonal projection ablation reports `TF rewrite=58.0%`,
`TF PS@All=25.0%`, and `TF locality=6.6%`; stricter protected-key projection
alone is therefore not the current locality fix.

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
