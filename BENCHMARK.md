# Benchmarks And Claim Boundaries

This repository contains multiple benchmark families. The important distinction
is whether a run is evaluated with real EasyEdit-compatible metrics, an older
AGIM-local CounterFact protocol, or a retrieval-memory test.

Short current status: [CURRENT_STATUS.md](CURRENT_STATUS.md).

## Current Source Of Truth

For EasyEdit-style weight-editing claims, use only:

- Runner: `src/agim/eval/easyedit_official_runner.py`
- Artifacts: `results/easyedit_official/`
- Summary: `results/easyedit_official/current/easyedit_agim_status_2026-05-18.md`

The older `results/local_protocol/official_eval_*.json` artifacts are legacy
local diagnostics. They are preserved for audit, but they are not official
EasyEdit results.

## Real EasyEdit-Compatible Results

Model: `meta-llama/Llama-3.1-8B-Instruct`

Dataset: CounterFact from `https://rome.baulab.info/data/dsets/counterfact.json`;
selection varies by artifact.

Evaluation path: AGIM edit implementation + local EasyEdit evaluation functions.

### Single Edit, Rollback After Each Fact

Artifact:
`results/easyedit_official/current/easyedit_official_50_first42_psall_baseline.json`

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing | 100.0% | 71.0% | 67.0% | 58.4% |
| Contextual generation | 100.0% | 70.0% | 66.0% | n/a |
| Probability compare | 100.0% | 88.0% | 89.0% | 37.4% |
| Non-edited lm_head row diff | 0.0 |  |  |  |
| Non-edited embed row diff | 0.0 |  |  |  |

Readout: strong single-edit continuation editor, but locality is not solved.

Positive-prompt ablation:
`results/easyedit_official/ablations/easyedit_official_50_first42_psall_positive_prompts.json`

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing | 100.0% | 96.0% | 95.0% | 45.2% |
| Contextual generation | 100.0% | 96.0% | 95.0% | n/a |
| Probability compare | 100.0% | 96.0% | 98.0% | 25.2% |

Readout: multi-positive keys are a strong paraphrase/PS@All ablation, but they
trade off locality and therefore are not the default headline profile.

Token-mode matrix on the same random n=200 seed-42 facts:
`results/easyedit_official/ablations/token_mode_matrix_report_2026-05-18.md`

| Target token mode | TF rewrite | TF PS@All | TF locality | Vanilla rewrite | CTX rewrite | Prob locality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `standalone` | 0.2% | 0.5% | 99.6% | 57.5% | 0.0% | 88.5% |
| `contextual` | 96.0% | 27.0% | 95.9% | 0.0% | 96.0% | 86.0% |
| `both` | 97.5% | 28.2% | 94.8% | 0.5% | 97.5% | 84.4% |

Readout: `contextual` is the current correct default for continuation-aligned
AGIM WAL claims on Llama. `standalone` mostly tests a different token target,
and `both` should stay an ablation until the primary-sequence handling is fixed.

Component ablation on the same random n=200 seed-42 facts:
`results/easyedit_official/ablations/component_ablation_report_2026-05-18.md`

| Ablation | TF rewrite | TF PS@All | TF locality | Prob PS@All | Prob locality |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lm_head_only` | 100.0% | 47.0% | 88.9% | 68.5% | 71.4% |
| `embed_only` | 0.5% | 0.9% | 99.9% | 8.5% | 88.5% |
| `dual` | 96.5% | 26.8% | 95.8% | 44.2% | 86.0% |
| `dual_no_eos` | 96.5% | 27.0% | 95.8% | 44.8% | 85.9% |
| `dual_no_anti` | 100.0% | 46.5% | 88.8% | 68.0% | 71.4% |
| `dual_no_eos_anti` | 100.0% | 46.5% | 88.8% | 68.0% | 71.4% |

Readout: the current default is not the highest-PS setting; it is a
locality-protected profile. `clamp_anti` accounts for most of that locality
protection. `clamp_eos` has little single-edit value here.

Random-seed validation of the current default single-edit profile:
`results/easyedit_official/current/random_50_report_2026-05-18.md`

| Seed | TF rewrite | TF rephrase | TF PS@All | TF locality | Prob PS@All | Prob locality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 98.0% | 30.0% | 25.0% | 97.2% | 52.0% | 87.4% |
| 43 | 86.0% | 18.0% | 22.0% | 96.8% | 38.0% | 88.4% |
| 44 | 94.0% | 26.0% | 23.0% | 97.4% | 43.0% | 87.6% |
| Mean | 92.7% | 24.7% | 23.3% | 97.1% | 44.3% | 87.8% |

Readout: the default locality-protected profile is stable for exact rewrite and
teacher-forcing locality across random samples, but paraphrase transfer remains
weak. Do not mix this operating point with the first-50 PS@All baseline without
stating the hyperparameter/profile difference.

Random-200 validation of the same profile:
`results/easyedit_official/current/random_200_report_2026-05-18.md`

| Seed | TF rewrite | TF rephrase | TF PS@All | TF locality | Prob PS@All | Prob locality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 97.0% | 28.5% | 27.0% | 95.9% | 44.5% | 85.9% |
| 43 | 89.0% | 20.8% | 22.2% | 96.9% | 38.0% | 88.6% |
| 44 | 93.5% | 26.5% | 27.5% | 96.4% | 46.5% | 85.2% |
| Mean | 93.2% | 25.3% | 25.6% | 96.4% | 43.0% | 86.6% |

Random-1000 validation of the same profile:
`results/easyedit_official/current/random_1000_report_2026-05-18.md`

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing seed 42 | 94.5% | 23.8% | 23.5% | 96.4% |
| Contextual generation seed 42 | 94.2% | 23.0% | 22.7% | n/a |
| Probability compare seed 42 | 97.2% | 40.9% | 41.8% | 86.5% |

Readout: the random n=1000 sample validates that the first-1000 scale result
is not a selection-only artifact. Rewrite and locality remain stable; PS@All is
still weak.

First-1000 scale check of the current default single-edit profile:
`results/easyedit_official/current/easyedit_1000_first_default_report_2026-05-18.md`

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing | 91.1% | 25.4% | 24.7% | 96.2% |
| Contextual generation | 91.0% | 24.8% | 24.1% | n/a |
| Probability compare | 96.3% | 43.5% | 43.6% | 87.5% |

Readout: this is the current correct EasyEdit-compatible n=1000 scale check,
not the legacy local 1000-fact protocol. It validates rewrite/locality at scale
for the default profile, while confirming weak paraphrase transfer.

### Sequential Editing, Tuned Projection

Best balanced artifact:
`results/easyedit_official/sequential/easyedit_official_50_first42_psall_seq_lm015_negx05_noeosanti_retention.json`

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing | 73.0% | 21.0% | 20.0% | 25.4% |
| Contextual generation | 70.0% | 20.0% | 19.0% | n/a |
| Probability compare | 86.0% | 62.0% | 61.0% | 61.4% |

Sequential retention:

| Checkpoint | TF rewrite | TF rephrase | TF PS@All | TF locality |
| --- | ---: | ---: | ---: | ---: |
| after 1 | 100.0% | 0.0% | 0.0% | 60.0% |
| after 10 | 100.0% | 50.0% | 50.0% | 59.0% |
| after 50 | 73.0% | 21.0% | 20.0% | 25.4% |

Random-seed sequential validation:
`results/easyedit_official/sequential/sequential_random_50_report_2026-05-18.md`

| Seed | TF rewrite | TF rephrase | TF PS@All | TF locality | Prob PS@All | Prob locality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 84.0% | 30.0% | 27.0% | 32.0% | 71.0% | 66.6% |
| 43 | 68.0% | 18.0% | 20.0% | 19.8% | 64.0% | 65.6% |
| 44 | 84.0% | 32.0% | 29.0% | 50.0% | 62.0% | 63.0% |
| Mean | 78.7% | 26.7% | 25.3% | 33.9% | 65.7% | 65.1% |

The same runs average `TF rewrite=100.0%` and `TF locality=83.0%` after 10
accumulated edits, then fall to `TF rewrite=78.7%` and `TF locality=33.9%`
after 50 edits. This is the clearest current evidence for sequential
degradation.

Sequential positive-prompt ablation:
`results/easyedit_official/sequential/easyedit_official_50_first42_psall_seq_lm015_negx05_noeosanti_positive_prompts_retention.json`

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing | 47.0% | 31.0% | 29.5% | 16.2% |
| Probability compare | 84.0% | 78.0% | 79.0% | 56.0% |

Readout: positive prompts improve sequential paraphrase/PS@All, but exact
rewrite and locality worsen. Sequential interference remains the main blocker.

Sequential orthogonal-projection ablation:
`results/easyedit_official/sequential/easyedit_official_50_first42_psall_seq_lm015_negx05_orthogonal_noeosanti_retention.json`

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing | 58.0% | 25.0% | 25.0% | 6.6% |
| Probability compare | 90.0% | 82.0% | 81.0% | 30.4% |

Readout: orthogonal protected-subspace projection is available as an opt-in
method knob, but this n=50 run is worse on exact-token locality. It should not
be treated as a solved locality fix.

Best exact-token locality artifact:
`results/easyedit_official/sequential/easyedit_official_50_contextual_neg4x05_seq_lm012_noeosanti.json`

| Metric group | Rewrite | Rephrase | Locality |
| --- | ---: | ---: | ---: |
| Teacher-forcing | 61.0% | 21.0% | 35.0% |
| Contextual generation | 60.0% | 20.0% | n/a |
| Probability compare | 82.0% | 58.0% | 65.6% |

Readout: projection tuning improved sequential behavior, but exact-token
locality/rephrase remain too weak for a strong EasyEdit or lifelong-editing
claim.

## Legacy Local Protocol Result

Artifact:
`results/local_protocol/official_eval_1000.json`

Historical reported values:

| Metric | Value |
| --- | ---: |
| `EasyEdit.ES` | 91.4% |
| `EasyEdit.PS` | 35.75% |
| `EasyEdit.NS_absence` | 76.2% |
| `EasyEdit.Composite` | 67.78% |
| `rep_rate` | 0.0% |
| `NT` | 0.00025464 |

Why this is not the current EasyEdit claim:

- It was produced by an AGIM-local evaluator, not
  `agim.eval.easyedit_official_runner`.
- `NS_absence` checks whether the new target is absent in neighbor outputs;
  EasyEdit locality checks whether neighbor outputs remain unchanged pre/post.
- The composite score is a local aggregate, not an official EasyEdit leaderboard
  metric.
- The artifact has no command/git metadata, unlike the current official-compatible
  artifacts.

Use this only as a historical local stress test.

## Memory/Retrieval Results

Artifacts under `results/memory_retrieval/` measure Path A memory behavior:
JSON/key-value storage, retrieval, SQuAD recall, and LoCoMo-style retrieval.

These are valid for memory-layer claims, but they are not weight-editing results
and should not be compared with EasyEdit, ROME, MEMIT, or AlphaEdit.

## Safe Claims

```text
On internal EasyEdit-compatible CounterFact single-edit runs for
Llama-3.1-8B-Instruct, AGIM WAL dual-layer reaches 100% teacher-forcing rewrite
and 71% teacher-forcing rephrase at n=50, with measured zero non-edited
lm_head/embed row diff.
```

```text
On a historical AGIM-local 1000-fact CounterFact diagnostic, the legacy evaluator
reported ES=91.4%, NS_absence=76.2%, Composite=67.8%, 0% repetition, and NT≈0.
This is not the current official-compatible EasyEdit result.
```

## Unsafe Claims

```text
AGIM is #1 on EasyEdit.
AGIM beats AlphaEdit/MEMIT on official EasyEdit.
The 1000-fact local result proves official EasyEdit performance.
AGIM has solved lifelong/sequential editing.
```
