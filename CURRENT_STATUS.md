# Current Status - 2026-05-18

This is the short source of truth for the repository.

Detailed docs:

- `docs/PATH_A_MEMORY.md`
- `docs/PATH_B_WEIGHT_EDITING.md`
- `docs/EASYEDIT_PROTOCOL.md`
- `docs/CLAIMS_AND_EVIDENCE.md`
- `docs/ROADMAP_REALISTIC.md`
- `docs/PATH_B_PRODUCTIZATION_PLAN.md`

## What This Project Currently Is

AGI Personal Memory is an experimental verified-memory substrate with two
separate paths:

| Path | What it does | Current maturity |
| --- | --- | --- |
| Path A: retrieval memory | Stores and retrieves facts through AGIM memory stores | Works as a memory layer |
| Path B: WAL weight editing | Applies reversible model weight edits for research experiments | Promising single-edit results; sequential/locality still weak |

## Current EasyEdit-Compatible Result

Use `results/easyedit_official/current/` for current single-edit
EasyEdit-style claims and `results/easyedit_official/sequential/` only for
explicitly marked sequential/weakness claims.

Best n=50 single-edit baseline on `meta-llama/Llama-3.1-8B-Instruct`:

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing | 100.0% | 71.0% | 67.0% | 58.4% |
| Contextual generation | 100.0% | 70.0% | 66.0% | n/a |
| Probability compare | 100.0% | 88.0% | 89.0% | 37.4% |

Positive-prompt ablation improves single-edit PS@All but hurts locality:

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing | 100.0% | 96.0% | 95.0% | 45.2% |
| Probability compare | 100.0% | 96.0% | 98.0% | 25.2% |

Random-seed n=50 validation of the current default single-edit profile:

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing mean over seeds 42/43/44 | 92.7% | 24.7% | 23.3% | 97.1% |
| Contextual generation mean over seeds 42/43/44 | 92.0% | n/a | 22.7% | n/a |
| Probability compare mean over seeds 42/43/44 | 97.3% | n/a | 44.3% | 87.8% |

Readout: the current default is stable for rewrite and locality, but weak for
paraphrase transfer. See
`results/easyedit_official/current/random_50_report_2026-05-18.md`.

Random-seed n=200 validation of the same profile:

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing mean over seeds 42/43/44 | 93.2% | 25.3% | 25.6% | 96.4% |
| Contextual generation mean over seeds 42/43/44 | 93.0% | n/a | 24.8% | n/a |
| Probability compare mean over seeds 42/43/44 | 97.3% | n/a | 43.0% | 86.6% |

See `results/easyedit_official/current/random_200_report_2026-05-18.md`.

Random-seed n=1000 validation of the same profile:

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing seed 42 | 94.5% | 23.8% | 23.5% | 96.4% |
| Contextual generation seed 42 | 94.2% | 23.0% | 22.7% | n/a |
| Probability compare seed 42 | 97.2% | 40.9% | 41.8% | 86.5% |

See `results/easyedit_official/current/random_1000_report_2026-05-18.md`.

Token-mode ablation on the same random n=200 seed-42 facts:

| Target token mode | TF rewrite | TF PS@All | TF locality | Vanilla rewrite | CTX rewrite | Prob locality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `standalone` | 0.2% | 0.5% | 99.6% | 57.5% | 0.0% | 88.5% |
| `contextual` | 96.0% | 27.0% | 95.9% | 0.0% | 96.0% | 86.0% |
| `both` | 97.5% | 28.2% | 94.8% | 0.5% | 97.5% | 84.4% |

Readout: `contextual` remains the default profile. `standalone` measures a
different target-token alignment, while `both` still needs the planned primary
sequence fix. See
`results/easyedit_official/ablations/token_mode_matrix_report_2026-05-18.md`.

Component ablation on the same random n=200 seed-42 facts:

| Ablation | TF rewrite | TF PS@All | TF locality | Prob PS@All | Prob locality |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lm_head_only` | 100.0% | 47.0% | 88.9% | 68.5% | 71.4% |
| `embed_only` | 0.5% | 0.9% | 99.9% | 8.5% | 88.5% |
| `dual` | 96.5% | 26.8% | 95.8% | 44.2% | 86.0% |
| `dual_no_eos` | 96.5% | 27.0% | 95.8% | 44.8% | 85.9% |
| `dual_no_anti` | 100.0% | 46.5% | 88.8% | 68.0% | 71.4% |
| `dual_no_eos_anti` | 100.0% | 46.5% | 88.8% | 68.0% | 71.4% |

Readout: `lm_head` is the rewrite component. `clamp_anti` is the main
locality-preserving knob, while `clamp_eos` does not materially improve
single-edit n=200 metrics. See
`results/easyedit_official/ablations/component_ablation_report_2026-05-18.md`.

Official-compatible first-1000 scale check of the current default single-edit
profile:

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing | 91.1% | 25.4% | 24.7% | 96.2% |
| Contextual generation | 91.0% | 24.8% | 24.1% | n/a |
| Probability compare | 96.3% | 43.5% | 43.6% | 87.5% |

Readout: the n=1000 result confirms the default rewrite/locality profile at
scale. It does not solve paraphrase transfer. See
`results/easyedit_official/current/easyedit_1000_first_default_report_2026-05-18.md`.

Best balanced n=50 sequential run:

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing | 73.0% | 21.0% | 20.0% | 25.4% |
| Probability compare | 86.0% | 62.0% | 61.0% | 61.4% |

Sequential random-seed validation of the same tuned profile:

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing mean over seeds 42/43/44 | 78.7% | 26.7% | 25.3% | 33.9% |
| Probability compare mean over seeds 42/43/44 | 96.0% | n/a | 65.7% | 65.1% |

Retention mean: after 10 edits `TF rewrite=100.0%` and
`TF locality=83.0%`; after 50 edits `TF rewrite=78.7%` and
`TF locality=33.9%`. This confirms sequential degradation after longer edit
chains. See
`results/easyedit_official/sequential/sequential_random_50_report_2026-05-18.md`.

Interpretation: single-edit is strong; sequential editing and exact-token
locality are not solved. The current PS@All report is
`results/easyedit_official/current/easyedit_psall_report_2026-05-18.md`.

## Historical 1000-Fact Local Result

`results/local_protocol/official_eval_1000.json` reports:

```text
ES=91.4%, NS_absence=76.2%, Composite=67.8%, repetition=0%, NT≈0
```

This is a historical AGIM-local diagnostic. It is not a current official
EasyEdit result and should not be compared directly with EasyEdit leaderboard
methods.

## Test Status

Current full local suite:

```text
101 passed, 13 skipped
```

The skipped tests are Gemma E2E tests when the installed Transformers build does
not support the local `gemma4` checkpoint architecture.

## Safe Claims

- AGIM WAL has strong n=50 single-edit EasyEdit-compatible results on
  Llama-3.1-8B-Instruct.
- AGIM WAL has random-seed n=200 and n=1000 evidence that the current
  locality-protected single-edit profile stays stable for rewrite/locality.
- AGIM has measured rollback/audit/non-target-diff diagnostics for its WAL edit
  paths.
- Historical local 1000-fact diagnostics are useful for stress testing but not
  official EasyEdit evidence.

## Not Safe Claims

- AGIM is number one on EasyEdit.
- AGIM has solved lifelong or sequential knowledge editing.
- The historical 1000-fact local result proves official EasyEdit performance.
