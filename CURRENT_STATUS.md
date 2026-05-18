# Current Status - 2026-05-18

This is the short source of truth for the repository.

Detailed docs:

- `docs/PATH_A_MEMORY.md`
- `docs/PATH_B_WEIGHT_EDITING.md`
- `docs/EASYEDIT_PROTOCOL.md`
- `docs/CLAIMS_AND_EVIDENCE.md`
- `docs/ROADMAP_REALISTIC.md`

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

Best balanced n=50 sequential run:

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing | 73.0% | 21.0% | 20.0% | 25.4% |
| Probability compare | 86.0% | 62.0% | 61.0% | 61.4% |

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
96 passed, 13 skipped
```

The skipped tests are Gemma E2E tests when the installed Transformers build does
not support the local `gemma4` checkpoint architecture.

## Safe Claims

- AGIM WAL has strong n=50 single-edit EasyEdit-compatible results on
  Llama-3.1-8B-Instruct.
- AGIM has measured rollback/audit/non-target-diff diagnostics for its WAL edit
  paths.
- Historical local 1000-fact diagnostics are useful for stress testing but not
  official EasyEdit evidence.

## Not Safe Claims

- AGIM is number one on EasyEdit.
- AGIM has solved lifelong or sequential knowledge editing.
- The historical 1000-fact local result proves official EasyEdit performance.
