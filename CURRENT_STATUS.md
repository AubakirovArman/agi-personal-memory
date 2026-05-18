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

Best n=50 single-edit run on `meta-llama/Llama-3.1-8B-Instruct`:

| Metric group | Rewrite | Rephrase | Locality |
| --- | ---: | ---: | ---: |
| Teacher-forcing | 100.0% | 71.0% | 58.4% |
| Contextual generation | 100.0% | 70.0% | n/a |
| Probability compare | 100.0% | 88.0% | 37.6% |

Best balanced n=50 sequential run:

| Metric group | Rewrite | Rephrase | Locality |
| --- | ---: | ---: | ---: |
| Teacher-forcing | 71.0% | 21.0% | 25.4% |
| Probability compare | 86.0% | 62.0% | 61.2% |

Interpretation: single-edit is strong; sequential editing and exact-token
locality are not solved.

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
88 passed, 13 skipped
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
