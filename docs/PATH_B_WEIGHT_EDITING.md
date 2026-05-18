# Path B: WAL Weight Editing

Path B is the research part of AGI Personal Memory. It applies reversible
WAL-backed edits to model weights and evaluates them with CounterFact /
EasyEdit-compatible metrics.

## Current Scope

- Model used for current artifacts: `meta-llama/Llama-3.1-8B-Instruct`.
- Primary runner: `agim.eval.easyedit_official_runner`.
- Current artifacts:
  - `results/easyedit_official/current/`
  - `results/easyedit_official/sequential/`

## Current Readout

Single-edit n=50 is promising:

| Metric group | Rewrite | Rephrase | Locality |
| --- | ---: | ---: | ---: |
| Teacher-forcing | 100.0% | 71.0% | 58.4% |
| Probability compare | 100.0% | 88.0% | 37.6% |

Sequential n=50 remains weak:

| Metric group | Rewrite | Rephrase | Locality |
| --- | ---: | ---: | ---: |
| Teacher-forcing | 71.0% | 21.0% | 25.4% |
| Probability compare | 86.0% | 62.0% | 61.2% |

## Known Bottlenecks

- Rephrase/generalization drops sharply in sequential runs.
- Locality is weak even in the best current profile.
- Global EOS/anti rows help some single-edit behavior but hurt sequential
  exact-token results when accumulated.

## Next Method Work

1. Multi-positive keys: optional prompt + paraphrase key averaging is now
   available through `--use-positive-prompts`; it still needs fresh GPU runs.
2. Protected-key/null-space projection: preserve neighborhood and previous edit
   keys.
3. Sequential side memory: isolate edits instead of merging every edit into the
   same shared rows.
4. Relation sharding: separate edit subspaces by `relation_id`.
5. Batch consolidation: periodically solve constrained updates for stable edits.

## Safe Claim

```text
AGIM WAL has strong internal n=50 EasyEdit-compatible single-edit results on
Llama-3.1-8B-Instruct, while sequential editing and locality remain open.
```

## Unsafe Claim

```text
AGIM has solved lifelong editing or is number one on EasyEdit.
```
