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

Single-edit n=50 baseline is promising:

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing | 100.0% | 71.0% | 67.0% | 58.4% |
| Probability compare | 100.0% | 88.0% | 89.0% | 37.4% |

`--use-positive-prompts` improves single-edit PS@All to 95.0% under
teacher-forcing, but drops teacher-forcing locality to 45.2%.

`--projection-mode orthogonal` is implemented as a protected-subspace ablation,
but the n=50 sequential run is worse on exact-token locality than the tuned
sequential baseline.

Sequential n=50 remains weak:

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing | 73.0% | 21.0% | 20.0% | 25.4% |
| Probability compare | 86.0% | 62.0% | 61.0% | 61.4% |

## Known Bottlenecks

- Rephrase/generalization drops sharply in sequential runs.
- Locality is weak even in the best current profile.
- Global EOS/anti rows help some single-edit behavior but hurt sequential
  exact-token results when accumulated.

## Next Method Work

1. Sequential side memory: isolate edits instead of merging every edit into the
   same shared rows.
2. Relation sharding: separate edit subspaces by `relation_id`.
3. Constrained row updates or MEMIT/ROME-style layer edits for locality.
4. Batch consolidation: periodically solve constrained updates for stable edits.
5. Tune positive-prompt weight only after locality-preserving constraints exist.

## Safe Claim

```text
AGIM WAL has strong internal n=50 EasyEdit-compatible single-edit results on
Llama-3.1-8B-Instruct, while sequential editing and locality remain open.
```

## Unsafe Claim

```text
AGIM has solved lifelong editing or is number one on EasyEdit.
```
