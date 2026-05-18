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

The current default single-edit profile was also run on random seeds 42/43/44.
It averages 92.7% teacher-forcing rewrite and 97.1% teacher-forcing locality,
but only 23.3% PS@All. This validates rewrite/locality stability for that
profile, not paraphrase transfer.

The same default profile was run on the first 1000 CounterFact facts with the
EasyEdit-compatible runner. It reports 91.1% teacher-forcing rewrite, 96.2%
teacher-forcing locality, and 24.7% PS@All. This is the correct current n=1000
EasyEdit-compatible scale check, not the legacy local protocol.

The same default profile was also validated on a random n=1000 seed-42 sample.
It reports 94.5% teacher-forcing rewrite, 96.4% teacher-forcing locality, and
23.5% PS@All.

The component ablation on random n=200 seed-42 shows that `lm_head` performs
the rewrite, embeddings alone do not, and `clamp_anti` is the main
locality-preserving knob in the current default profile.

The exact-additive ablation shows that skipping WAL reconstruction produces
only a small single-edit gain. The current bottleneck is therefore more likely
the editing locus/control policy than WAL quantization.

`--projection-mode orthogonal` is implemented as a protected-subspace ablation,
but the n=50 sequential run is worse on exact-token locality than the tuned
sequential baseline.

Sequential n=50 remains weak:

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing | 73.0% | 21.0% | 20.0% | 25.4% |
| Probability compare | 86.0% | 62.0% | 61.0% | 61.4% |

Random sequential seeds 42/43/44 average 100.0% teacher-forcing rewrite and
83.0% teacher-forcing locality after 10 accumulated edits, then fall to 78.7%
rewrite and 33.9% locality after 50 edits.

## Known Bottlenecks

- Rephrase/generalization drops sharply in sequential runs.
- Locality is weak even in the best current profile.
- Global EOS/anti rows help some single-edit behavior but hurt sequential
  exact-token results when accumulated.
- The first `--history-slot-mode relation` ablation did not improve seed 42.
- `--positive-constraint-mode projected` improved sequential PS@All on seed 42
  but dropped exact rewrite too much.

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
