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

New EasyEdit-compatible artifacts also record deterministic NT sampled row ids
for both `lm_head` and embeddings, making non-target drift checks auditable
beyond a single max-diff scalar.

New artifacts also emit edited-row delta L2 norm metrics under `NT`, so patch
growth can be monitored without loading full row tensors.

`PatchArtifact` now has a `NormBudgetPolicy` foundation. It can return a
structured `allow_commit` / `no_commit` decision for row count, patch delta
norm, max row delta norm, and mean row delta norm. The EasyEdit runner also has
optional runtime budget flags; when a proposal exceeds a limit, the runner
rolls it back before post-edit evaluation and records `edit_status=no_commit`.

`conflict_summary()` now reports row overlaps plus metadata overlaps for subject
token ids, target token ids, control row ids, protected basis ids, same relation,
and same subject. This gives patch review a concrete conflict-risk surface
before patches are applied together.

`RuntimeSparseOverlay` provides the first frozen-base overlay foundation for
Path B: lm_head deltas are applied as logit corrections and embed deltas are
applied through embedding hooks, without mutating the model weights.
`add_patch_artifact()` can load `PatchArtifact` rows into the overlay. It is
not yet the default EasyEdit backend.

`WALDualLayerEditor` mutable state is now namespaced by `--state-namespace`.
History keys and relation-protected banks can be isolated by tenant, batch, or
workflow while keeping the old `default` namespace behavior.

`SideSlotMemory` adds a routed side-memory foundation for Path B patches. It
stores `PatchArtifact` objects as slots and builds a `RuntimeSparseOverlay` for
matching subject/relation requests, keeping the base model frozen.
Relation sharding is lifted into explicit patch-slot isolation via
`relation_slot_id`, relation-slot summaries, and enable/disable controls for a
whole relation slot.

The EasyEdit runner now has `--edit-backend side_slot` for sequential runs.
Each proposal is converted into a `PatchArtifact`, rolled back from the base
weights, and evaluated through a routed runtime overlay.

The first seed-42 side-slot sequential EasyEdit run reaches `TF rewrite=100.0%`,
`TF PS@All=46.0%`, and `TF locality=88.9%` after 50 accumulated edits. This is
a strong architectural signal, but it still needs random-seed validation before
being promoted to a headline sequential profile.

`WALRomeEditor` adds the first located internal-layer backend. It applies
sparse ROME-style row updates to a selected `mlp.down_proj` layer, records FFN
row backups for rollback, and exposes FFN non-target/budget diagnostics through
the same EasyEdit runner via `--edit-backend wal_rome`. A first n=5 smoke
artifact preserved locality but got `TF rewrite=0.0%`, so this is an
implemented backend foundation that still needs tuning before any official
Llama-3.1-8B n=50 headline result.

`WALMemitBatchEditor` adds the first batch-consolidation foundation for Path B
patches. It merges multiple verified `PatchArtifact` row deltas into one
auditable batch artifact, records overlapping-row conflicts, applies the
consolidated rows, and rolls them back as one unit. It is not yet wired into
the official EasyEdit runner comparison loop.

The EasyEdit runner now has a `--compare-backends` matrix mode. It can run the
same selected CounterFact slice through runnable backends such as `dual_row`,
`wal_rome`, and sequential `side_slot`, while recording `wal_memit` as skipped
because it is an offline consolidation backend. The random-50 matrix confirms
the current ordering: `dual_row` reached `TF rewrite=98.0%`, `TF locality=97.2%`,
while `wal_rome` reached `TF rewrite=0.0%`, `TF locality=98.2%`.

`agim.eval.ripple_diagnostic` adds a RippleEdits-style post-hoc diagnostic for
EasyEdit artifacts. It reports direct rewrite success against related/locality
preservation and relation-level break rates. This is not an official
RippleEdits dataset result; it is the first local related-fact diagnostic. It
also provides JSON dataset-adapter and scored-output payloads for preparing
external RippleEdits-style runs.

`agim.eval.mquake_diagnostic` adds a MQuAKE-style post-hoc diagnostic for
artifacts with EasyEdit/KnowEdit portability fields. It reports direct rewrite,
multi-hop portability, composite accuracy, and direct-success/hop-failure rate.
It also provides a JSON dataset adapter payload for multi-edit + multi-hop
MQuAKE records. A first-50 adapter artifact exists under
`results/external_benchmark_adapters/`, and the module can score direct/hop
model outputs against that adapter schema. This is not an official MQuAKE
dataset score until backed by a documented model-output run.

`agim.eval.raw_text_edit_pipeline` adds the first AKEW-style raw update path:
short text or JSON updates are converted into structured EasyEdit-style
`requested_rewrite` proposals and draft `PatchArtifact` objects that can enter
`PatchService` proposal/simulation flows. `PatchService.materialize_patch`
can now apply an editor, capture row deltas, roll the live model back, and keep
the patch in `materialized` status for canary/approval gates. The module also
provides a scored-output layer for documented model-output payloads. This is
still not an official AKEW benchmark until backed by a tracked external
raw-text run.

`agim.eval.product_diagnostic` adds a KnowEdit-inspired product diagnostic over
EasyEdit artifacts. It combines rewrite, PS@All, locality, and portability when
available into a product-facing composite. This is not an external
KnowEdit/UniEdit/ScEdit leaderboard result. It also provides
KnowEdit/UniEdit-style dataset-adapter and scored-output payloads for preparing
product benchmark runs.

`PatchService` adds the first in-process PatchService API for Path B patch
lifecycle. It supports `propose_patch`, `simulate_patch`, `run_canaries`,
`approve_patch`, `apply_patch`, `rollback_patch`, `inspect_patch`, and
`diff_patch` over `PatchArtifact` objects. This is not yet an HTTP service.

`PatchGovernance` adds the first governance layer for Path B patches:
signature metadata, signature verification, per-patch ACL checks, and an
append-only audit chain for patch lifecycle events.

`agim.integrations.easyedit_agimwal` adds the first external EasyEdit adapter
package. It exposes `AGIMWALHyperParams` and `apply_agimwal_to_model` in the
same shape as standard EasyEdit methods, so an EasyEdit checkout can register
AGIMWAL in `ALG_DICT` without copying the AGIM implementation into the
EasyEdit tree. This adapter has unit coverage, but it is not an upstream
EasyEdit PR.

The persistent default no longer edits the global EOS row (`clamp_eos=0.0`).
The random-200 no-EOS seed check matched the prior default quality while
reporting `EOS_changed=0%`.

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
- `--positive-constraint-mode ridge` restores single-edit positive-prompt
  locality on n=50 seed 42, but also drops exact rewrite to 80.0%, so it needs
  strength/weight tuning before any default promotion.
- `--relation-protected-mode accumulate/preload` improves sequential locality
  on seed 42, but the rewrite/PS@All cost confirms that projection-only
  relation isolation is still not enough.

## Next Method Work

1. Sequential side memory: isolate edits instead of merging every edit into the
   same shared rows.
2. Norm budgets and no-commit rules: reject edits that exceed shared-row or
   protected-subspace drift budgets.
3. Run and tune the new `wal_rome` backend on n=50 before any larger scale
   claim.
4. Promote backend comparison from runner support to tracked n=50 artifacts.
5. Replace the post-hoc Ripple-style diagnostic with an official RippleEdits
   dataset adapter.
6. Replace the post-hoc MQuAKE-style diagnostic with an official MQuAKE dataset
   adapter and tracked artifact.
7. Replace raw-text proposal/scoring scaffolding with a tracked external
   AKEW-style model-output run.
8. Replace product diagnostics with a tracked external KnowEdit, UniEdit,
   ScEdit, or MLaKE run.
9. Expose `PatchService` through a stable HTTP/SDK API with persistence.
10. Connect the EasyEdit adapter to persistent patch governance and publish an
    upstreamable EasyEdit PR/tech report package.

## Safe Claim

```text
AGIM WAL has strong internal n=50 EasyEdit-compatible single-edit results on
Llama-3.1-8B-Instruct, while sequential editing and locality remain open.
```

## Unsafe Claim

```text
AGIM has solved lifelong editing or is number one on EasyEdit.
```
