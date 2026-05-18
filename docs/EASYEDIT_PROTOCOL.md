# EasyEdit-Compatible Protocol Notes

AGIM keeps the editor local, but the current runner evaluates with local
EasyEdit metric functions where possible.

## Source Of Truth

- Runner: `agim.eval.easyedit_official_runner`
- Current artifacts: `results/easyedit_official/current/`
- Sequential artifacts: `results/easyedit_official/sequential/`
- Summary: `BENCHMARK.md`

## Metric Groups

| Metric group | Meaning | Headline use |
| --- | --- | --- |
| Teacher-forcing token EM | EasyEdit-style target token equality under teacher forcing | Main internal metric |
| Official vanilla generation | EasyEdit generation equality against `tok.encode(target_new)` | Report separately |
| Contextual generation | Greedy equality against `prompt + space + target` suffix ids | Diagnostic for Llama tokenization |
| Probability compare | Checks whether `P(target_new) > P(target_true)` | Diagnostic |
| Locality | Pre/post neighbor output consistency | Main weakness today |
| `PS@All` / `rephrase_all_acc` | Rephrase aggregate across all CounterFact paraphrases | Diagnostic for cherry-pick risk |
| Sequential retention | Metrics after selected accumulated-edit checkpoints | Diagnostic for collapse timing |
| Multi-positive key ablation | Optional edit-key averaging with paraphrase prompts | Experimental method knob |
| Orthogonal projection ablation | Optional protected-subspace projection mode | Negative result so far |
| NT diff | Non-edited `lm_head`/embedding row drift | Diagnostic |

## Why Vanilla Generation Is Separate

For Llama tokenization, EasyEdit's vanilla generation target ids can differ from
the teacher-forcing continuation ids. Example: standalone `English` can tokenize
differently from the suffix in `The language is English`.

Because of this, reports must show:

```text
Teacher-forcing token EM
Official vanilla generation
Contextual generation diagnostic
Probability diagnostic
```

Do not merge those into one headline number.

The n=200 token-mode matrix confirms the practical impact:

| Target token mode | TF rewrite | Vanilla rewrite | CTX rewrite | TF locality |
| --- | ---: | ---: | ---: | ---: |
| `standalone` | 0.2% | 57.5% | 0.0% | 99.6% |
| `contextual` | 96.0% | 0.0% | 96.0% | 95.9% |
| `both` | 97.5% | 0.5% | 97.5% | 94.8% |

Detailed report:
`results/easyedit_official/ablations/token_mode_matrix_report_2026-05-18.md`.

The n=200 component ablation identifies the default tradeoff:

| Ablation | TF rewrite | TF PS@All | TF locality | Prob locality |
| --- | ---: | ---: | ---: | ---: |
| `lm_head_only` | 100.0% | 47.0% | 88.9% | 71.4% |
| `embed_only` | 0.5% | 0.9% | 99.9% | 88.5% |
| `dual` | 96.5% | 26.8% | 95.8% | 86.0% |
| `dual_no_eos` | 96.5% | 27.0% | 95.8% | 85.9% |
| `dual_no_anti` | 100.0% | 46.5% | 88.8% | 71.4% |

Detailed report:
`results/easyedit_official/ablations/component_ablation_report_2026-05-18.md`.

The exact-additive update ablation checks whether WAL reconstruction is the
main bottleneck:

| Update path | TF rewrite | TF PS@All | TF locality | Prob locality |
| --- | ---: | ---: | ---: | ---: |
| WAL-encoded dual | 96.5% | 26.8% | 95.8% | 86.0% |
| Exact-additive dual | 97.5% | 27.0% | 95.8% | 85.9% |

Detailed report:
`results/easyedit_official/ablations/exact_additive_report_2026-05-18.md`.

## Completed Random-Seed Runs

The required random n=50 single-edit presets have been run with the current
default locality-protected profile:

```text
results/easyedit_official/current/random_50_seed_42.json
results/easyedit_official/current/random_50_seed_43.json
results/easyedit_official/current/random_50_seed_44.json
```

Mean readout over seeds 42/43/44: `TF rewrite=92.7%`,
`TF PS@All=23.3%`, `TF locality=97.1%`, `Prob locality=87.8%`.

The random n=200 layer is also complete for seeds 42/43/44. Mean readout:
`TF rewrite=93.2%`, `TF PS@All=25.6%`, `TF locality=96.4%`,
`Prob locality=86.6%`.

The random n=1000 layer is complete for seed 42:
`results/easyedit_official/current/random_1000_seed_42.json`. Readout:
`TF rewrite=94.5%`, `TF PS@All=23.5%`, `TF locality=96.4%`,
`Prob locality=86.5%`.

This validates rewrite/locality stability for the default profile, while also
showing weak paraphrase transfer. Detailed reports are
`results/easyedit_official/current/random_50_report_2026-05-18.md`,
`results/easyedit_official/current/random_200_report_2026-05-18.md`, and
`results/easyedit_official/current/random_1000_report_2026-05-18.md`.

The current official-compatible first-1000 CounterFact run has also completed:

```text
results/easyedit_official/current/easyedit_official_1000_first_default.json
```

Readout: `TF rewrite=91.1%`, `TF PS@All=24.7%`, `TF locality=96.2%`,
`Prob locality=87.5%`. This is the n=1000 EasyEdit-compatible scale check for
the default single-edit profile; it is not the legacy local 1000-fact protocol.

## Required Future Runs

For sequential runs, add retention reporting:

```text
after edit 1: evaluate edits 1..1
after edit 10: evaluate edits 1..10
after edit 50: evaluate edits 1..50
```

The random-seed sequential retention check has been run for seeds 42/43/44.
Mean readout: after 10 edits `TF rewrite=100.0%` and `TF locality=83.0%`; after
50 edits `TF rewrite=78.7%` and `TF locality=33.9%`. See
`results/easyedit_official/sequential/sequential_random_50_report_2026-05-18.md`.

## Current Gaps

- `PS@All`, `metrics_by_relation_id`, and sequential retention summaries are
  emitted for new artifacts; older artifacts do not contain them.
- `--dry-run-summary` writes the selected case ids, relation counts, and prompt
  counts without loading the model or EasyEdit.
- `--save-failures-only` writes a compact failure triage artifact next to the
  main run output.
- `--failure-families` controls failure triage families. The default is
  `tf,ctx_gen,prob`, so EasyEdit standalone vanilla generation mismatch does
  not mark every case as the primary triage failure. Add `vanilla_gen` when
  specifically auditing the official vanilla generation path.
- New artifacts include `artifact_schema_version`, `method_profile_id`,
  `base_model_digest`, and `atoms_digest`.
- New NT rows include deterministic sampled row ids:
  `lm_head_sampled_row_ids` and `embed_sampled_row_ids`. The payload records
  `nt_sample_mode=deterministic_lcg` and `nt_sample_size`.
- NT rows and summaries also include edited-row delta L2 norm metrics:
  `edited_lm_delta_l2_mean/max` and `edited_embed_delta_l2_mean/max`.
- `PatchArtifact.budget_decision(NormBudgetPolicy(...))` and the EasyEdit
  `--max-*-norm` / `--max-edited-rows` flags provide an optional no-commit
  gate. These flags are disabled by default; when enabled, rejected proposals
  are rolled back and marked as `edit_status=no_commit`.
- `--edit-backend side_slot` is a sequential experimental backend. It converts
  each proposed edit into a side-slot `PatchArtifact`, rolls the base row
  mutation back, and evaluates through a `RuntimeSparseOverlay`.
- `--edit-backend wal_rome` is an experimental internal-layer backend. It
  applies sparse ROME-style updates to `mlp.down_proj` rows and reports FFN row
  counts, delta norms, and non-target row drift under `NT`. It is implemented
  for backend comparison, but it does not yet have an official n=50 Llama
  headline artifact.
- `WALMemitBatchEditor` exists as an offline consolidation backend for
  `PatchArtifact` batches. It is not yet a runner backend; use it for patch
  lifecycle tests, not EasyEdit headline claims.
- `--compare-backends` writes a matrix artifact and per-backend outputs for a
  comma-separated backend list. `wal_memit` is reported as skipped until direct
  fact-to-batch evaluation exists.
- `python -m agim.eval.ripple_diagnostic --input <artifact.json>` writes a
  post-hoc related-fact diagnostic over EasyEdit locality/relation fields. It is
  RippleEdits-style, not an official RippleEdits dataset score.
- `python -m agim.eval.mquake_diagnostic --input <artifact.json>` writes a
  post-hoc multi-hop diagnostic over EasyEdit portability fields. It is
  MQuAKE-style, not an official MQuAKE dataset score.
- `python -m agim.eval.raw_text_edit_pipeline --text "The capital of France is
  Berlin." --output <proposal.json>` converts raw update text into
  EasyEdit-style `requested_rewrite` proposals. This is AKEW-style parsing, not
  a benchmark score.
- `--history-slot-mode relation` is a relation-sharded sequential history
  ablation. It keeps edit-key history slots by CounterFact `relation_id`.
- `--state-namespace` isolates mutable editor state: history keys and
  relation-protected banks are stored per namespace. Use this for tenant or
  batch separation; the default namespace preserves prior behavior.
- `--relation-protected-mode accumulate/preload` builds relation_id-scoped
  protected locality banks. On seed-42 sequential n=50, `accumulate` improved
  TF locality from 32.0% to 51.2% with TF rewrite dropping to 76.0%; `preload`
  improved TF locality to 73.8% with TF rewrite dropping to 62.0%.
- `--positive-constraint-mode projected` is a constrained positive/protected
  key ablation. It projects paraphrase-positive keys away from protected
  locality keys before mixing them into the edit key.
- `--positive-constraint-mode ridge` is a stronger constrained
  positive/protected key solve. On the n=50 seed-42 official run it recovered
  locality relative to naive positive prompts (`TF locality=99.5%`,
  `Prob locality=93.2%`) but reduced exact rewrite to `80.0%`, so it remains
  an ablation knob rather than the default.
- Sequential retention is currently summary-only. Full retention metrics should
  be stored only when needed because they can make artifacts large.
- `--use-positive-prompts` is backed by n=50 GPU artifacts. It improves
  PS@All but currently hurts locality, so it is an ablation knob rather than the
  default headline setting.
- `--projection-mode orthogonal` is backed by an n=50 GPU artifact. It is worse
  on exact-token locality than the tuned sequential baseline, so it is not a
  recommended default.
- `--target-token-mode both` is backed by the n=200 token-mode matrix. It
  now uses contextual continuation ids as its primary sequence. It remains an
  ablation knob rather than the default.
- `clamp_anti` is backed by the n=200 component ablation as the main
  locality-preserving knob for the current single-edit profile.
- `clamp_eos` does not materially improve the single-edit n=200 component
  ablation, but default removal still needs follow-up random-seed validation.
- `clamp_eos=0.0` is now the persistent default after the random-200 no-EOS
  seed check matched the old default operating point.
- `--no-wal-encode-updates` is an ablation flag only. It shows that WAL
  reconstruction is not the main bottleneck for the current default profile.
- Edited-row norm metrics are emitted under `NT` for new artifacts, so future
  patch growth can be monitored without loading full row tensors.
