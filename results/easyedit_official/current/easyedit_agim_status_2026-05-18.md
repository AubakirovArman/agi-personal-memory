# AGIM WAL / EasyEdit Status - 2026-05-18

## Scope

Model: `meta-llama/Llama-3.1-8B-Instruct`

Primary GPU used: `cuda:3`

Dataset: CounterFact from `https://rome.baulab.info/data/dsets/counterfact.json`

Primary single-edit operating point:

```text
target_token_mode=contextual
use_neg_prompts=true
neg_prompt_limit=4
neg_projection_strength=0.30
clamp_lm=0.20
clamp_embed=0.06
clamp_eos=0.16
clamp_anti=0.06
```

Current sequential tuned profiles disable the global EOS/anti rows and increase
locality projection:

```text
--sequential-edit
--target-token-mode contextual
--use-neg-prompts --neg-prompt-limit 4
--neg-projection-strength 0.50
--clamp_eos 0 --clamp_anti 0
```

## Current 50-Fact Results

Primary artifact:
`results/easyedit_official/current/easyedit_official_50_first42_psall_baseline.json`

Code commit used by artifact:
`cadd6f82f4e5b979610caf0441e6790028ed62ab`

| Metric group | Metric | Result | Readout |
| --- | ---: | ---: | --- |
| Teacher-forcing | rewrite | 100.0% | Strong |
| Teacher-forcing | rephrase | 71.0% | Useful, not solved |
| Teacher-forcing | PS@All | 67.0% | Useful, not solved |
| Teacher-forcing | locality | 58.4% | Weak |
| Official vanilla generation | rewrite | 0.0% | Tokenization-mismatched official score |
| Official vanilla generation | rephrase | 0.0% | Tokenization-mismatched official score |
| Official vanilla generation | PS@All | 0.3% | Tokenization-mismatched official score |
| Contextual generation | rewrite | 100.0% | Strong under prompt+space target ids |
| Contextual generation | rephrase | 70.0% | Useful, not solved |
| Contextual generation | PS@All | 66.0% | Useful, not solved |
| Probability compare | rewrite | 100.0% | Strong |
| Probability compare | rephrase | 88.0% | Strong |
| Probability compare | PS@All | 89.0% | Strong |
| Probability compare | locality | 37.4% | Weak |
| Dual NT diff | lm_head non-edited max | 0.0 | Good |
| Dual NT diff | embed non-edited max | 0.0 | Good |
| Control row | EOS changed rate | 100.0% | Expected global control row edit |
| Fluency | ngram entropy | 0.0 | Weak / degenerate short generation |

### Random-Seed Single-Edit Validation

Report:
`results/easyedit_official/current/random_50_report_2026-05-18.md`

Artifacts:

```text
results/easyedit_official/current/random_50_seed_42.json
results/easyedit_official/current/random_50_seed_43.json
results/easyedit_official/current/random_50_seed_44.json
```

Code commit used by artifacts:
`8d3f69991fbc6611d30691c00e3ef181d0a5ac05`

These runs use the current default locality-protected single-edit profile and
random CounterFact samples of 50 facts. They are a stability check, not a
replacement for the first-50 PS@All baseline above.

| Seed | TF rewrite | TF rephrase | TF PS@All | TF locality | CTX rewrite | Prob PS@All | Prob locality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 98.0% | 30.0% | 25.0% | 97.2% | 98.0% | 52.0% | 87.4% |
| 43 | 86.0% | 18.0% | 22.0% | 96.8% | 86.0% | 38.0% | 88.4% |
| 44 | 94.0% | 26.0% | 23.0% | 97.4% | 92.0% | 43.0% | 87.6% |
| Mean | 92.7% | 24.7% | 23.3% | 97.1% | 92.0% | 44.3% | 87.8% |

Readout: exact single-edit rewrite is fairly stable across random samples and
exact-token locality is strong in this profile. Paraphrase generalization is
weak: PS@All stays near 22-25%. This confirms the current method tradeoff:
locality-protected defaults do not yet give strong paraphrase transfer, while
positive-prompt variants improve PS@All at a locality cost.

### Positive-Prompt Single-Edit Ablation

Artifact:
`results/easyedit_official/ablations/easyedit_official_50_first42_psall_positive_prompts.json`

Command delta:

```text
--use-positive-prompts --positive-prompt-limit 4 --positive-key-weight 1.0
```

| Metric group | Rewrite | Rephrase | PS@All | Locality | Readout |
| --- | ---: | ---: | ---: | ---: | --- |
| Teacher-forcing | 100.0% | 96.0% | 95.0% | 45.2% | Stronger paraphrase, weaker locality |
| Contextual generation | 100.0% | 96.0% | 95.0% | n/a | Stronger paraphrase |
| Probability compare | 100.0% | 96.0% | 98.0% | 25.2% | Stronger paraphrase, weaker locality |

Readout: multi-positive keys are useful, but they are a tradeoff rather than a
strict win. They improve PS@All and rephrase sharply, while locality drops.

## Sequential 50-Fact Result

### With EOS / anti-boost enabled

Artifact:
`results/easyedit_official/sequential/easyedit_official_50_contextual_neg4_sequential.json`

Code commit used by artifact:
`7baae83acfc88cb3f2b76e9955cd7f6c7652834e`

| Metric group | Metric | Result | Readout |
| --- | ---: | ---: | --- |
| Teacher-forcing | rewrite | 0.0% | Fails under accumulated edits |
| Teacher-forcing | rephrase | 0.0% | Fails under accumulated edits |
| Teacher-forcing | locality | 0.0% | Fails under accumulated edits |
| Contextual generation | rewrite | 0.0% | Fails under accumulated edits |
| Contextual generation | rephrase | 0.0% | Fails under accumulated edits |
| Probability compare | rewrite | 90.0% | Probability shift mostly remains |
| Probability compare | rephrase | 80.0% | Probability shift mostly remains |
| Probability compare | locality | 43.0% | Weak |

### With EOS / anti-boost disabled

Artifact:
`results/easyedit_official/sequential/easyedit_official_50_contextual_neg4_sequential_noeosanti.json`

Code commit used by artifact:
`fca715c356c0dad106749c31f247d892c61537d0`

Command delta:

```text
--sequential-edit --clamp_eos 0 --clamp_anti 0
```

| Metric group | Metric | Result | Readout |
| --- | ---: | ---: | --- |
| Teacher-forcing | rewrite | 64.0% | Better than EOS mode, not solved |
| Teacher-forcing | rephrase | 23.0% | Weak |
| Teacher-forcing | locality | 5.8% | Very weak |
| Contextual generation | rewrite | 64.0% | Better than EOS mode, not solved |
| Contextual generation | rephrase | 22.0% | Weak |
| Probability compare | rewrite | 90.0% | Probability shift mostly remains |
| Probability compare | rephrase | 82.0% | Probability shift mostly remains |
| Probability compare | locality | 32.6% | Weak |

### Sequential no-EOS/no-anti clamp sweep

Artifacts:

```text
results/easyedit_official/sequential/easyedit_official_50_contextual_neg4_sequential_noeosanti.json
results/easyedit_official/sequential/easyedit_official_50_contextual_neg4_seq_lm015_noeosanti.json
results/easyedit_official/sequential/easyedit_official_50_contextual_neg4_seq_lm012_noeosanti.json
```

| clamp_lm | TF rewrite | TF rephrase | TF locality | Context rewrite | Probability locality | Readout |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.20 | 64.0% | 23.0% | 5.8% | 64.0% | 32.6% | Best exact rewrite in this sweep |
| 0.15 | 62.0% | 23.0% | 8.8% | 62.0% | 38.6% | Slight locality gain, no real win |
| 0.12 | 59.0% | 25.0% | 16.0% | 58.0% | 44.8% | Better locality, weaker exact rewrite |

### Sequential negative-projection sweep

This run exposes the previous hard-coded neighborhood projection coefficient as
`--neg-projection-strength`. The tested profile keeps no-EOS/no-anti sequential
mode and raises projection from `0.30` to `0.50`.

Artifacts:

```text
results/easyedit_official/sequential/easyedit_official_50_contextual_neg4x05_seq_lm012_noeosanti.json
results/easyedit_official/sequential/easyedit_official_50_contextual_neg4x05_seq_lm015_noeosanti.json
results/easyedit_official/sequential/easyedit_official_50_contextual_neg4x05_seq_lm020_noeosanti.json
results/easyedit_official/sequential/easyedit_official_50_contextual_neg4x07_seq_lm020_noeosanti.json
```

Code commit used by tuned artifacts:
`e884b1aa302a4498beda105de7279e7aa6222dcb`

| profile | TF rewrite | TF rephrase | TF locality | Context rewrite | Probability rewrite | Probability locality | Readout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| lm=0.12, neg=0.30 | 59.0% | 25.0% | 16.0% | 58.0% | 88.0% | 44.8% | Previous best-locality baseline |
| lm=0.12, neg=0.50 | 61.0% | 21.0% | 35.0% | 60.0% | 82.0% | 65.6% | Best TF locality among 50-fact tuned runs |
| lm=0.15, neg=0.30 | 62.0% | 23.0% | 8.8% | 62.0% | 90.0% | 38.6% | Previous balanced baseline |
| lm=0.15, neg=0.50 | 71.0% | 21.0% | 25.4% | 70.0% | 86.0% | 61.2% | Best balanced tuned profile |
| lm=0.20, neg=0.50 | 78.0% | 23.0% | 12.2% | 78.0% | 92.0% | 55.4% | Best exact rewrite, locality still weak |
| lm=0.20, neg=0.70 | 67.0% | 15.0% | 23.6% | 66.0% | 90.0% | 74.4% | Strong probability locality, weak rephrase |

Smoke result:
`results/easyedit_official/smoke/easyedit_official_smoke_seq20_histproj05_noeosanti.json`
tested `history_projection_strength=0.50` and
`embed_history_projection_strength=0.50`. It did not improve locality
(`TF locality=13.5%` at `lm=0.20`), so history projection remains an optional
diagnostic knob, not a recommended default.

## Interpretation

AGIM WAL dual-layer is currently a strong single-edit logit/continuation editor:
it reliably moves the target continuation under teacher-forcing and contextual
generation. It is not yet a robust lifelong/sequential editing method.

The sequential experiment shows that accumulated EOS/anti-boost control rows
were part of the exact-token collapse: disabling them raises sequential rewrite
from 0.0% to 64.0%. It does not solve sequential editing because locality and
rephrase remain weak.

Lowering `clamp_lm` in no-EOS/no-anti sequential mode improves locality only
modestly and costs exact rewrite. This points to edit interference across target
rows and subject embeddings, not only EOS accumulation.

Increasing neighborhood projection is the first confirmed sequential/locality
improvement. At `lm=0.15`, raising the coefficient from `0.30` to `0.50`
improves 50-fact sequential rewrite from 62.0% to 71.0% and TF locality from
8.8% to 25.4%, with a small rephrase cost. At `lm=0.12`, TF locality improves
from 16.0% to 35.0%, with a small rewrite gain and a rephrase cost. This is a
real improvement, but not a solved locality story: exact-token locality is still
well below the level needed for a strong EasyEdit claim.

### Fresh Sequential Retention

Artifact:
`results/easyedit_official/sequential/easyedit_official_50_first42_psall_seq_lm015_negx05_noeosanti_retention.json`

Code commit used by artifact:
`cadd6f82f4e5b979610caf0441e6790028ed62ab`

| Checkpoint | TF rewrite | TF rephrase | TF PS@All | TF locality | Probability PS@All | Probability locality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| after 1 | 100.0% | 0.0% | 0.0% | 60.0% | 50.0% | 60.0% |
| after 10 | 100.0% | 50.0% | 50.0% | 59.0% | 90.0% | 39.0% |
| after 50 | 73.0% | 21.0% | 20.0% | 25.4% | 61.0% | 61.4% |

Sequential positive-prompt artifact:
`results/easyedit_official/sequential/easyedit_official_50_first42_psall_seq_lm015_negx05_noeosanti_positive_prompts_retention.json`

| Checkpoint | TF rewrite | TF rephrase | TF PS@All | TF locality | Probability PS@All | Probability locality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| after 1 | 100.0% | 100.0% | 100.0% | 90.0% | 100.0% | 90.0% |
| after 10 | 100.0% | 100.0% | 100.0% | 51.0% | 100.0% | 29.0% |
| after 50 | 47.0% | 31.0% | 29.5% | 16.2% | 79.0% | 56.0% |

Readout: the tuned sequential profile keeps exact rewrite through 10 edits but
loses generalization/locality by 50 edits. Positive prompts improve sequential
PS@All but hurt exact rewrite and exact-token locality.

### Sequential Random-Seed Retention

Report:
`results/easyedit_official/sequential/sequential_random_50_report_2026-05-18.md`

Artifacts:

```text
results/easyedit_official/sequential/random_50_seed_42_seq_lm015_negx05_noeosanti_retention.json
results/easyedit_official/sequential/random_50_seed_43_seq_lm015_negx05_noeosanti_retention.json
results/easyedit_official/sequential/random_50_seed_44_seq_lm015_negx05_noeosanti_retention.json
```

| Seed | TF rewrite | TF rephrase | TF PS@All | TF locality | Prob PS@All | Prob locality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 84.0% | 30.0% | 27.0% | 32.0% | 71.0% | 66.6% |
| 43 | 68.0% | 18.0% | 20.0% | 19.8% | 64.0% | 65.6% |
| 44 | 84.0% | 32.0% | 29.0% | 50.0% | 62.0% | 63.0% |
| Mean | 78.7% | 26.7% | 25.3% | 33.9% | 65.7% | 65.1% |

Mean retention:

| Checkpoint | TF rewrite | TF PS@All | TF locality |
| --- | ---: | ---: | ---: |
| after 10 | 100.0% | 31.7% | 83.0% |
| after 50 | 78.7% | 25.3% | 33.9% |

Readout: this confirms a longer-chain sequential degradation pattern. The tuned
profile is healthy at 10 accumulated edits, then loses exact-token locality and
some rewrite strength by 50 edits.

Seed 42 ablation results:

| Profile | TF rewrite | TF PS@All | TF locality | Readout |
| --- | ---: | ---: | ---: | --- |
| Baseline | 84.0% | 27.0% | 32.0% | Reference |
| `--history-slot-mode relation` | 82.0% | 27.0% | 31.8% | No win on this seed |
| `--positive-constraint-mode projected` with positive prompts | 54.0% | 42.0% | 34.6% | Better PS, much weaker rewrite |

Orthogonal projection artifact:
`results/easyedit_official/sequential/easyedit_official_50_first42_psall_seq_lm015_negx05_orthogonal_noeosanti_retention.json`

Code commit used by artifact:
`0d21721c1918225f9071b7ee852d8d4cff2e050d`

| Metric group | Rewrite | Rephrase | PS@All | Locality | Readout |
| --- | ---: | ---: | ---: | ---: | --- |
| Teacher-forcing | 58.0% | 25.0% | 25.0% | 6.6% | Worse exact locality than sequential tuned |
| Probability compare | 90.0% | 82.0% | 81.0% | 30.4% | Probability gains, locality still weak |

Readout: `--projection-mode orthogonal` is implemented and tested as an opt-in
knob, but it is not the locality fix. It makes the protected-key operation
stricter while exact-token locality gets worse, so the likely next step is
per-edit isolation or relation sharding.

The official vanilla generation score is intentionally preserved, but it is not
the whole story for Llama tokenization. EasyEdit compares generated ids to
`tok.encode(target_new)` while teacher-forcing evaluates `prompt + " " + target`.
For Llama these ids often differ, for example `English` vs ` ĠEnglish`.
That is why the repo now reports both official vanilla generation and contextual
generation.

## Done From Kimi / 1111 Audit

| Item | Status | Evidence |
| --- | --- | --- |
| Official EasyEdit teacher-forcing metric | Done | `easyedit_official_runner.py`, 50-fact artifacts |
| Probability `P(new) > P(true)` metric | Done | `post_probability` in 50-fact artifacts |
| EasyEdit locality pre/post consistency | Done | `attach_locality_acc` in runner |
| Fluency n-gram entropy metric | Done | `post_fluency` in 50-fact artifact |
| Sequential editing mode | Done | `--sequential-edit`, 50-fact sequential artifact |
| Portability parser/support | Partial | Runner normalizes portability fields, but CounterFact has no portability data |
| Token-based prompt truncation | Done | CounterFact evaluator and WAL negative prompts |
| ROME causal tracing / NameError fixes | Done | `rome_causal.py`, `rome_editor.py` |
| WAL shape validation | Done | `wal_editor.py`, `wal_dual_editor.py` paths |
| Dual NT measurement | Done | `measure_non_target_diffs`, NT summary in artifacts |
| Configurable negative projection | Done | `--neg-projection-strength`, 50-fact tuned artifacts |
| Optional history projection diagnostics | Done | `--history-projection-strength`, smoke artifact showed no win |
| Rollback consistency metrics | Done in custom evaluator | `easyedit_counterfact.py` strict/practical rollback fields |
| Generated artifact ignore rules | Done | `.gitignore` ignores smoke/runtime outputs |
| Relation breakdown for new artifacts | Done | `metrics_by_relation_id` in `easyedit_official_runner.py` summaries |
| PS@All for new artifacts | Done | `rephrase_all_acc` fields in post/generation/contextual/probability summaries |
| Sequential retention summaries for new artifacts | Done | `--retention-steps`, default `1,10,50` |
| Optional multi-positive key ablation | Done with GPU artifacts | Single-edit and sequential positive-prompt artifacts |
| Orthogonal protected projection | Done with GPU artifact | `--projection-mode orthogonal`; n=50 result is not recommended |

## Not Done / Next Work

| Priority | Gap | Why it matters | Next action |
| ---: | --- | --- | --- |
| P0 | Sequential exact-token/locality weakness | Blocks lifelong memory claim | Add per-edit isolation or relation sharding |
| P0 | Locality still weak after projection tuning | Blocks EasyEdit-quality claim | Try constrained row updates or MEMIT/ROME-style layer edits instead of lm_head-only row boosts |
| P1 | Real portability benchmark missing | CounterFact has no multi-hop portability | Run KnowEdit or MQuAKE portability split |
| P1 | MQuAKE / KnowEdit not run | Needed for broader benchmark claims | Add reproducible dataset loader and 50/200 smoke |
| P1 | EasyEdit method package missing | Needed for upstream PR | Add `AGIMWAL_main.py`, `AGIMWAL_hparams.py`, hparams YAML |
| P1 | Relation-level failure analysis incomplete | Needed to target method fixes | Use `metrics_by_relation_id` from fresh artifacts |
| P2 | Fresh random-seed and 100/1000 official runs missing | Needed before any leaderboard language | Run `random_50_seed_42`, `random_50_seed_43`, then 100/1000 |
| P2 | Fluency is degenerate | Current short exact outputs give entropy 0 | Add better generation-quality prompt set or report as not meaningful |

## Claim Boundary

Safe claim:

```text
On internal official-EasyEdit-compatible CounterFact single-edit runs for
Llama-3.1-8B-Instruct, AGIM WAL dual-layer reaches 100% rewrite and 71%
rephrase / 67% PS@All under teacher-forcing, with measured zero non-edited
lm_head/embed row diffs at n=50.
```

Unsafe claim:

```text
AGIM is #1 on EasyEdit or solved lifelong/sequential knowledge editing.
```
