# AGIM WAL / EasyEdit Status - 2026-05-18

## Scope

Model: `meta-llama/Llama-3.1-8B-Instruct`

Primary GPU used: `cuda:3`

Dataset: CounterFact from `https://rome.baulab.info/data/dsets/counterfact.json`

Primary operating point:

```text
target_token_mode=contextual
use_neg_prompts=true
neg_prompt_limit=4
clamp_lm=0.20
clamp_embed=0.06
clamp_eos=0.16
clamp_anti=0.06
```

## Current 50-Fact Results

Primary artifact:
`results/easyedit_official_50_contextual_neg4_ctxgen_nt.json`

Code commit used by artifact:
`6aba33b78148cdd334717881001f92cd5014ca35`

| Metric group | Metric | Result | Readout |
| --- | ---: | ---: | --- |
| Teacher-forcing | rewrite | 100.0% | Strong |
| Teacher-forcing | rephrase | 71.0% | Useful, not solved |
| Teacher-forcing | locality | 58.4% | Weak |
| Official vanilla generation | rewrite | 0.0% | Tokenization-mismatched official score |
| Official vanilla generation | rephrase | 0.0% | Tokenization-mismatched official score |
| Contextual generation | rewrite | 100.0% | Strong under prompt+space target ids |
| Contextual generation | rephrase | 70.0% | Useful, not solved |
| Probability compare | rewrite | 100.0% | Strong |
| Probability compare | rephrase | 88.0% | Strong |
| Probability compare | locality | 37.6% | Weak |
| Dual NT diff | lm_head non-edited max | 0.0 | Good |
| Dual NT diff | embed non-edited max | 0.0 | Good |
| Control row | EOS changed rate | 100.0% | Expected global control row edit |
| Fluency | ngram entropy | 0.0 | Weak / degenerate short generation |

## Sequential 50-Fact Result

Artifact:
`results/easyedit_official_50_contextual_neg4_sequential.json`

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

## Interpretation

AGIM WAL dual-layer is currently a strong single-edit logit/continuation editor:
it reliably moves the target continuation under teacher-forcing and contextual
generation. It is not yet a robust lifelong/sequential editing method.

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
| Rollback consistency metrics | Done in custom evaluator | `easyedit_counterfact.py` strict/practical rollback fields |
| Generated artifact ignore rules | Done | `.gitignore` ignores smoke/runtime outputs |

## Not Done / Next Work

| Priority | Gap | Why it matters | Next action |
| ---: | --- | --- | --- |
| P0 | Sequential exact-token collapse | Blocks lifelong memory claim | Add conflict-aware/null-space projection or per-edit isolation |
| P0 | Locality weak in single-edit mode | Blocks EasyEdit-quality claim | Tune negative projection and add preserved-key constraints |
| P1 | Real portability benchmark missing | CounterFact has no multi-hop portability | Run KnowEdit or MQuAKE portability split |
| P1 | MQuAKE / KnowEdit not run | Needed for broader benchmark claims | Add reproducible dataset loader and 50/200 smoke |
| P1 | EasyEdit method package missing | Needed for upstream PR | Add `AGIMWAL_main.py`, `AGIMWAL_hparams.py`, hparams YAML |
| P2 | 100/1000 official runs with new metrics | Needed before any leaderboard language | Run random and first policies with fixed seeds |
| P2 | Fluency is degenerate | Current short exact outputs give entropy 0 | Add better generation-quality prompt set or report as not meaningful |

## Claim Boundary

Safe claim:

```text
On internal official-EasyEdit-compatible CounterFact single-edit runs for
Llama-3.1-8B-Instruct, AGIM WAL dual-layer reaches 100% rewrite and 71%
rephrase under teacher-forcing, with measured zero non-edited lm_head/embed row
diffs at n=50.
```

Unsafe claim:

```text
AGIM is #1 on EasyEdit or solved lifelong/sequential knowledge editing.
```

