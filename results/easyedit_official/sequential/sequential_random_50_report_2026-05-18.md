# Sequential Random 50 EasyEdit-Compatible Report - 2026-05-18

Model: `meta-llama/Llama-3.1-8B-Instruct`

Runner: `agim.eval.easyedit_official_runner`

Code commit in artifacts: `abdb8ef8ffcedcdff95434870a3bddf8c97137b9`

Sequential profile:

```text
--sequential-edit
--target-token-mode contextual
--use-neg-prompts --neg-prompt-limit 4
--neg-projection-strength 0.50
--clamp_lm 0.15
--clamp_eos 0 --clamp_anti 0
--retention-steps 1,10,50
```

## Random-Seed Baseline

| Seed | TF rewrite | TF rephrase | TF PS@All | TF locality | CTX rewrite | Prob PS@All | Prob locality | after 10 locality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 84.0% | 30.0% | 27.0% | 32.0% | 86.0% | 71.0% | 66.6% | 82.0% |
| 43 | 68.0% | 18.0% | 20.0% | 19.8% | 68.0% | 64.0% | 65.6% | 94.0% |
| 44 | 84.0% | 32.0% | 29.0% | 50.0% | 84.0% | 62.0% | 63.0% | 73.0% |
| Mean | 78.7% | 26.7% | 25.3% | 33.9% | 79.3% | 65.7% | 65.1% | 83.0% |

Retention mean over seeds 42/43/44:

| Checkpoint | TF rewrite | TF PS@All | TF locality |
| --- | ---: | ---: | ---: |
| after 10 | 100.0% | 31.7% | 83.0% |
| after 50 | 78.7% | 25.3% | 33.9% |

Readout: the tuned sequential profile survives 10 accumulated edits well on
exact rewrite and locality, but it degrades by 50 edits. Sequential editing is
therefore still partial, not solved.

## Seed 42 Ablations

| Profile | TF rewrite | TF rephrase | TF PS@All | TF locality | Prob PS@All | Prob locality | Readout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | 84.0% | 30.0% | 27.0% | 32.0% | 71.0% | 66.6% | Reference |
| `--history-slot-mode relation` | 82.0% | 30.0% | 27.0% | 31.8% | 71.0% | 66.6% | No win on this seed |
| `--use-positive-prompts --positive-constraint-mode projected` | 54.0% | 46.0% | 42.0% | 34.6% | 86.0% | 71.4% | Better PS, much weaker rewrite |

Readout: relation-sharded history slots are implemented but do not improve this
seed. Projected positive prompts recover more paraphrase transfer and probability
quality, but exact rewrite drops too much for a default setting.

## Artifacts

- `random_50_seed_42_seq_lm015_negx05_noeosanti_retention.json`
- `random_50_seed_43_seq_lm015_negx05_noeosanti_retention.json`
- `random_50_seed_44_seq_lm015_negx05_noeosanti_retention.json`
- `random_50_seed_42_seq_lm015_negx05_relation_slots_noeosanti_retention.json`
- `random_50_seed_42_seq_lm015_negx05_projected_positive_noeosanti_retention.json`

Each run also has a matching `.failures.json` compact triage artifact.
