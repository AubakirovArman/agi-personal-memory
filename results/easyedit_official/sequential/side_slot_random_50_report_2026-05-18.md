# Side-Slot Sequential Random 50 Report - 2026-05-18

Model: `meta-llama/Llama-3.1-8B-Instruct`

Runner: official EasyEdit-compatible CounterFact runner

Code commit in artifact: `dc1f938dad1d0ad693c68ede51517b44f3753451`

Artifact:

- `random_50_seed_42_seq_side_slot_lm015_negx05_noeosanti_retention.json`
- `random_50_seed_42_seq_side_slot_lm015_negx05_noeosanti_retention.failures.json`

Command:

```bash
AGIM_EASYEDIT_ROOT=/mnt/hf_model_weights/arman/3bit/sites/EasyEdit \
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --n 50 \
  --sample-policy random \
  --seed 42 \
  --device cuda:2 \
  --output results/easyedit_official/sequential/random_50_seed_42_seq_side_slot_lm015_negx05_noeosanti_retention.json \
  --save-failures-only \
  --sequential-edit \
  --edit-backend side_slot \
  --target-token-mode contextual \
  --use-neg-prompts \
  --neg-prompt-limit 4 \
  --neg-projection-strength 0.50 \
  --clamp_lm 0.15 \
  --clamp_eos 0 \
  --clamp_anti 0 \
  --retention-steps 1,10,50 \
  --nt-sample-size 32
```

## Metrics

| Seed 42 sequential profile | TF rewrite | TF rephrase | TF PS@All | TF locality | CTX rewrite | Prob rewrite | Prob PS@All | Prob locality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| In-place tuned baseline | 84.0% | 30.0% | 27.0% | 32.0% | 86.0% | 96.0% | 71.0% | 66.6% |
| Side-slot backend | 100.0% | 52.0% | 46.0% | 88.9% | 100.0% | 100.0% | 75.0% | 65.0% |

Retention checkpoints:

| Checkpoint | TF rewrite | TF PS@All | TF locality |
| --- | ---: | ---: | ---: |
| after 1 | 100.0% | 0.0% | 100.0% |
| after 10 | 100.0% | 40.0% | 95.0% |
| after 50 | 100.0% | 46.0% | 88.9% |

Side-slot summary:

```json
{"slots": 50, "enabled": 50}
```

## Readout

This is the first strong sequential result for Path B. Unlike the in-place
baseline, each proposal is converted into a `PatchArtifact`, rolled back from
base weights, and evaluated through a routed runtime overlay. On seed 42 this
keeps exact rewrite at 100.0% and raises teacher-forcing locality from 32.0% to
88.9% after 50 accumulated edits.

This does not yet prove solved lifelong editing. It is one random seed, and
probability locality remains around the previous tuned baseline. It does show
that side-slot isolation is the right architecture for sequential serving, and
that in-place row accumulation was the main source of the earlier exact-token
sequential collapse.
