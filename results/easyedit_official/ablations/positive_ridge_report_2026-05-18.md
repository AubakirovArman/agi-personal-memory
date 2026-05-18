# Positive Ridge Constraint Ablation

Date: 2026-05-18

Runner: official EasyEdit-compatible CounterFact runner

Model: `meta-llama/Llama-3.1-8B-Instruct`

Artifact:

- `positive_ridge_50_seed_42.json`
- `positive_ridge_50_seed_42.failures.json`

Command:

```bash
AGIM_EASYEDIT_ROOT=/mnt/hf_model_weights/arman/3bit/sites/EasyEdit \
PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --n 50 \
  --sample-policy random \
  --seed 42 \
  --device cuda:2 \
  --target-token-mode contextual \
  --use-positive-prompts \
  --positive-constraint-mode ridge \
  --positive-prompt-limit 4 \
  --positive-key-weight 1.0 \
  --nt-sample-size 32 \
  --output results/easyedit_official/ablations/positive_ridge_50_seed_42.json \
  --save-failures-only
```

## Metrics

| Profile | TF rewrite | TF rephrase | TF PS@All | TF locality | Prob rewrite | Prob PS@All | Prob locality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current seed-42 default | 98.0% | 30.0% | 25.0% | 97.2% | 100.0% | 52.0% | 87.4% |
| Naive positive prompts | 100.0% | 96.0% | 95.0% | 45.2% | 100.0% | 98.0% | 25.2% |
| Positive ridge constraint | 80.0% | 66.0% | 60.0% | 99.5% | 92.0% | 83.0% | 93.2% |

Contextual generation for the ridge run matched teacher forcing:
`rewrite=80.0%`, `rephrase=66.0%`, `PS@All=60.0%`.

The old vanilla generation template stays at 0.0% because this run edits the
contextual continuation target. It should not be used as the primary score for
this profile.

## Failure Triage

The failures-only artifact records 38 failed cases across the configured
families `tf`, `ctx_gen`, and `prob`.

Failure-mode counts:

- `tf_rewrite`: 10
- `tf_rephrase`: 17
- `tf_ps_all`: 27
- `tf_locality`: 3
- `ctx_gen_rewrite`: 10
- `prob_locality`: 18

## Readout

The ridge constrained solve fixes the main failure of naive positive prompts:
locality recovers from 45.2% teacher-forcing locality to 99.5%, and probability
locality recovers from 25.2% to 93.2%.

The tradeoff is lower exact rewrite than the current seed-42 default
(`80.0%` vs `98.0%`). This means `ridge` is useful evidence that constrained
positive/protected key solving is the right direction, but it is not a new
default profile yet. The next useful work is tuning the constraint strength and
positive-key weight, then validating at n=200 before promoting it.
