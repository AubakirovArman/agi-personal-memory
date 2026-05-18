# Relation Protected Bank Sequential Ablation

Date: 2026-05-18

Runner: official EasyEdit-compatible CounterFact runner

Model: `meta-llama/Llama-3.1-8B-Instruct`

Code commit in artifacts: `45d6758259b999a0670128cfcd65d453747e0c85`

Shared sequential profile:

```text
--sequential-edit
--target-token-mode contextual
--use-neg-prompts --neg-prompt-limit 4
--neg-projection-strength 0.50
--clamp_lm 0.15
--clamp_eos 0 --clamp_anti 0
--retention-steps 1,10,50
```

## What Was Added

`--relation-protected-mode` adds relation_id-scoped locality prompt banks:

- `accumulate`: add each completed edit's neighborhood prompt keys to that
  relation's protected bank.
- `preload`: build each relation bank from the selected batch's neighborhood
  prompts before editing starts.

Both modes use `--relation-protected-prompt-limit` per record and
`--max-relation-protected-keys` per relation.

## Metrics

| Seed 42 sequential profile | TF rewrite | TF rephrase | TF PS@All | TF locality | Prob rewrite | Prob PS@All | Prob locality | after 10 rewrite | after 10 locality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 84.0% | 30.0% | 27.0% | 32.0% | 96.0% | 71.0% | 66.6% | 100.0% | 82.0% |
| Relation bank accumulate | 76.0% | 22.0% | 21.0% | 51.2% | 96.0% | 58.0% | 74.2% | 80.0% | 83.0% |
| Relation bank preload | 62.0% | 22.0% | 16.0% | 73.8% | 92.0% | 42.0% | 87.8% | 60.0% | 91.0% |

## Artifacts

- `random_50_seed_42_seq_lm015_negx05_relation_protected_accumulate_noeosanti_retention.json`
- `random_50_seed_42_seq_lm015_negx05_relation_protected_accumulate_noeosanti_retention.failures.json`
- `random_50_seed_42_seq_lm015_negx05_relation_protected_preload_noeosanti_retention.json`
- `random_50_seed_42_seq_lm015_negx05_relation_protected_preload_noeosanti_retention.failures.json`

## Readout

Relation-specific protected banks are a real locality knob, but they are not a
complete sequential editing fix. `accumulate` is the more balanced setting on
this seed: locality improves from 32.0% to 51.2% with a smaller rewrite drop
than `preload`. `preload` protects locality much more strongly, but it is too
conservative for exact rewrite.

The result supports the productization plan's direction: relation-conditioned
protected subspaces reduce interference, but the next step needs explicit
commit/no-commit budgets or side-slot isolation rather than only stronger
projection.
