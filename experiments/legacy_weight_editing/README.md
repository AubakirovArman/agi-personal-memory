# Legacy Weight-Editing Experiments

This folder contains old one-off scripts and sweep prototypes that were moved
out of the repository root to avoid confusing them with supported tests.

Current supported evaluation entry points live under `src/agim/eval/`:

- `agim.eval.easyedit_official_runner`: current EasyEdit-compatible CounterFact runner
- `agim.eval.easyedit_counterfact`: older AGIM-local CounterFact evaluator

The scripts here may still be useful for archaeology, but they are not CI tests
and should not be used for current claims without revalidation.

