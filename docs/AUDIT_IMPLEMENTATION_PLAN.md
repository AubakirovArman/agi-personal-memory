# Audit Implementation Plan

Source files:

- `/mnt/hf_model_weights/arman/3bit/AGIM_Final_Complete_Analysis.md`
- `/mnt/hf_model_weights/arman/3bit/1111.md`

## Requirements Extracted

1. Keep EasyEdit, local CounterFact, Path A memory, and Path B weight editing
   separated in docs and claims.
2. Before new method work, keep Python/script files small enough to review:
   target limit is at most 300 lines per file.
3. Make EasyEdit runs reproducible and inspectable:
   PS@All, retention, relation breakdown, failure analysis, and command metadata.
4. Treat positive prompts as an ablation until locality-preserving constraints
   exist.
5. Stop spending effort on clamp sweeps as the main path.
6. Prioritize interference reduction:
   side-memory edit slots, relation sharding, and constrained positive/protected
   updates.

## Implemented In This Pass

- Split `src/agim/eval/easyedit_official_runner.py` into focused modules.
- Split `src/agim/eval/easyedit_counterfact.py` into data, protocol, summary,
  CLI, and evaluator modules.
- Split helper logic out of `src/agim/model/wal_dual_editor.py`.
- Verified all files under `src/agim/eval/` and `src/agim/model/` are now at or
  below 300 lines.

## Remaining File-Size Work

The next cleanup pass should split the remaining large non-EasyEdit modules:

| Area | Current issue | Likely split |
| --- | --- | --- |
| `src/agim/dwl2/runtime.py` | Very large runtime/deployment module | runtime layers, grouped RVQ, replacement helpers, stage controls |
| `src/agim/dwl2/block_vq.py` | Large encoding implementation | dataclasses, encoding kernels, public API |
| `src/agim/wal/v1/*` | Several modules over 300 lines | separate IO/CLI/helpers from core logic |
| `src/agim/wal/v2/grammar.py` | Grammar and parser logic together | grammar specs vs parser helpers |
| `src/agim/cli/agim_full_benchmark.py` | CLI plus benchmark logic | CLI wrapper plus benchmark runner |

## Method Work After Size Cleanup

1. Add `--dry-run-summary` for EasyEdit sample/dataset inspection without model
   loading.
2. Add `--save-failures-only` for locality/rephrase failure triage.
3. Prototype constrained positive/protected solve on n=10.
4. Prototype side-memory edit slots for sequential n=20.
5. Add relation sharding using `relation_id`.
6. Run `random_50_seed_42`, `random_50_seed_43`, and `random_50_seed_44`.

## Verification Gate

Each cleanup pass must keep:

```text
PYTHONPATH=src python -m pytest -q
```

green before commit.
