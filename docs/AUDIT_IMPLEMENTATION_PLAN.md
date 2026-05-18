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
- Split DWL2 runtime and block-VQ modules into compatibility facades plus
  focused implementation modules.
- Split WAL v1/v2 modules, cross-model protocol helpers, and benchmark CLIs
  into smaller modules.
- Fixed stale WAL v1/v2 relative imports that resolved to the non-existent
  `agim.wal.wal` package during package-level imports.
- Verified all Python files outside ignored directories are now at or below
  300 lines.

## File-Size Gate

Current gate:

```text
find . -path ./.venv -prune -o -path ./knowedit_cache -prune -o -name '*.py' -print0 \
  | xargs -0 wc -l \
  | awk '$2 != "total" && $1>300 {print $1, $2}'
```

Expected output: empty.

## Method Work After Size Cleanup

1. Done: add `--dry-run-summary` for EasyEdit sample/dataset inspection
   without model loading.
2. Done: add `--save-failures-only` for locality/rephrase failure triage.
3. Done: add `--positive-constraint-mode projected` as the first constrained
   positive/protected-key ablation.
4. Done: add `--history-slot-mode relation` as the first relation-sharded
   side-slot ablation for sequential editing.
5. Done: add named presets for `random_50_seed_42`, `random_50_seed_43`, and
   `random_50_seed_44`.
6. Done: run the three random_50 presets on GPU 2/3 and compare artifacts.
   See `results/easyedit_official/current/random_50_report_2026-05-18.md`.
7. Done: run sequential random-seed retention checks for the tuned profile.
   See `results/easyedit_official/sequential/sequential_random_50_report_2026-05-18.md`.
8. Done: test `--history-slot-mode relation` and
   `--positive-constraint-mode projected` on sequential seed 42.
9. Next: design a stronger isolation mechanism for 50-edit sequential locality;
   current relation slots do not improve the seed 42 run, and projected
   positive prompts trade rewrite for PS@All.

## Verification Gate

Each cleanup pass must keep:

```text
PYTHONPATH=src python -m pytest -q
```

green before commit.
