# Realistic Roadmap

This roadmap replaces broad v10-style claims with concrete evidence-driven
steps.

## Now

- Keep Path A and Path B separated in docs, results, and claims.
- Use `CURRENT_STATUS.md` as the short source of truth.
- Use `BENCHMARK.md` for metric details and caveats.
- Treat `results/local_protocol/` as historical diagnostics only.

## Next Two Weeks

- Run fresh n=50 artifacts with the new `PS@All`, `metrics_by_relation_id`, and
  sequential retention fields.
- Run an ablation with `--use-positive-prompts` to see whether multi-positive
  key averaging improves `PS@All`.
- Use the newly emitted `metrics_by_relation_id` in fresh artifacts to find
  relation-level collapse cases.
- Run `first_50`, `random_50_seed_42`, and `random_50_seed_43`.
- Keep outputs under `results/easyedit_official/current/` or
  `results/easyedit_official/sequential/`.

## Next Month

- Implement protected-key/null-space projection for locality.
- Add relation sharding by CounterFact `relation_id`.
- Run KnowEdit or MQuAKE portability splits as diagnostic benchmarks.
- Build an EasyEdit method adapter package for external review.

## Research Frontier

- Sequential side memory with routing.
- Batch WAL consolidation with preserved-key constraints.
- Rollback consistency for individual edits after long sequential runs.
- Interference matrices across edits, relations, and target rows.

## Exit Criteria For Stronger Claims

Before claiming strong sequential/lifelong editing, AGIM needs:

- stable random-seed EasyEdit-compatible runs;
- relation breakdown without hidden collapse cases;
- sequential retention curves;
- locality above a defensible threshold;
- reproducible commands and artifacts from the current runner.
