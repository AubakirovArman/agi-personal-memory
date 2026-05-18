# Realistic Roadmap

This roadmap replaces broad v10-style claims with concrete evidence-driven
steps.

## Now

- Keep Path A and Path B separated in docs, results, and claims.
- Use `CURRENT_STATUS.md` as the short source of truth.
- Use `BENCHMARK.md` for metric details and caveats.
- Treat `results/local_protocol/` as historical diagnostics only.

## Next Two Weeks

- Fresh `first_50` artifacts now include `PS@All`, `metrics_by_relation_id`,
  and sequential retention fields.
- The `--use-positive-prompts` ablation improves single-edit PS@All but hurts
  locality, so it is not a default method setting yet.
- The `--projection-mode orthogonal` ablation is implemented and tested; it did
  not fix locality, so stricter key projection alone is not enough.
- Use the newly emitted `metrics_by_relation_id` in fresh artifacts to find
  relation-level collapse cases.
- Run `random_50_seed_42` and `random_50_seed_43`, then 100/1000 only after the
  n=50 method tradeoffs are understood.
- Keep outputs under `results/easyedit_official/current/` or
  `results/easyedit_official/sequential/`.

## Next Month

- Add per-edit side memory or isolation for sequential edits.
- Add relation sharding by CounterFact `relation_id`.
- Try constrained row updates or MEMIT/ROME-style layer edits for locality.
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
