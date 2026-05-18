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
- Use `--dry-run-summary` before GPU runs to verify the selected cases and
  relation mix.
- Use `--save-failures-only` after GPU runs to review only the failed
  rewrite/rephrase/locality cases.
- The random n=50 presets for seeds 42, 43, and 44 are complete. Their mean
  readout is `TF rewrite=92.7%`, `TF PS@All=23.3%`, and
  `TF locality=97.1%`: stable rewrite/locality, weak paraphrase transfer.
- The first-1000 official-compatible scale check is complete. The default
  profile reports `TF rewrite=91.1%`, `TF PS@All=24.7%`, and
  `TF locality=96.2%`: rewrite/locality scale, paraphrase transfer does not.
- Sequential random-seed retention checks are complete. The tuned profile
  averages `TF rewrite=100.0%` and `TF locality=83.0%` after 10 accumulated
  edits, then falls to `TF rewrite=78.7%` and `TF locality=33.9%` after 50.
- Keep outputs under `results/easyedit_official/current/` or
  `results/easyedit_official/sequential/`.

## Next Month

- Replace simple relation slots with a stronger isolation mechanism for
  sequential edits; the first relation-slot ablation did not improve seed 42.
- Continue positive-prompt work only with a rewrite-preserving constraint;
  projected positives improved PS@All but dropped exact rewrite too much.
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
