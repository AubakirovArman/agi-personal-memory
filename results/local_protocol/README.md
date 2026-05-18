# Legacy Local Protocol Results

This folder contains historical CounterFact and WAL experiment outputs. They are
kept for audit/debugging, but they are not official EasyEdit results.

Short invalidation notice: `README_INVALIDATED.md`.

Important: files named `official_eval_*.json` in this folder predate the real
EasyEdit runner. They were produced by AGIM-local evaluators and custom metrics.
They should not be used for leaderboard-style claims.

Some old markdown files in this folder still show the original historical tables
and comparisons. Treat those tables as experiment notes, not as current claims.

## The 1000-Fact Result

`official_eval_1000.json` reports:

| Metric | Value |
| --- | ---: |
| `EasyEdit.ES` | 91.4% |
| `EasyEdit.PS` | 35.75% |
| `EasyEdit.NS_absence` | 76.2% |
| `EasyEdit.Composite` | 67.78% |
| `rep_rate` | 0.0% |
| `NT` | 0.00025464 |

This is useful as a local stress/diagnostic test, but it does not guarantee the
current official-compatible EasyEdit metrics because:

- `NS_absence` checks that `target_new` does not appear in neighbor generations;
  EasyEdit locality checks whether neighbor outputs stay the same before and
  after the edit.
- The composite score is an AGIM-local aggregate. EasyEdit does not define this
  exact overall score for these runs.
- The artifact has no command/git metadata and was not produced by
  `agim.eval.easyedit_official_runner`.
- It does not replace the current n=50 official-compatible artifacts in
  `../easyedit_official/`.

Safe wording:

```text
Historical AGIM-local CounterFact 1000-fact diagnostic: ES=91.4% under the
legacy token-exact/local protocol, NS_absence=76.2%, Composite=67.8%, with
0% repetition and very small measured non-target diff.
```

Unsafe wording:

```text
AGIM gets 67.8% on official EasyEdit or beats EasyEdit leaderboard methods.
```
