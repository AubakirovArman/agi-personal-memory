# Claims And Evidence

Use this file when deciding what AGIM can safely claim.

| Claim | Status | Evidence | Limitation |
| --- | --- | --- | --- |
| Path A stores and retrieves facts | Supported | `tests/`, `results/memory_retrieval/` | Runtime memory, not weight editing |
| Path A supports audit/rollback | Supported | `tests/test_core.py`, governance tests | AGIM layer rollback, not necessarily model-weight rollback |
| Path B single-edit works on n=50/n=200/n=1000 | Supported internally | `results/easyedit_official/current/` | Current profile is profile-dependent: high-PS settings hurt locality; locality-protected settings hurt PS@All; no external benchmark claim |
| Path B sequential editing works | Partial | `results/easyedit_official/sequential/` | Random-seed retention falls after 50 edits; sequential/locality unresolved |
| `wal_memit` official n=50 baseline (EasyEdit-compatible) | Supported | `results/easyedit_official/current/random_50_seed_42_wal_memit.json` | `rephrase` and PS@All are weak on this backend; locality is preserved |
| Full 4-backend official comparison (`dual_row`, `wal_rome`, `wal_memit`, `side_slot`) | Supported | `results/easyedit_official/ablations/backend_matrix_random_50_seed42*.json` | Per-backend JSONs + aggregate + report are present; `wal_memit` and `side_slot` quality are weak |
| Side-slot sequential retention (`n=10/50/100`, seeds 42/43/44) | Done | `results/easyedit_official/sequential/side_slot_random_{10,50,100}_seed_{42,43,44}_seq.json` and failures | Locality drops with chain length; still not solved for lifelong editing claims |
| Historical 1000 local CounterFact diagnostic exists | Supported as history | `results/local_protocol/official_eval_1000.json` | Not official EasyEdit-compatible |
| AGIM is number one on EasyEdit | Unsafe | None | External validation missing, no leaderboard result |
| AGIM solved lifelong editing | Unsafe | None | Sequential retention/locality unresolved |
| External consequence benchmark evidence (RippleEdits / MQuAKE / product) | Supported | `results/external_benchmark_runs/ripple_wal_memit_n50_seed42.json`, `results/external_benchmark_runs/mquake_wal_memit_n50_seed42_outputs.json`, `..._scored.json`, `results/external_benchmark_runs/raw_text_wal_memit_n50_seed42.json`, `results/external_benchmark_runs/product_scedit_wal_memit_n50_seed42.json` | Local diagnostics and tracked outputs only; not yet external leaderboard submissions |
| EasyEdit leaderboard submission path | Partial | `src/agim/integrations/easyedit_agimwal.py`, `docs/EASYEDIT_ADAPTER.md`, `docs/EASYEDIT_PROTOCOL.md` | Adapter/docs exist, official submission or accepted leaderboard result is still pending |

## Safe Headline

```text
AGIM is a verified memory substrate with Path A for auditable runtime memory and
Path B for reversible WAL-based model editing research.
```

## Safe Research Claim

```text
On internal EasyEdit-compatible CounterFact single-edit runs for
Llama-3.1-8B-Instruct, AGIM WAL reaches 100% teacher-forcing rewrite and 71%
teacher-forcing rephrase / 67% PS@All at n=50, with measured zero non-edited
lm_head/embed row diff. Sequential editing and locality remain open challenges.
```

```text
On three random n=50 EasyEdit-compatible CounterFact samples with the current
default locality-protected profile, AGIM WAL averages 92.7% teacher-forcing
rewrite and 97.1% teacher-forcing locality, but only 23.3% PS@All. This supports
rewrite/locality stability for that profile, not solved paraphrase transfer.
```

```text
On three random n=200 EasyEdit-compatible CounterFact samples with the same
profile, AGIM WAL averages 93.2% teacher-forcing rewrite and 96.4%
teacher-forcing locality, but only 25.6% PS@All. This strengthens the
rewrite/locality hotfix claim and keeps the paraphrase limitation explicit.
```

```text
On the current official-compatible first-1000 CounterFact run with the default
locality-protected profile, AGIM WAL reaches 91.1% teacher-forcing rewrite and
96.2% teacher-forcing locality, but only 24.7% PS@All. This is evidence for
rewrite/locality scaling, not solved paraphrase generalization.
```

```text
On a random n=1000 EasyEdit-compatible CounterFact sample with seed 42 and the
same default profile, AGIM WAL reaches 94.5% teacher-forcing rewrite and 96.4%
teacher-forcing locality, but only 23.5% PS@All. This confirms that the
first-1000 scale result is not just an ordered-sample artifact.
```

```text
On three random n=50 sequential EasyEdit-compatible runs with the tuned profile,
AGIM WAL averages 100.0% teacher-forcing rewrite and 83.0% locality after 10
accumulated edits, then 78.7% rewrite and 33.9% locality after 50 edits. This is
partial sequential editing, not solved lifelong editing.
```

## Unsafe Language

Avoid:

```text
official leaderboard result
beats AlphaEdit/MEMIT
solved lifelong editing
AGI memory solved
```
