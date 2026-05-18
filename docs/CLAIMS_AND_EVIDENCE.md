# Claims And Evidence

Use this file when deciding what AGIM can safely claim.

| Claim | Status | Evidence | Limitation |
| --- | --- | --- | --- |
| Path A stores and retrieves facts | Supported | `tests/`, `results/memory_retrieval/` | Runtime memory, not weight editing |
| Path A supports audit/rollback | Supported | `tests/test_core.py`, governance tests | AGIM layer rollback, not necessarily model-weight rollback |
| Path B single-edit works on n=50 | Supported internally | `results/easyedit_official/current/` | Profile tradeoff: high-PS settings hurt locality; locality-protected settings hurt PS@All |
| Path B sequential editing works | Partial | `results/easyedit_official/sequential/` | Random-seed retention falls by 50 edits |
| Historical 1000 local CounterFact diagnostic exists | Supported as history | `results/local_protocol/official_eval_1000.json` | Not official EasyEdit-compatible |
| AGIM is number one on EasyEdit | Unsafe | None | External validation missing |
| AGIM solved lifelong editing | Unsafe | None | Sequential retention/locality unresolved |

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
On the current official-compatible first-1000 CounterFact run with the default
locality-protected profile, AGIM WAL reaches 91.1% teacher-forcing rewrite and
96.2% teacher-forcing locality, but only 24.7% PS@All. This is evidence for
rewrite/locality scaling, not solved paraphrase generalization.
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
