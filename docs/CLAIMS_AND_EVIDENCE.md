# Claims And Evidence

Use this file when deciding what AGIM can safely claim.

| Claim | Status | Evidence | Limitation |
| --- | --- | --- | --- |
| Path A stores and retrieves facts | Supported | `tests/`, `results/memory_retrieval/` | Runtime memory, not weight editing |
| Path A supports audit/rollback | Supported | `tests/test_core.py`, governance tests | AGIM layer rollback, not necessarily model-weight rollback |
| Path B single-edit works on n=50 | Supported internally | `results/easyedit_official/current/` | Locality still weak |
| Path B sequential editing works | Partial | `results/easyedit_official/sequential/` | Rephrase/locality weak |
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
teacher-forcing rephrase at n=50, with measured zero non-edited lm_head/embed
row diff. Sequential editing and locality remain open challenges.
```

## Unsafe Language

Avoid:

```text
official leaderboard result
beats AlphaEdit/MEMIT
solved lifelong editing
AGI memory solved
```
