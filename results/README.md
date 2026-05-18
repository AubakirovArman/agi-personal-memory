# Results Directory

This directory is intentionally split by evaluation protocol. Do not compare
numbers across folders without reading the protocol notes.

| Folder | Meaning | Can support official EasyEdit-style claims? |
| --- | --- | --- |
| `easyedit_official/` | Runs evaluated through the local EasyEdit functions via `agim.eval.easyedit_official_runner` | Yes, with the stated caveats |
| `local_protocol/` | Historical AGIM/CounterFact local evaluators, including files named `official_eval_*` | No |
| `memory_retrieval/` | Path A retrieval-memory benchmarks: AGIM store/lookup, SQuAD recall, LoCoMo retrieval | No, these are not weight editing |
| `other_benchmarks/` | Early KnowEdit, MQuAKE, WikiBio runs | No, diagnostic only |

Current evaluation summary:

- Main benchmark document: `../BENCHMARK.md`
- Detailed EasyEdit status: `easyedit_official/easyedit_agim_status_2026-05-18.md`
- Legacy protocol caveats: `local_protocol/README.md`

