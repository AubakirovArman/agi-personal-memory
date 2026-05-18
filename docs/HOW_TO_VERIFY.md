# How To Verify AGIM

Legacy marker: this file is an index only. Public proofs should start from
`CURRENT_STATUS.md`, `BENCHMARK.md`, `docs/CLAIMS_AND_EVIDENCE.md`, and the two
path-specific verification docs.

Verification is split by claim type. Do not mix Path A memory checks with Path B
weight-editing evidence.

| Goal | Use |
| --- | --- |
| Verify runtime memory behavior | `docs/VERIFY_PATH_A.md` |
| Verify current EasyEdit-compatible Path B claims | `docs/VERIFY_PATH_B_CURRENT.md` |
| Inspect legacy WAL/ROME substrate tests | `docs/VERIFY_PATH_B_LEGACY.md` |

For public claims, start with `CURRENT_STATUS.md`, `BENCHMARK.md`, and
`docs/CLAIMS_AND_EVIDENCE.md`.

For execution, follow `docs/PATH_B_MAX_EXECUTION_CHECKLIST.md` and
`docs/PATH_B_MAX_EXECUTION_RUNBOOK.md` so verification and delivery gates stay aligned.
