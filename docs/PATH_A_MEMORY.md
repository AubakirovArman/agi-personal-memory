# Path A: Verified Runtime Memory

Path A is the product-like part of AGI Personal Memory. It stores, verifies,
audits, retrieves, corrects, and rolls back facts through AGIM memory stores.

## What It Does

- Accepts teach/correct/forget/ask operations through the CLI and library API.
- Runs verification gates before commit.
- Stores accepted facts in memory tiers.
- Keeps JSONL audit/provenance records.
- Supports rollback at the AGIM memory layer.

## What It Does Not Prove

- It does not prove that base model weights changed.
- It does not support EasyEdit, ROME, MEMIT, or AlphaEdit claims.
- It should not be compared with weight-editing leaderboards.

## Evidence

- Unit and integration tests under `tests/`.
- Memory/retrieval artifacts under `results/memory_retrieval/`.
- Current full local suite: `135 passed, 13 skipped`.

## Safe Claim

```text
AGIM Path A supports auditable runtime memory: facts can be stored, retrieved,
corrected, and rolled back through the AGIM memory layer.
```

## Unsafe Claim

```text
Path A proves model-weight editing or EasyEdit performance.
```
