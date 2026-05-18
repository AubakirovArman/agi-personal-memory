# Verify Path A Runtime Memory

Path A verifies AGIM runtime memory: storage, retrieval, correction, history,
export, and audit. It does not prove model weight editing.

## CLI Smoke

```bash
pip install -e ".[api]"
agim teach "Paris is the capital of France"
agim ask "What is the capital of France?"
agim correct "No, Napoleon was born in 1769, not 1768"
agim history
agim export /tmp/agim_memories.json
```

Expected claim: the fact is present in AGIM-managed memory and history. This is
not an EasyEdit, ROME, MEMIT, or model-weight claim.

## Test Gate

```bash
PYTHONPATH=src python -m pytest tests/test_core.py tests/test_memory_retrieval.py -q
```

For the current full-suite status, read `CURRENT_STATUS.md`.

## Valid Path A Claims

- AGIM can store and retrieve facts through its runtime memory layer.
- AGIM memory operations are auditable through history/provenance surfaces.
- Path A results can support memory-layer claims, not weight-editing claims.
