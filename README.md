# AGI Personal Memory

**Accumulative verified memory substrate for language models.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

## What is this?

AGI Personal Memory solves the fundamental problem that today's AI systems are **stateless**. Every conversation starts from zero. You can't teach them a fact and have it stick. Fine-tuning is batch and destructive. RAG is retrieval, not learning.

This system lets you:
- **Teach** a fact → it's verified, committed, and never forgotten
- **Correct** a mistake → the old fact is replaced, non-target knowledge untouched
- **Ask** a question → answers from memory first, falls back to model
- **Forget** (rollback) → any change is reversible
- **Inspect** → full audit trail of every change

## How it works

```
User: "Paris is the capital of France"
  → Intent: FACT_TEACH
  → Extract: question="Capital of France?", answer="Paris"
  → Verify: no contradictions, no secrets, confidence ok
  → Compile: tier=wal_recipe
  → Commit: stored permanently

User: "No, actually Napoleon was born in 1769"
  → Intent: FACT_CORRECT
  → Extract: question="When was Napoleon born?", answer="1769"
  → Verify: correction overrides previous, all gates pass
  → Commit: old fact replaced

User: "When was Napoleon born?"
  → Memory lookup: found "1769" from wal_recipe
  → Answer: "1769" ✓
```

## Key Properties

| Property | How |
|----------|-----|
| **Incremental** | One fact at a time, not batch fine-tuning |
| **Verified** | Every change tested before commit |
| **Reversible** | Rollback any commit |
| **Auditable** | JSONL event log of every action |
| **Non-destructive** | Frozen vocabulary = 0% non-target diff |

## Quick Start

```bash
pip install -e .
agim teach "Paris is the capital of France"
agim ask "What is the capital of France?"
agim correct "No, Napoleon was born in 1769"
agim history
agim stats
agim shell
```

## Architecture

```
User Input → IntentRouter → MemoryExtractor → MemoryCandidate
                                              ↓
                                         VERIFY gates
                                              ↓
                                    MemoryCompiler (tier)
                                              ↓
                                    ┌─ WAL recipe (permanent)
                                    ├─ Retrieval (volatile)
                                    ├─ LoRA adapter (complex)
                                    └─ Reject / Refusal
                                              ↓
                                    COMMIT + Event Log
```

## Requirements

- Python 3.10+
- Optional: PyTorch, transformers (for HF model backend)

## License

MIT — see [LICENSE](LICENSE)

## Related

This project builds on [WAL (Weight-Aligned Language)](https://github.com/AubakirovArman/wal2026) — a research framework for representing neural network weights as structured programs.
