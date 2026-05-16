# AGI Personal Memory

**Accumulative verified memory substrate for language models.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-76%2F76-green)]()
[![Status: v10.0](https://img.shields.io/badge/version-v10.0-blue)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

[English](README.md) | [Русский](docs/ru/README.md) | [中文](docs/zh/README.md) | [Қазақша](docs/kk/README.md)

---

## What is this?

Today's AI systems are **stateless**. Every conversation starts from zero. You can't teach them a fact and have it stick. Fine-tuning is batch and destructive. RAG is retrieval, not learning.

**AGI Personal Memory** solves this: **incremental, verified, persistent knowledge accumulation** — the model itself learns, one fact at a time.

```
User: "Paris is the capital of France"
  → Fact extracted, 5 gates verified, committed to memory

User: "No, actually Napoleon was born in 1769"
  → Correction verified, old fact replaced, non-target diff = 0%

User: "When was Napoleon born?"
  → Answers from memory: "1769" ✓
```

## Why it's different

| | RAG | Fine-tuning | LoRA | AGI Personal Memory |
|---|---|---|---|---|
| **Changes model** | No | Yes (destructive) | Yes (additive) | **Yes (verified)** |
| **Incremental** | Yes | No | No | **Yes** |
| **Reversible** | Yes | No | Partially | **Yes (rollback any commit)** |
| **Auditable** | No | No | Partial | **Yes (full JSONL trail)** |
| **Non-target diff** | N/A | ~25% | Medium | **0%** (frozen vocabulary) |

## Quick Start

```bash
pip install -e .
agim teach "Paris is the capital of France"
agim ask "What is the capital of France?"
agim correct "No, Napoleon was born in 1769, not 1768"
agim history
agim stats
agim webui --port 8720
```

## Interfaces

| Interface | Command | Description |
|-----------|---------|-------------|
| **CLI** | `agim teach/ask/correct/forget/history/stats` | Command-line memory operations |
| **Shell** | `agim shell` | Interactive REPL |
| **REST API** | `agim api --port 8720` | 11 endpoints (teach/ask/verify/history/search) |
| **Web Dashboard** | `agim webui --port 8720` | Reactive JS dashboard with 5 tabs |
| **MCP** | `MCPServer` | Model Context Protocol — 5 tools for any MCP client |
| **A2A** | `A2AServer` | Agent-to-Agent protocol for multi-agent memory sharing |
| **GraphQL** | `GraphQLResolver` | GraphQL query interface |
| **Export** | `agim export memories.json` | Export all memories to JSON |
| **Import** | `agim import memories.json` | Import memories from JSON |

## How it works

```
User Input → IntentRouter (LLM + regex fallback) → Memory Extractor → MemoryCandidate
                                                                        ↓
                                                                   VERIFY (12 gates)
                                                                        ↓
                                                              Memory Compiler (5 tiers)
                                                                        ↓
                                                    ┌─ WAL recipe → model weight edit
                                                    ├─ Retrieval   → key-value store
                                                    ├─ LoRA        → orthogonal adapter
                                                    ├─ Refusal     → policy patterns
                                                    └─ Reject      → blocked
                                                                        ↓
                                                                   COMMIT + Audit Trail
```

## Architecture

```
src/agim/
├── core/          ← AGIMSystem, MultiUserAGIM, RecursiveImprovement, UniversalSubstrate
├── memory/        ← WAL recipes, FAISS+BM25, Knowledge Graph, SQLite, Distributed, Multimodal
├── verify/        ← 12 constitutional gates, contracts, regression, adversarial testing
├── learn/         ← SelfLearner, ReflectionEngine, Curriculum, Multi-Agent, Evolutionary
├── model/         ← WALWeightEditor, ROMEEditor, MEMITEditor, OLoRA, WISE, CrossModel
├── governance/    ← Provenance, Budget, RiskLedger, Constitutional, Watermarking
├── cli/           ← IntentRouter, LLMRouter, Extractor, API, WebUI, MCP, A2A, GraphQL
├── wal/           ← WAL core: encoder, decoder, ISA, v1, v2, backends, Triton kernels
└── dwl2/          ← Route encoder: calibrate, codebook, block VQ, runtime
```

## Roadmap

| Version | Key Feature | Status |
|---------|------------|--------|
| v0.1 | Core loop: propose→compile→commit→rollback | ✓ |
| v0.2 | LLM Intent Router + Structured Extractor | ✓ |
| v0.3 | WAL backend + ROMEEditor + Confidence v2 | ✓ |
| v0.4 | REST API + Multi-User + Docker | ✓ |
| v0.5 | Memory Testing Suite + Contract Regression | ✓ |
| v1.0 | ROME + MEMIT + O-LoRA + WISE + FAISS + SQLite | ✓ |
| v2.0 | Self-Learning + Reflection + Knowledge Graph | ✓ |
| v2.5 | Curriculum + PageRank + Memory Decay | ✓ |
| v3.0 | Multi-Agent: Teacher, Verifier, Researcher, Curator | ✓ |
| v4.0 | Multimodal Memory (text/image/audio/video) | ✓ |
| v5.0 | Distributed (CRDT + P2P) + Federated + Marketplace | ✓ |
| v6.0 | Constitutional Governance + Adversarial + Watermarking | ✓ |
| v7.0 | MCP + A2A + Plugin Marketplace | ✓ |
| v8.0 | Cognitive: Causal + Hypotheses + Counterfactuals | ✓ |
| v9.0 | Evolutionary: AutoOptimizer + Emergent Types + Cross-Model | ✓ |
| v10.0 | Recursive Self-Improvement + Safety Governor + AGIM-MEM | ✓ |

**76/76 tests. 15 on real Gemma-4-31B.**

## Links

- **GitHub:** https://github.com/AubakirovArman/agi-personal-memory
- **Pages:** https://aubakirovarman.github.io/agi-personal-memory/
- [Full Vision](VISION.md) | [Architecture](ARCHITECTURE.md) | [Roadmap](agim_roadmap_v0_to_v10.md)
- [Developer Diary](DIARY.md)
- Built on [WAL — Weight-Aligned Language](https://github.com/AubakirovArman/wal2026)

## License

MIT
