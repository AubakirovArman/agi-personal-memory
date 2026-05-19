# AGI Personal Memory

**Accumulative verified memory substrate for language models.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-154%20passed%2C%2013%20skipped-green)]()
[![Status](https://img.shields.io/badge/status-research%20prototype-blue)]()
[![Version](https://img.shields.io/badge/version-0.2.0a1-blue)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

[English](README.md) | [Русский](docs/ru/README.md) | [中文](docs/zh/README.md) | [Қазақша](docs/kk/README.md)

---

## What is this?

Today's AI systems are **stateless**. Every conversation starts from zero. You
can't teach them a fact and have it stick. Fine-tuning is batch and destructive.
RAG is retrieval, not learning.

**AGI Personal Memory** is an experimental verified-memory substrate with two
separate paths:

- **Path A:** retrieval memory for persistent teach/ask behavior.
- **Path B:** WAL-backed weight editing for research-grade model edits.

```
User: "Paris is the capital of France"
  → Fact extracted, 5 gates verified, committed to memory

User: "No, actually Napoleon was born in 1769"
  → Correction verified, old fact replaced, non-target diff = 0%

User: "When was Napoleon born?"
  → Answers from memory: "1769" ✓
```

## Evaluation Status

Current EasyEdit-style weight-editing claims must come from the real
EasyEdit-compatible runner, not the older local CounterFact scripts.

| Protocol | Current status | Where |
| --- | --- | --- |
| Real EasyEdit-compatible CounterFact | Single-edit strong; sequential/locality still weak | [BENCHMARK.md](BENCHMARK.md) |
| Legacy local CounterFact | Historical diagnostics only, including the 1000-fact `ES=91.4%` run | [results/local_protocol](results/local_protocol) |
| Memory/retrieval | Path A storage and retrieval tests, not weight editing | [results/memory_retrieval](results/memory_retrieval) |

Current n=50 EasyEdit-compatible single-edit result on
`meta-llama/Llama-3.1-8B-Instruct`:

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing | 100.0% | 71.0% | 67.0% | 58.4% |
| Contextual generation | 100.0% | 70.0% | 66.0% | n/a |
| Probability compare | 100.0% | 88.0% | 89.0% | 37.4% |

Current n=1000 EasyEdit-compatible scale check for the default locality profile:

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing | 91.1% | 25.4% | 24.7% | 96.2% |
| Contextual generation | 91.0% | 24.8% | 24.1% | n/a |
| Probability compare | 96.3% | 43.5% | 43.6% | 87.5% |

Current sequential-tuned profile (single run shown, plus random-seed mean in
CURRENT_STATUS.md and linked verification docs):

Current n=50 sequential tuned profile:

| Metric group | Rewrite | Rephrase | PS@All | Locality |
| --- | ---: | ---: | ---: | ---: |
| Teacher-forcing | 73.0% | 21.0% | 20.0% | 25.4% |
| Probability compare | 86.0% | 62.0% | 61.0% | 61.4% |

This is useful progress, but it does not support a claim that AGIM is #1 on
EasyEdit or has solved lifelong/sequential editing.

For the current source of truth, read [CURRENT_STATUS.md](CURRENT_STATUS.md).

## Why it's different

| | RAG | Fine-tuning | LoRA | AGI Personal Memory |
|---|---|---|---|---|
| **Changes model** | No | Yes (destructive) | Yes (additive) | **Experimental** |
| **Incremental** | Yes | No | No | **Yes** |
| **Reversible** | Yes | No | Partially | **Yes (rollback any commit)** |
| **Auditable** | No | No | Partial | **Yes (full JSONL trail)** |
| **Non-target diff** | N/A | ~25% | Medium | **0% in WAL diagnostics** |

## Quick Start: Path A Runtime Memory

```bash
pip install -e ".[eval,api]"
agim teach "Paris is the capital of France"
agim ask "What is the capital of France?"
agim correct "No, Napoleon was born in 1769, not 1768"
agim history
agim stats
agim webui --port 8720
```

This verifies the runtime memory plane. It does not prove model-weight editing.
For details, read [Verify Path A](docs/VERIFY_PATH_A.md).

## Quick Start: Path B Weight Editing Evaluation

Path B claims use the EasyEdit-compatible runner and explicit artifacts.

```bash
export AGIM_MODEL=meta-llama/Llama-3.1-8B-Instruct
export AGIM_DEVICE=cuda
export AGIM_LOCAL_FILES_ONLY=0
export AGIM_EASYEDIT_ROOT="<YOUR_EASYEDIT_REPO_PATH>"

PYTHONPATH=src python -m agim.eval.easyedit_official_runner \
  --n 1000 --sample-policy first \
  --model "$AGIM_MODEL" --device "$AGIM_DEVICE" \
  --easyedit-root "$AGIM_EASYEDIT_ROOT" \
  --output results/easyedit_official/current/easyedit_official_1000_first_default.json \
  --save-failures-only
```

For reproducibility bundles and caveats, read
[Verify Path B Current](docs/VERIFY_PATH_B_CURRENT.md). For old WAL/ROME
substrate checks, read [Verify Path B Legacy](docs/VERIFY_PATH_B_LEGACY.md).

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
├── model/         ← WAL editors, SideSlotMemory, PatchService, MEMIT/ROME foundations
├── governance/    ← Provenance, Budget, RiskLedger, Constitutional, Watermarking
├── cli/           ← IntentRouter, LLMRouter, Extractor, API, WebUI, MCP, A2A, GraphQL
├── wal/           ← WAL core: encoder, decoder, ISA, v1, v2, backends, Triton kernels
└── dwl2/          ← Route encoder: calibrate, codebook, block VQ, runtime
```

## Repository Map

| Path | Purpose |
| --- | --- |
| `src/agim/` | Supported library and CLI code |
| `src/agim/eval/` | Current evaluation entry points |
| `tests/` | Supported pytest suite |
| `results/easyedit_official/current/` | Current EasyEdit-compatible single-edit artifacts |
| `results/easyedit_official/sequential/` | Sequential EasyEdit-compatible artifacts and limitations |
| `results/easyedit_official/ablations/` | Historical EasyEdit-compatible tuning artifacts |
| `results/local_protocol/` | Legacy AGIM-local CounterFact artifacts |
| `results/memory_retrieval/` | Path A memory/retrieval artifacts |
| `experiments/legacy_weight_editing/` | Archived one-off scripts and sweeps |

## Roadmap

| Scope | Status | Notes |
|---|---|---|
| Path A runtime memory (retrieval/store/retrieve) | Supported | Core project plane is stable and documented in `docs/PATH_A_MEMORY.md`. |
| Path B EasyEdit-compatible single-edit baseline | Implemented | Baseline artifacts are in `results/easyedit_official/current/` and `results/easyedit_official/ablations/`. |
| Path B sequential editing hardening | In progress | Sequential and locality are the current weakness; side-slot and relation-aware modes are being measured. |
| External consequence evidence (RippleEdits/MQuAKE/product) | In progress (partial) | Adapter chain exists; tracked model-output+score artifacts are still being completed. |
| Deep-edit governance proof packet | In progress | `PatchService` and `PatchGovernance` are in place; release-gate lifecycle documentation is still open. |

Historical v0-to-v10 vision is kept as archival reference in
[`agim_roadmap_v0_to_v10.md`](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/agim_roadmap_v0_to_v10.md).

Current full local suite: **154 passed, 13 skipped** on 2026-05-18.
The skipped tests are Gemma E2E checks when the installed Transformers build
does not support the local `gemma4` checkpoint architecture.

## Links

- **GitHub:** https://github.com/AubakirovArman/agi-personal-memory
- **Pages:** https://aubakirovarman.github.io/agi-personal-memory/
- [Current Status](CURRENT_STATUS.md) | [Benchmarks](BENCHMARK.md) | [Evaluation Protocols](docs/evaluation/README.md)
- [Path A Memory](docs/PATH_A_MEMORY.md) | [Path B Weight Editing](docs/PATH_B_WEIGHT_EDITING.md) | [Claims And Evidence](docs/CLAIMS_AND_EVIDENCE.md)
- [EasyEdit Protocol](docs/EASYEDIT_PROTOCOL.md) | [Realistic Roadmap](docs/ROADMAP_REALISTIC.md)

## Path B Progress Snapshot

The current hard-gate status is maintained in:
- [PATH_B_MAX_EXECUTION_CHECKLIST.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_EXECUTION_CHECKLIST.md)
- [PATH_B_COMPLETION_AUDIT.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_COMPLETION_AUDIT.md)
- [PATH_B_MAX_COMPLETION_MATRIX.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_COMPLETION_MATRIX.md)
- [PATH_B_MAX_STATUS_BOARD.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_STATUS_BOARD.md)

Current hard blockers:
- Gate 1–4 artifacts are present (`wal_memit` n=50, backend matrix, side-slot 10/50/100×seeds) and can be cited in the completion matrix.
- Hard evidence path still open: public PatchService/governance release packet and claims-safe public-facing lifecycle contract.

These blockers are tracked by:
- [PATH_B_MAX_ACTION_PLAN.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_ACTION_PLAN.md)
- [PATH_B_MAX_HARDGATE_QUEUE.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_HARDGATE_QUEUE.md)
- [PATH_B_MAX_GATE_COMMANDS.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_GATE_COMMANDS.md)
- [PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md)
- [PATH_B_MAX_EXECUTION_BLUEPRINT.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_EXECUTION_BLUEPRINT.md)
- [PATH_B_MAX_EXECUTION_MANIFEST.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_EXECUTION_MANIFEST.md)
- [Path B Productization Plan](docs/PATH_B_PRODUCTIZATION_PLAN.md)
- [Path B Maximal Checklist](docs/PATH_B_MAX_EXECUTION_CHECKLIST.md)
- [Maximal Runbook](docs/PATH_B_MAX_EXECUTION_RUNBOOK.md)
- [Verify Path B Current](docs/VERIFY_PATH_B_CURRENT.md)
- [Full Vision](VISION.md) | [Architecture](ARCHITECTURE.md) | [Historical Roadmap](agim_roadmap_v0_to_v10.md)
- [Developer Diary](DIARY.md)
- Built on [WAL — Weight-Aligned Language](https://github.com/AubakirovArman/wal2026)

Run the official Path B hard-gate chain in canonical order from repo root:

```bash
bash scripts/run_path_b_max_bootstrap.sh 1 2 4 3 5
```

Run on one GPU index (`2` or `3`) and clear previous runs if needed:

```bash
export AGIM_DEVICE="cuda:2"   # or cuda:3
AGIM_CLEAN_GPU=1 AGIM_FORCE_KILL_GPU=1 bash scripts/run_path_b_max_bootstrap.sh 1 2 4 3 5
```

## License

MIT
