# AGI Personal Memory — Vision & Architecture

## Why A-G-I, not A-I-G-I

AIGI was "verified memory accumulation for AI." AGI Personal Memory is different: **AGI** here means the system
*accumulates* knowledge toward general intelligence, not that it *is* AGI. The name reflects ambition:
this is the memory substrate that a future AGI would need. Not a chatbot with RAG. Not a vector database.
A system that *learns* from every interaction and never forgets.

## The Core Problem

Today's AI systems are stateless. Every conversation starts from zero. You can't teach them. You can't correct
them. A month of daily use leaves no trace. This is fundamentally broken.

Fine-tuning exists but is batch, expensive, and destructive — it overwrites previous knowledge. RAG exists but
is retrieval, not learning — the model itself never changes.

**AGI Personal Memory** solves this: incremental, verified, persistent knowledge accumulation in the model itself.

## What It Does

```
User: "No, Napoleon was born in 1769, not 1768."
System: [proposes memory candidate: napoleon_birth_year=1769]
        [verifies: doesn't contradict existing facts]
        [compiles: creates WAL recipe for the edit]
        [tests: checks model now answers 1769, still answers other facts correctly]
        [commits: applies the change permanently]
        [logs: napoleon_birth_year=1769, 2026-05-16, confidence=1.0]

— 3 months and 10,000 corrections later —

User: "When was Napoleon born?"
Model: "1769"  ← because it LEARNED this, not because it was in training data
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    AGI Personal Memory                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  User Input ──→ Intent Router ──→ Memory Candidate      │
│       │              │                │                  │
│       │         ┌────┴────┐     ┌─────┴──────┐          │
│       │         │ Question │     │ Fact Update │          │
│       │         │ Preference│    │ Correction  │          │
│       │         │ Feedback │     │ New Concept │          │
│       │         └─────────┘     └─────┬───────┘          │
│       │                               │                  │
│       │                        ┌──────┴──────┐          │
│       │                        │  VERIFY     │          │
│       │                        │  - no conflict         │
│       │                        │  - confidence ok       │
│       │                        │  - not refusal         │
│       │                        └──────┬──────┘          │
│       │                               │                  │
│       │                        ┌──────┴──────┐          │
│       │                        │  COMPILE    │          │
│       │                        │  tier select:          │
│       │                        │  - WAL recipe          │
│       │                        │  - Retrieval           │
│       │                        │  - LoRA adapter        │
│       │                        │  - Reject              │
│       │                        └──────┬──────┘          │
│       │                               │                  │
│       │                        ┌──────┴──────┐          │
│       │                        │  TEST       │          │
│       │                        │  - exact match         │
│       │                        │  - negative check       │
│       │                        │  - context check        │
│       │                        │  - non-target diff=0%   │
│       │                        └──────┬──────┘          │
│       │                               │                  │
│       │                        ┌──────┴──────┐          │
│       │                        │  COMMIT     │          │
│       │                        │  - apply    │          │
│       │                        │  - log      │          │
│       │                        │  - version  │          │
│       │                        └──────┬──────┘          │
│       │                               │                  │
│       ▼                               ▼                  │
│  ┌─────────────────────────────────────────┐            │
│  │           Memory Store                   │            │
│  │  ┌────────┐ ┌──────────┐ ┌───────────┐  │            │
│  │  │ WAL    │ │ Retrieval │ │ LoRA      │  │            │
│  │  │ Recipes│ │ Memory   │ │ Adapters  │  │            │
│  │  └────────┘ └──────────┘ └───────────┘  │            │
│  │  ┌──────────────────────────────────┐    │            │
│  │  │     Event Log (JSONL audit)      │    │            │
│  │  └──────────────────────────────────┘    │            │
│  └─────────────────────────────────────────┘            │
│                                                         │
│  ┌─────────────────────────────────────────┐            │
│  │           Model Backend                  │            │
│  │  ┌──────────────────────────────────┐    │            │
│  │  │  Base Model (HF)                 │    │            │
│  │  │  + WAL Overlay (recipes)         │    │            │
│  │  │  + Retrieval Overlay (facts)     │    │            │
│  │  │  + LoRA Overlay (adapters)       │    │            │
│  │  └──────────────────────────────────┘    │            │
│  └─────────────────────────────────────────┘            │
│                                                         │
│  ┌─────────────────────────────────────────┐            │
│  │           Governance                     │            │
│  │  - Memory budget (max changes/day)       │            │
│  │  - Risk ledger (dangerous = rollback)    │            │
│  │  - Contract regression (don't break old) │            │
│  │  - Provenance chain (who changed what)   │            │
│  └─────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────┘
```

## Memory Tiers

| Tier | What | When | How |
|------|------|------|-----|
| `wal_recipe` | Stable facts, corrections | High confidence, permanent | WAL encode → model weight edit |
| `retrieval` | Volatile facts, updates | Medium confidence, may change | Vector store + retrieval |
| `lora_adapter` | Behavioral changes, style | Complex, needs gradient | LoRA training on WAL overlay |
| `refusal` | Unsafe requests | Policy violations | Refusal pattern stored |
| `reject` | Bad memory | Fails verification | Logged, not stored |

## Key Properties

1. **Incremental.** One fact at a time. Not batch fine-tuning.

2. **Verified.** Every change tested before commit. Exact match ("what year?"), negative check ("it's NOT 1768"), context check ("still answers other history correctly"), non-target diff (frozen vocabulary = 0%).

3. **Reversible.** Every commit has a rollback. Commit history is a linked list. Rollback restores previous state.

4. **Inspectable.** You can see every change: who, what, when, which layers were touched, what tests passed.

5. **Non-destructive.** Frozen vocabulary means editing one fact doesn't touch other knowledge. Unlike fine-tuning which overwrites everything.

6. **Auditable.** JSONL event log. Every propose/compile/commit/rollback recorded. Full provenance chain.

## What Makes This Different

| | RAG | Fine-tuning | LoRA | AGI Personal Memory |
|---|---|---|---|---|
| Model changes? | No | Yes (destructive) | Yes (additive) | Yes (verified) |
| Incremental? | Yes | No (batch) | No (batch) | Yes |
| Reversible? | Yes (delete doc) | No | Yes (remove adapter) | Yes (rollback any commit) |
| Auditable? | No | No | Partial | Yes (full JSONL trail) |
| Non-target diff? | N/A | High (~25%) | Medium | **0%** (frozen vocab) |
| Tests before apply? | No | No | No | **Yes** |
| Years of accumulation? | No (context window) | No (catastrophic forgetting) | Degrades with many adapters | **Yes** (designed for it) |

## Development Roadmap

### Phase 1: Core Memory Loop (weeks 1-2)
- [x] AIGISystem (propose → compile → commit → rollback) ← from WAL project
- [ ] Intent Router: classify user input as fact/preference/correction/question
- [ ] Memory Candidate Generator: extract structured fact from natural language
- [ ] CLI interface: `agim ask`, `agim teach`, `agim correct`, `agim forget`, `agim history`

### Phase 2: Verification & Testing (weeks 3-4)
- [ ] Behavioural Contract system: define test templates per fact
- [ ] Non-target regression suite: verify old facts still work after new commit
- [ ] Confidence scoring: how sure are we this fact is correct?
- [ ] Conflict detection: "you said X before, now you're saying Y"

### Phase 3: Model Backend (weeks 5-6)
- [ ] WAL recipe backend: apply frozen vocabulary edit to real model
- [ ] Retrieval backend: vector store for volatile facts
- [ ] LoRA adapter backend: gradient-based edits for complex changes
- [ ] Model serving: HF backend with memory overlay

### Phase 4: Governance & Safety (weeks 7-8)
- [ ] Memory budget: max facts/day, max facts total, decay for old facts
- [ ] Risk ledger: safety score per edit, auto-rollback dangerous edits
- [ ] Provenance chain: cryptographic signing of commits
- [ ] Multi-user: shared model with per-user memory overlay

### Phase 5: Interface & Experience (weeks 9-12)
- [ ] Web UI: timeline of memories, search, diff view
- [ ] Model diff visualization: heatmap of changed atoms
- [ ] Memory health dashboard: facts count, confidence distribution, risk scores
- [ ] Export/import: share memory sets between users

## The End State

After 12 months of use, AGI Personal Memory contains:
- 50,000+ verified facts about your domain
- 10,000+ corrections to model mistakes
- 5,000+ personal preferences and style adaptations
- Full audit trail of every change
- Zero degradation on original model capabilities (frozen vocab)
- The model is YOURS — it knows what you know, the way you know it

This is not a product. This is infrastructure for the next decade of AI:
models that learn, not just generate.
