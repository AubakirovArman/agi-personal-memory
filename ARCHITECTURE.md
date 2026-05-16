# AGI Personal Memory — Technical Architecture

## Starting Point: What We Already Have

From the WAL project (`/mnt/hf_model_weights/arman/3bit/wal/src/aigi/`):

```
aigi/
├── core/
│   ├── state.py      ← MemoryCandidate, CompileReport, MemoryPolicy, AIGIResponse, CommitRecord
│   └── system.py     ← AIGISystem: propose/compile/commit/rollback/ask
├── memory/
│   ├── wal_memory.py       ← WAL recipe write/remove
│   ├── retrieval_memory.py ← vector store for facts
│   └── compiler.py         ← MemoryCompiler: tier selection
├── verify/
│   ├── gates.py       ← MemoryVerifier: contradiction, confidence, secret scan, refusal
│   └── contracts.py   ← Behavioural contracts: must_answer, must_not_answer, must_refuse
├── learn/
│   ├── loop.py        ← VerifiedLearningLoop
│   └── experience.py  ← Experience → Lesson → MemoryCandidate
├── model/
│   ├── backends.py       ← TextModelBackend, StaticTextModelBackend
│   ├── huggingface.py    ← HuggingFaceTextBackend
│   ├── soft_prompt.py    ← SoftPromptAdapter
│   ├── logit_lora.py     ← LogitLoRAAdapter
│   └── module_lora.py    ← ModuleLoRAAdapter
├── governance/
│   ├── budget.py    ← MemoryBudget
│   ├── risk.py      ← RiskLedger
│   └── report.py    ← CommitDecisionReport
└── event_log.py     ← AIGIEventLog (JSONL)
```

## What We Need to Build

### Layer 1: Intent Router (NEW)

The current AIGI system takes structured MemoryCandidate objects. Users type natural language.
We need to parse intent from text.

```python
class IntentRouter:
    """Route user input to memory operations."""

    INTENTS = {
        "fact_teach": "User is teaching a new fact: 'Paris is the capital of France'",
        "fact_correct": "User is correcting: 'No, it's 1769 not 1768'",
        "fact_question": "User is asking: 'When was Napoleon born?'",
        "preference": "User expressing preference: 'I prefer short answers'",
        "feedback": "User giving feedback: 'That answer was wrong because...'",
        "forget": "User wants to forget: 'Forget what I said about X'",
        "history": "User wants history: 'What have I taught you?'",
    }

    def route(self, text: str) -> Intent:
        ...
```

### Layer 2: Memory Candidate Extractor (NEW)

```python
class MemoryExtractor:
    """Extract structured MemoryCandidate from natural language."""

    def extract(self, text: str, intent: Intent) -> MemoryCandidate:
        # "Paris is the capital of France"
        # → MemoryCandidate(
        #     question="What is the capital of France?",
        #     answer="Paris",
        #     kind="fact_teach",
        #     source="user",
        #     confidence=1.0
        #   )
        ...
```

### Layer 3: Enhanced Verification (EXTEND)

Add to existing gates:
- **Non-target regression test**: does the edit break existing facts?
- **Consistency check**: does the new fact contradict any existing memory?
- **Confidence calibration**: is the user likely to be right?

### Layer 4: Unified Store (EXTEND)

```python
class UnifiedMemoryStore:
    """Single interface over WAL + retrieval + LoRA + refusals."""

    def remember(self, candidate: MemoryCandidate) -> str:
        """Store a fact. Returns memory_id."""

    def recall(self, question: str) -> AIGIResponse:
        """Answer from memory if known."""

    def forget(self, memory_id: str) -> bool:
        """Remove a memory."""

    def history(self, limit: int = 100) -> list[CommitRecord]:
        """Timeline of changes."""

    def stats(self) -> MemoryStats:
        """Fact count, tier distribution, confidence histogram."""
```

### Layer 5: CLI Interface (NEW)

```bash
# Teach a fact
agim teach "Paris is the capital of France"

# Correct a mistake
agim correct "Napoleon was born in 1769, not 1768"

# Ask a question (checks memory first, falls back to model)
agim ask "When was Napoleon born?"

# View history
agim history --limit 20

# Show what changed
agim diff --memory-id 42

# Rollback
agim rollback --memory-id 42

# Stats
agim stats

# Export memories
agim export --format json --output my_memories.json
```

### Layer 6: Web Dashboard (NEW)

```
┌─────────────────────────────────────────────────────┐
│  AGI Personal Memory                    [Stats] [⚙] │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─ Memory Timeline ────────────────────────────┐   │
│  │ 16:42  ✓  napoleon_birth=1769     [diff]     │   │
│  │ 15:31  ✓  tokyo_population=14M    [diff]     │   │
│  │ 14:15  ✗  bad_fact (rolled back)  [why]      │   │
│  │ 13:00  ✓  user_prefers_short      [diff]     │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌─ Memory Stats ──────────────────────────────┐   │
│  │  Facts: 1,247    Corrections: 89             │   │
│  │  Preferences: 34    Refusals: 12              │   │
│  │  WAL: 892  Retrieval: 310  LoRA: 45          │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌─ Model View ────────────────────────────────┐   │
│  │  Layer 0   ████████░░  12 atoms changed      │   │
│  │  Layer 1   ██░░░░░░░░   2 atoms changed      │   │
│  │  ...                                         │   │
│  │  Non-target diff: 0.000%  ✓                   │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## Data Flow

```
User says: "Actually, Tokyo has 14 million people, not 37 million"

1. IntentRouter.route(text)
   → Intent.FACT_CORRECT

2. MemoryExtractor.extract(text, intent)
   → MemoryCandidate(
       question="What is the population of Tokyo?",
       answer="14 million",
       previous_answer="37 million",
       kind="fact_correction",
       source="user",
       confidence=0.9
     )

3. AIGISystem.compile(candidate)
   → MemoryCompiler.select_tier()
   → "wal_recipe" (stable fact, high confidence)
   → MemoryVerifier.evaluate()
     → Gate 1: no contradiction with existing facts ✓
     → Gate 2: confidence > 0.8 ✓
     → Gate 3: no secrets/refusals ✓
   → CompileReport(pass=True, tier="wal_recipe")

4. AIGISystem.commit(report)
   → WALMemory.write_recipe(candidate)
   → RetrievalMemory.upsert(question, answer)
   → CommitRecord stored in history

5. Future: "What's the population of Tokyo?"
   → AIGISystem.ask(question)
   → RetrievalMemory.lookup(question)
   → Found: "14 million" from memory_id=43
   → AIGIResponse(answer="14 million", source="wal_recipe")
```

## Storage Layout

```
~/.agi_personal_memory/
├── config.json              ← model, policy, budget settings
├── memories/
│   ├── wal_recipes/         ← WAL recipe artifacts (JSON)
│   │   └── {uuid}.json
│   ├── retrieval.json       ← key-value fact store
│   ├── refusals.json        ← refusal patterns
│   └── lora_adapters/       ← saved LoRA weights
│       └── {uuid}.pt
├── logs/
│   └── events.jsonl         ← every propose/compile/commit/rollback
├── governance/
│   ├── budget.json          ← memory budget state
│   └── risk_ledger.json     ← per-memory risk scores
├── contracts/
│   └── regression_suite.json ← protected facts that must survive
└── stats.json               ← aggregated statistics
```

## Scaling Strategy

### Now (1 user, 1 model, local)
- SQLite/JSON for memory store
- Single HF model backend
- Local event log

### Later (N users, shared model)
- PostgreSQL for memory store
- Model server with per-user WAL overlay
- Central event log with auth

### Eventually (distributed)
- Memory replication across nodes
- Cryptographic signing of commits
- Decentralized memory registry
