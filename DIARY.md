# Developer Diary — AGI Personal Memory

## 2026-05-16 — Project Initialization

### Setup
- Created project at `/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory`
- GitHub repo: `AubakirovArman/agi-personal-memory` (public, MIT)
- Python package: `agim` (pip install -e .)

### Architecture designed
- Intent Router: classifies user input (teach/ask/correct/preference/feedback/forget/history/stats)
- Memory Extractor: regex-based extraction of structured facts from natural language
- Verification Pipeline: 5 gates (non_empty, confidence_range, no_secrets, no_contradiction, refusal_shape)
- Memory Tiers: WAL_RECIPE (permanent), RETRIEVAL (volatile), LORA_ADAPTER, REFUSAL, REJECT
- Core System: AGIMSystem with propose → compile → commit → rollback cycle
- Event Log: JSONL audit trail
- CLI: agim teach/ask/correct/forget/history/stats/shell

### Files created
- `src/agim/core/state.py` — MemoryCandidate, CompileReport, Intent, MemoryTier, CommitRecord, MemoryStats
- `src/agim/core/system.py` — AGIMSystem: propose_memory, compile, commit, rollback_last, ask, stats
- `src/agim/event_log.py` — JSONL event logger
- `src/agim/memory/retrieval_memory.py` — Key-value volatile fact store
- `src/agim/memory/wal_memory.py` — WAL recipe artifact store
- `src/agim/memory/compiler.py` — Tier selection router
- `src/agim/verify/gates.py` — 5 verification gates
- `src/agim/cli/intent_router.py` — Pattern-based intent classifier
- `src/agim/cli/extractor.py` — Regex-based fact extractor
- `src/agim/cli/main.py` — Full CLI (teach/ask/correct/forget/history/stats/shell)
- `src/agim/model/backends.py` — StaticTextModelBackend + HuggingFaceTextBackend
- `src/agim/learn/experience.py` — Experience → Lesson → MemoryCandidate
- `src/agim/governance/budget.py` — Memory budget with daily/hourly/total limits
- `src/agim/governance/risk.py` — Risk ledger with auto-rollback for dangerous edits
- `src/agim/governance/report.py` — Commit decision report
- `tests/test_core.py` — 16 tests covering full workflow
- `VISION.md` — Project vision and long-term roadmap
- `ARCHITECTURE.md` — Technical architecture and data flow
- `README.md` — Public README with quick start
- `LICENSE` — MIT
- `pyproject.toml` — Package config with CLI entry point

### Bugs fixed
1. Secret scanning regex missed "sk-" prefixed API keys → added to SECRET_PATTERNS
2. MemoryExtractor inverted subject/attribute in "X is Y" patterns → fixed question/answer assignment

### Test results
- 16/16 tests pass (0.05s)
- Coverage: system init, propose/compile/commit, ask after commit, rollback, contradiction detection, correction override, secret scanning, stats, event log, intent router, fact extraction, correction extraction, memory compiler, retrieval memory, WAL memory, full workflow

### Next steps
- [ ] Create own venv for agi_personal_memory
- [ ] GitHub Pages + Wiki
- [ ] Real HF model backend test
- [ ] Web dashboard

## 2026-05-16 (evening) — Phase 1 complete

### Added
- VerifiedLearningLoop: feedback → extract → verify → commit cycle
- Behavioural contracts: must_answer, must_refuse, exact/contains/not_contains checks
- GitHub Pages: deployed at aubakirovarman.github.io/agi-personal-memory
- 3 new tests (lesson extraction positive/negative, learning loop)

### Results
- 19/19 tests pass
- GitHub repo: public, MIT license
- GitHub Pages: live
- 2 commits pushed

### Bugs
- venv needed separate pytest install

### Architecture decisions
- Regex-based extraction is fast but limited. Future: LLM-based extraction.
- Default tier routing: fact_teach/correct → WAL_RECIPE, preference/feedback → RETRIEVAL
- Risk ledger with auto-rollback for score >= 8.0
- Memory budget: 100K total, 500/day, 50/hour

## 2026-05-16 (night) — Phases 2-5 implemented

### Phase 2: Verification & Testing ✓
- Regression suite with protected facts
- Confidence scorer (source-aware, history-aware)
- 5 new verify tests

### Phase 3: Model Backend ✓
- WALWeightEditor: frozen vocabulary edits with non-target diff verification
- MemoryOverlay: serves from memory first, falls back to model
- HuggingFaceTextBackend with local_files_only

### Phase 4: Governance & Safety ✓
- ProvenanceChain: cryptographic chain of memory commits
- MemoryBudget: total, daily, hourly limits
- RiskLedger: auto-rollback for dangerous edits (score >= 8.0)
- 7 new governance tests

### Phase 5: Interface ✓
- Export/import memories (JSON format)
- CLI: teach/ask/correct/forget/history/stats/shell/export/import
- GitHub Pages landing page

### Results
- 31/31 tests pass (0.08s)
- GitHub Pages live
- Full provenance chain with verification
- WAL frozen vocabulary integration

### Known limitations
- WALWeightEditor needs real model on GPU for production use
- IntentRouter is regex-based, not LLM-based
- Web dashboard is static, not dynamic
- Multi-user overlay not implemented yet
