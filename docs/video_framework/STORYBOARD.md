# AGI Personal Memory — Video Framework / Storyboard

## Overview

**Duration:** 8–12 minutes  
**Style:** Technical demo + architectural explanation  
**Audience:** AI engineers, ML researchers, open-source community  
**Goal:** Show that incremental model editing is real, working, and open-source

---

## Scene 1: The Problem (0:00–1:00)

**Visual:** ChatGPT/Claude conversation. User corrects a fact. Next conversation — model forgot.

**Narration:** "Every AI conversation starts from zero. You correct a fact today — tomorrow it's gone. Fine-tuning rewrites everything. RAG doesn't learn. What if you could teach a model one fact, and it would remember it forever?"

**Transition:** Logo: AGI Personal Memory

---

## Scene 2: The Core Idea (1:00–2:00)

**Visual:** Animation showing the propose→compile→commit→rollback cycle as a circular flow diagram.

```
   PROPOSE ──→ COMPILE ──→ COMMIT
      ↑                      │
      └──── ROLLBACK ←───────┘
```

**Narration:** "AGI Personal Memory is a memory substrate for language models. It accumulates knowledge one fact at a time. Every change is verified before it's committed. Every commit can be rolled back. Full audit trail."

---

## Scene 3: Live Demo — CLI (2:00–4:00)

**Visual:** Terminal. Live typing.

```bash
$ agim teach "Zanikland's capital is Blorptown"
  MEMORIZED [a3f2] What is the capital of Zanikland? → Blorptown
  tier=wal_recipe

$ agim ask "What is the capital of Zanikland?"
  Q: What is the capital of Zanikland?
  A: Blorptown
  source=wal_recipe

$ agim correct "No, Zanikland's capital is actually Florpville"
  CORRECTED → Florpville

$ agim ask "What is the capital of Zanikland?"
  A: Florpville

$ agim forget
  ROLLED BACK

$ agim ask "What is the capital of Zanikland?"
  A: Blorptown  ← back to original
```

**Narration:** "Let's teach it a fictional fact — something no model could know. Zanikland doesn't exist. Its capital is Blorptown. Now it knows. We correct it. It changes. We roll back. It reverts. Full cycle."

---

## Scene 4: Architecture Overview (4:00–5:30)

**Visual:** Zoom out to architecture diagram showing all layers.

```
┌────── CLI ──────┐  ┌── REST API ───┐  ┌── WebUI ──┐  ┌── MCP ──┐
│ agim teach       │  │ 11 endpoints   │  │ Dashboard │  │ 5 tools  │
│ agim ask         │  │ ask/teach/     │  │ 5 tabs    │  │ any MCP  │
│ agim correct     │  │ verify/history │  │ live JS   │  │ client   │
└──────────────────┘  └────────────────┘  └───────────┘  └──────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │        AGIMSystem               │
                    │  propose → compile → commit     │
                    │  → rollback → ask → stats       │
                    └───────────────┬────────────────┘
                                    │
        ┌───────────────┬───────────┼───────────┬──────────────┐
        ▼               ▼           ▼           ▼              ▼
    VERIFY          COMPILER     MEMORY       LEARN       GOVERNANCE
    12 gates        5 tiers     WAL+FAISS    SelfLearner  Provenance
    constitutional  wal_recipe  +SQLite+KG   +Reflection  +Budget
    +adversarial    retrieval   +Distributed +Curriculum  +Risk+Ledger
                    LoRA+Reject +Multimodal  +MultiAgent  +Watermarking
```

**Narration:** "The system has 7 layers. CLI, REST API, WebUI, and MCP on top. The core cycle in the middle. Verification with 12 constitutional gates. Five memory tiers — from WAL recipes that edit model weights, to retrieval, LoRA adapters, refusals, and rejects. Self-learning engine. Full governance."

---

## Scene 5: How WAL Editing Works (5:30–7:00)

**Visual:** Animation of neural network weights being encoded into atoms.

```
weights [0.023, -0.451, 0.789, ...] → k-means++ → 256 atoms

weight = atom[atom_id] × coeff[coeff_id] + residual

Frozen vocabulary: build once, never rebuild
→ Edit one layer → other layers unchanged → non-target diff = 0%
```

**Narration:** "The key technology is WAL — Weight-Aligned Language. Each weight becomes a program: atom times coefficient plus residual. We build the atom vocabulary once and freeze it. After that, editing one layer changes only that layer — non-target diff equals exactly zero percent."

---

## Scene 6: Interfaces Tour (7:00–8:30)

**Visual:** Split screen showing:
1. Web Dashboard: Ask tab, user types a question, result appears on screen
2. REST API: curl commands with JSON responses
3. MCP: Claude Desktop connected, asks about stored facts

**Narration:** "AGI Personal Memory has multiple interfaces. Web dashboard with live search and governance monitoring. REST API with 11 endpoints. MCP server that any MCP-compatible client can use. A2A protocol for agent-to-agent memory sharing. GraphQL for flexible queries."

---

## Scene 7: What Was Proven (8:30–9:30)

**Visual:** Test results overlay on Gemma-4-31B architecture diagram.

```
✓ 76/76 tests pass
✓ 15 tests on real Gemma-4-31B (GPU)
✓ Frozen vocabulary: non-target diff = 0%
✓ AGIM teach → WAL edit → model change → rollback → restore
✓ 62 modules, ~5,000 lines, all <300 lines
```

**Narration:** "We tested this on Gemma-4-31B — a real 31-billion parameter model. WAL encodes its weights. Frozen vocabulary gives zero percent non-target diff. AGIM teaches fictional facts, edits the model, and rollback restores the original state. 76 tests pass, 15 on real hardware."

---

## Scene 8: What's Next / Call to Action (9:30–10:30)

**Visual:** GitHub repo stars animation. Community contributions graphic.

**Narration:** "This is open-source. MIT license. The Roadmap goes to v10.0 — distributed memory, cognitive reasoning, recursive self-improvement. What's next depends on you. Star the repo. Teach it a fact. Build an integration. Memory for AGI starts here."

**End screen:**
```
github.com/AubakirovArman/agi-personal-memory
AGI Personal Memory v10.0
MIT License
```

---

## Production Notes

### Visual Style
- Dark theme (matching GitHub dark mode)
- Code in monospace font
- Architecture diagrams as clean SVG
- Blue/orange color scheme (matches logo)

### Audio
- Clear narration (no background music during technical parts)
- Terminal typing sounds for CLI demo
- Subtitles in 4 languages (EN/RU/ZH/KK)

### Recording Setup
- Terminal: kitty + tmux for clean window management
- Web Dashboard: browser window at 1280×720
- Architecture diagrams: Excalidraw or draw.io SVG export
- Screen recording: OBS Studio

### Post-Production
- DaVinci Resolve for editing
- Chapters for YouTube navigation
- Thumbnail: AGI Personal Memory logo on dark background

---

## Key Messages

1. **AI models can learn incrementally** — one fact at a time, verified, reversible
2. **0% non-target diff is real** — frozen vocabulary editing works on 31B models
3. **Open-source, production-ready** — 76 tests, 15 interfaces, MIT license
4. **Not RAG, not fine-tuning, not LoRA alone** — it's actual model editing
5. **This is infrastructure for the future of AI** — models that learn, not just generate
