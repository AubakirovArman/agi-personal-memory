# LoCoMo Benchmark — AGI Personal Memory Full Results

Stanford SNAP, ACL 2024. 10 conversations, 1,986 QA pairs, 5 reasoning categories.

## Methodology

### Pipeline
```
Question → MPNet Dense Retrieval + Entity Boost → GPT-4o-mini Answer Generation → F1 Evaluation
```

### Components
- **Retrieval:** MPNet (all-mpnet-base-v2, 768-dim) dense embeddings + entity linking (spaCy PERSON/GPE/ORG/DATE) with 60/40 fusion weights
- **Multi-hop:** iterative retrieval — extract entities from first pass, re-query with expanded context
- **Generation:** GPT-4o-mini (gpt-4o-mini-2024-07-18), temperature=0.0, max 80 tokens
- **Evaluation:** Per-category metrics matching LoCoMo paper protocol

### Evaluation per Category
| Category | Metric | Description |
|----------|--------|-------------|
| Cat 1 (multi-hop) | F1 on first sub-answer | Multi-session reasoning |
| Cat 2 (temporal) | F1 word overlap | Time-based questions |
| Cat 3 (open-domain) | F1 word overlap | Persona + world knowledge |
| Cat 4 (single-hop) | F1 word overlap | Fact from one session |
| Cat 5 (adversarial) | "Not mentioned" check | Unanswerable questions |

## Results

### Overall: 39.0% F1 Score (774.5 / 1,986)

### Per Category
| Category | F1 Score | QA Pairs | Performance |
|----------|----------|----------|-------------|
| Cat 5 (adversarial) | **94.4%** | 446 | ⭐ Excellent — correctly identifies unanswerable |
| Cat 4 (single-hop) | 31.5% | 841 | Good — finds explicit facts |
| Cat 1 (multi-hop) | 19.0% | 282 | Needs multi-hop reasoning improvement |
| Cat 3 (open-domain) | 10.4% | 96 | Hard — requires world knowledge |
| Cat 2 (temporal) | 7.8% | 321 | Hardest — temporal reasoning needed |

### Per Sample
| Sample | Sessions | QA Pairs | F1 Score |
|--------|----------|----------|----------|
| conv-26 | 19 | 199 | 33.5% |
| conv-30 | 19 | 105 | 35.2% |
| conv-41 | 32 | 193 | 33.2% |
| conv-42 | 29 | 260 | 33.5% |
| conv-43 | 29 | 242 | 37.0% |
| conv-44 | 28 | 158 | 32.4% |
| conv-47 | 31 | 190 | 33.7% |
| conv-48 | 30 | 239 | 32.6% |
| conv-49 | 25 | 196 | 36.2% |
| conv-50 | 30 | 204 | 32.4% |

## Progression

| Iteration | Method | Score | Delta |
|-----------|--------|-------|-------|
| 1 | BM25 keyword only | 14.3% | — |
| 2 | MPNet dense only | 22.7% | +8.4pp |
| 3 | GPT-4o-mini generation | 39.0% | +16.3pp |

## Competitive Landscape

| System | LoCoMo Score | Key Technology |
|--------|-------------|----------------|
| MemU | 92.1% | GPT-4o-mini + Advanced Augmentation |
| MemMachine v0.2 | 91.7% | GPT-4.1-mini agent mode |
| Mem0 (new) | 91.6% | GPT-4o-mini + Entity Graph |
| Full Context | 87.5% | Entire dialog in prompt (upper bound) |
| Memori | 82.0% | Semantic triples + hybrid ranker |
| Zep | 79.1% | Temporal Knowledge Graph |
| Continua | 74.4% | Hebbian retrieval + KG |
| **AGIM** | **39.0%** | GPT-4o-mini + MPNet + Entity Boost |
| — | — | — |
| AGIM (dense only) | 22.7% | MPNet retrieval, no generation |
| AGIM (BM25 only) | 14.3% | Keyword matching, no generation |

## Gap Analysis: 39% → 80%+

| Improvement | Expected Gain | Cumulative |
|-------------|--------------|------------|
| Session-level indexing (not turn-level) | +15pp | 54% |
| LLM-as-judge (semantic F1, not word F1) | +10pp | 64% |
| Better retriever (Contriever/Dragon+) | +8pp | 72% |
| Temporal reasoning (date extraction) | +6pp | 78% |
| Multi-hop iterative improvement | +5pp | 83% |

## Key Insight

AGIM's **unique advantage** over all competitors: **Path B — weight editing.**
None of Mem0, MemU, Zep, or Continua can edit model parameters. AGIM can.
This benchmark tests Path A (retrieval memory). Path B (ROME/MEMIT/WAL editing)
is a separate benchmark (CounterFact, MQuAKE) where AGIM has no competitors.

## Files
- `locomo_gpt_full.json` — full 1986 QA results
- `src/agim/cli/locomo_benchmark.py` — benchmark runner
