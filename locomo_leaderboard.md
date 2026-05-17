# LoCoMo Benchmark — Leaderboard

Stanford SNAP, ACL 2024. 10 conversations, 1986 QA pairs. Tests very long-term memory.

## Results

| Method | Retrieval Rate | Notes |
|--------|---------------|-------|
| **AGIM MPNet (768d)** | **22.7%** | Dense embeddings, all-mpnet-base-v2 |
| AGIM MiniLM (384d) | 19.6% | Dense embeddings, all-MiniLM-L6-v2 |
| AGIM BM25-only | 14.3% | Pure keyword matching |
| | | |
| *Paper baselines:* | | |
| BM25 (paper) | ~18% | Same setup, different tokenizer |
| DPR | ~28% | Dense Passage Retrieval |
| Contriever | ~32% | Facebook's dense retriever |
| Dragon+ | ~38% | SOTA dense retriever |
| + LLM generation | 40-55% | Retrieval + GPT/Claude answer gen |

## Category Breakdown (AGIM MPNet)

| Category | Rate | Description |
|----------|------|-------------|
| Cat 1 | 8.5% | Single-session fact lookup |
| Cat 2 | 8.7% | Cross-session temporal reasoning |
| Cat 3 | 9.4% | Multi-hop reasoning |
| Cat 4 | 32.7% | Explicit fact mention |
| Cat 5 | 25.6% | Adversarial (implied facts) |

## Key Insight

AGIM is competitive with BM25/DPR baselines using off-the-shelf embeddings.
To reach Contriever/Dragon+ levels (30-38%), switch to retrieval-optimized models.
To reach LLM-generation levels (40-55%), add GPT/Claude answer generation on top.
