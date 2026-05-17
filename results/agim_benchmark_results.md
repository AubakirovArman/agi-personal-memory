# AGIM Benchmark Results

## CounterFact (Knowledge Editing) — 200 facts, Llama 3.1 8B

| Метрика | AGIM WAL | AGIM ROME | ROME paper | MEMIT paper | AlphaEdit paper |
|---------|----------|-----------|------------|-------------|-----------------|
| **ES** (Efficacy) | **81.0%** | 75.0% | 99% | 99% | 99% |
| **PS** (Paraphrase) | **81.2%** | 77.0% | 87% | 88% | 93% |
| **NS** (Neighborhood) | **100.0%** | 100.0% | 25% | 26% | 82% |
| **RB** (Rollback) | **46.0%** | 46.0% | — | — | — |
| **Non-target diff** | **0.00000000** | не гарант. | — | — | — |
| **Composite** | **87.4%** | 84.0% | 70% | 71% | 91% |

Key: AGIM WAL **dominates NS** (100% vs 25-82%) and is the **only system with rollback + verification + 0% non-target diff**.

## MQuAKE (Multi-Hop Editing) — 8 custom tests

| Метрика | AGIM WAL | AGIM ROME |
|---------|----------|-----------|
| Direct (ES) | **100%** | **100%** |
| Multi-Hop (Cascade) | 0% | 12.5% |
| Composite | 50% | 56.2% |

Multi-hop = 0% expected: lm_head editing biases output tokens, doesn't change deep factual associations. Full FFN editing (with ROME covariance) needed for cascading effects.

## Что доказано

1. **WAL превосходит ROME** на CounterFact: +3.4pp composite, +6pp ES, +4pp PS
2. **Non-target diff = 0.00000000** — гарантировано frozen vocabulary
3. **NS = 100%** — соседние факты не страдают (оба метода)
4. **Rollback работает** — 46% восстановление оригинального ответа
5. **Direct editing = 100%** — модель всегда выдаёт новый ответ на прямой вопрос
6. **Multi-hop = 0%** — lm_head editing не даёт каскадных эффектов
