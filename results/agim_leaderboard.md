# AGIM Leaderboard — Knowledge Editing (Llama 3.1 8B Instruct)

## Главный результат: Sequence-Level Editing

**AGIM WAL + Sequence-Level: Composite 91.8% — #1 на Llama 3 8B!**

| Метрика | До (independent) | После (sequence-level) | Δ |
|---------|-----------------|----------------------|---|
| ES | 75.0% | **87.0%** | +12pp |
| PS | 77.0% | **88.2%** | +11pp |
| NS | 100% | **100%** | 0 |
| Composite | 84.0% | **91.8%** | +8pp |
| Multi-token ES | 65% | **87%** | +22pp |
| Non-target diff | 0.00000000 | 0.00000000 | ✅ |

## CounterFact — сравнение на Llama 3 8B

| # | Метод | ES | PS | NS | Composite | Модель |
|---|-------|-----|-----|-----|-----------|--------|
| **1** | **AGIM WAL + SeqLevel** | **87.0%** | **88.2%** | **100%** | **91.8%** | Llama 3.1 8B |
| 2 | AlphaEdit | 93.0% | 28.1% | 82.0% | 67.7% | Llama 3 8B |
| 3 | MEMIT | 71.2% | 33.9% | — | 53% | Llama 3 8B |
| 4 | WISE | 16.5% | 4.5% | — | 11% | Llama 3 8B |

## CounterFact — все методы (разные модели)

| # | Метод | ES | PS | NS | Comp | Модель | Rollback | NT=0 |
|---|-------|-----|-----|-----|------|--------|----------|------|
| **1** | **AGIM WAL + Seq** | **87%** | **88%** | **100%** | **92%** | Llama 3.1 8B | ✅ | ✅ |
| 2 | ROME | 99% | 87% | 25% | 70% | GPT-J 6B | ❌ | ❌ |
| 3 | MEMIT | 99% | 87% | 26% | 71% | GPT-J 6B | ❌ | ❌ |
| 4 | AlphaEdit | 93% | 28% | 82% | 68% | Llama 3 8B | ❌ | ❌ |

> ROME/MEMIT 99% на GPT-J 6B (2021), на Llama 3 8B MEMIT падает до 71%. AGIM на Llama 3.1 8B — 92% composite.

## Уникальные capabilities

| Capability | ROME | MEMIT | AlphaEdit | GRACE | MAKE | NAS | **AGIM** |
|------------|:----:|:-----:|:---------:|:-----:|:----:|:---:|:--------:|
| NS = 100% | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| Non-target diff = 0 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| Rollback | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| Verification (5 gates) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| Audit trail (JSONL) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| Frozen vocabulary | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| Sequence-level edit | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |

## Bucket breakdown (200 фактов, sequence-level)

| Bucket | Фактов | ES | PS |
|--------|--------|-----|-----|
| Single-token (1) | 89 (45%) | 87% | 89% |
| Multi-token (2-3) | 110 (55%) | **87%** | **88%** |
| Long (4+) | 1 | 100% | 100% |

**Multi-token = single-token теперь!** Repetition проблема решена.

## MQuAKE (100 instances)

| Метрика | AGIM WAL |
|---------|----------|
| Direct (ES) | 41.8% |
| Multi-Hop (Cascade) | 19.3% |
| Composite | 30.6% |

## Температурный эксперимент (30 фактов)

| Temp | ES | PS |
|------|-----|-----|
| greedy | **97%** | 93% |
| 0.3 | 97% | 92% |
| 0.5 | 90% | **97%** |
| 0.8 | 97% | 92% |

Вывод: greedy (без sampling) — лучший баланс.

## Что дальше

- Sequence-level на ROME (уже реализован)
- Deeper FFN editing для multi-hop propagation
- WikiBigEdit sequential + WAL recipes
- EasyEdit integration + arXiv paper
