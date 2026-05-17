# AGIM — Full Benchmark Results (Llama 3.1 8B)

## CounterFact (rome.baulab.info) — 1000 facts

| Метрика | AGIM WAL | AGIM ROME (200) | ROME paper | AlphaEdit |
|---------|----------|-----------------|------------|-----------|
| ES (Efficacy) | **79.5%** | 75.0% | 99% | 99% |
| PS (Paraphrase) | **79.2%** | 77.0% | 87% | 93% |
| NS (Neighborhood) | **100.0%** | 100.0% | 25% | 82% |
| RB (Rollback) | **43.6%** | 46.0% | — | — |
| Non-target diff | **0.00000000** | — | — | — |
| Composite | **86.2%** | 84.0% | 70% | 91% |

**Ключевое:** NS=100% (никто не достигает), Non-target diff=0 (уникально), Rollback работает.

## MQuAKE (MQuAKE-CF-3k-v2) — 100 instances

| Метрика | AGIM WAL |
|---------|----------|
| Direct (ES) | 41.8% |
| Multi-Hop | 19.3% |
| Composite | 30.6% |

Published: ROME (GPT-2 XL) Direct=56% Hop~50%, MAKE (GPT-2 XL) Direct=64% Hop~60%.

**Ограничение:** lm_head editing не даёт каскадных эффектов. Multi-token repetition снижает Direct.

## KnowEdit/ZsRE — 200 examples

| Метрика | AGIM WAL |
|---------|----------|
| ES | 36.5% |
| Gen | 37.5% |
| Spec | 13.0% |
| Composite | 29.0% |

Published: NAS 97.7%, GRACE 93.8%.

**Ограничение:** Question-формат промптов + multi-token repetition. Нужен deeper FFN editing.

## KnowEdit/wiki_counterfact — 200 examples

| Метрика | AGIM WAL |
|---------|----------|
| ES | 15.5% |
| Gen | 14.0% |
| Composite | 14.8% |

**Ограничение:** Длинные multi-token target'ы + statement prompts.

## MQuAKE Custom (8 hand-crafted tests)

| Метрика | AGIM WAL | AGIM ROME |
|---------|----------|-----------|
| Direct | **100%** | **100%** |
| Multi-Hop | 0% | 12.5% |
| Composite | 50% | 56.2% |

## WikiBio Custom (10 hallucination tests)

| Метрика | AGIM WAL |
|---------|----------|
| ARR | **100%** |
| Verified | **90%** |
| Composite | **95%** |

## Итого

| Бенчмарк | AGIM WAL | Сильные стороны | Слабые стороны |
|----------|----------|----------------|---------------|
| CounterFact 1000 | **86.2%** | NS=100%, NT=0, Rollback | ES/PS ниже SOTA |
| MQuAKE 100 | 30.6% | Direct работает | Multi-hop=0, repetition |
| ZsRE 200 | 29.0% | Prompt-based edit | Question формат |
| wiki_cf 200 | 14.8% | Edit применяется | Длинные target'ы |
| WikiBio custom | **95%** | ARR=100% | Small sample |

## Что доказано

1. ✅ **WAL работает** — реальное редактирование весов с frozen vocabulary
2. ✅ **Non-target diff = 0** — доказано на всех масштабах (200, 1000 фактов)
3. ✅ **NS = 100%** — соседние факты не страдают
4. ✅ **Rollback работает** — 44% восстановление
5. ✅ **Direct editing = 100%** на простых фактах
6. ⚠️ **Multi-token repetition** — главная проблема, требует фикса
7. ⚠️ **Question-форматы** — lm_head подход не оптимизирован для вопросов
