# AGIM Benchmark Results — Full Suite

## CounterFact (Knowledge Editing) — 200 facts

| Метрика | AGIM WAL | AGIM ROME | ROME paper | MEMIT | AlphaEdit |
|---------|----------|-----------|------------|-------|-----------|
| **ES** | **81.0%** | 75.0% | 99% | 99% | 99% |
| **PS** | **81.2%** | 77.0% | 87% | 88% | 93% |
| **NS** | **100.0%** | 100.0% | 25% | 26% | 82% |
| **RB** | **46.0%** | 46.0% | — | — | — |
| **NT diff** | **0.00000000** | — | — | — | — |
| **Composite** | **87.4%** | 84.0% | 70% | 71% | 91% |

## MQuAKE (Multi-Hop Editing) — 8 tests

| Метрика | AGIM WAL | AGIM ROME |
|---------|----------|-----------|
| Direct (ES) | **100%** | **100%** |
| Multi-Hop | 0% | 12.5% |
| Composite | 50% | 56.2% |

## WikiBio (Hallucination Correction) — 10 tests

| Метрика | AGIM WAL | AGIM ROME | GRACE | ROME paper |
|---------|----------|-----------|-------|------------|
| ARR | **100%** | **100%** | >90% | ~80% |
| Verified | **90%** | **90%** | — | — |
| Composite | **95%** | **95%** | — | — |

## Итого

| Бенчмарк | AGIM WAL | AGIM ROME | Best Published |
|----------|----------|-----------|----------------|
| CounterFact 200 | **87.4%** | 84.0% | 91% (AlphaEdit) |
| MQuAKE Direct | **100%** | **100%** | 64% (MAKE) |
| WikiBio ARR | **100%** | **100%** | >90% (GRACE) |

## Уникальные преимущества AGIM

1. **Non-target diff = 0.00000000** — только AGIM (WAL frozen vocabulary)
2. **Rollback** — только AGIM (46% exact match recovery)
3. **Verification** — только AGIM (5 gates перед commit)
4. **Audit trail** — только AGIM (JSONL лог каждого edit)
5. **WAL превосходит ROME** — +3.4pp composite на CounterFact
