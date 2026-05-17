# AGIM Leaderboard — Knowledge Editing (Llama 3.1 8B)

## CounterFact (главный бенчмарк KE)

| # | Метод | ES | PS | NS | Composite | Rollback | 0% NT diff |
|---|-------|-----|-----|-----|-----------|----------|------------|
| 1 | **AlphaEdit** | 99.8% | 93.4% | 82.0% | **91.7%** | ❌ | ❌ |
| 2 | **GRACE** | 99.1% | — | 88.4% | — | ❌ | ❌ |
| 3 | **AGIM WAL** | 79.5% | 79.2% | **100%** | **86.2%** | ✅ | ✅ |
| 4 | **AGIM ROME** | 75.0% | 77.0% | **100%** | 84.0% | ✅ | ❌ |
| 5 | **MEMIT** | 99.2% | 87.0% | 25.7% | 70.7% | ❌ | ❌ |
| 6 | **ROME** | 99.0% | 86.9% | 25.2% | 70.4% | ❌ | ❌ |
| 7 | **WISE** | — | — | — | ~70% | ❌ | ❌ |

**AGIM место:** #3 по Composite. **#1 по NS (100%)**. Единственный с rollback + 0% non-target diff.

---

## MQuAKE (multi-hop, GPT-2 XL)

| # | Метод | Direct | Multi-Hop | Composite |
|---|-------|--------|-----------|-----------|
| 1 | **MAKE** | 63.6% | ~60% | **~62%** |
| 2 | **ROME** | 56.1% | ~50% | ~53% |
| 3 | **MEMIT** | 55.5% | ~50% | ~53% |
| 4 | **AGIM WAL** | 41.8% | 19.3% | 30.6% |

**AGIM место:** #4. Сильно отстаём — lm_head editing не даёт multi-hop propagation.

> Примечание: MQuAKE paper использует GPT-2 XL. AGIM на Llama 3.1 8B. Прямое сравнение некорректно (разные модели).

---

## ZsRE (KnowEdit)

| # | Метод | ES | Gen | Spec | Composite |
|---|-------|-----|-----|------|-----------|
| 1 | **NAS** | — | 97.7% | — | **~97%** |
| 2 | **GRACE** | — | 93.8% | — | **~94%** |
| 3 | **ROME** | — | ~50% | — | ~50% |
| 4 | **AGIM WAL** | 36.5% | 37.5% | 13.0% | 29.0% |

**AGIM место:** #4. Question-формат промптов + multi-token repetition.

---

## WikiBio (KnowEdit) — Hallucination Correction

| # | Метод | ARR |
|---|-------|-----|
| 1 | **AGIM WAL** (custom 10) | **100%** |
| 2 | **GRACE** | >90% |
| 3 | **ROME** | ~80% |
| 4 | **AGIM WAL** (KnowEdit 100) | 0% |

**AGIM место:** #1 на custom тестах (простые факты). 0% на KnowEdit WikiBio (формат не поддерживается — требуется генерация параграфов, а не fact editing).

---

## Сводный рейтинг (уникальные capabilities)

| Capability | ROME | MEMIT | AlphaEdit | GRACE | MAKE | NAS | **AGIM** |
|------------|------|-------|-----------|-------|------|-----|----------|
| Редактирование весов | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| NS = 100% | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| Non-target diff = 0 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| Rollback | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| Verification (5 gates) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| Audit trail (JSONL) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| Frozen vocabulary | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| Multi-hop propagation | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |

---

## Итог

AGIM — **не #1 по accuracy**. Но **единственный с 6 уникальными capabilities** (NS=100%, NT=0, rollback, verification, audit, frozen vocab).

Позиционирование: не "мы точнее всех", а **"мы единственные, кому можно доверить безопасное редактирование знаний"**.

### Что нужно чтобы догнать по accuracy:
1. **Multi-token repetition fix** → поднимет ES/PS на 40-60pp
2. **Deeper FFN editing** → multi-hop propagation
3. **Covariance constraint** (как в ROME) → specificity + generalization
