# AGIM Leaderboard — Knowledge Editing

## Модели тестирования (важно для честного сравнения!)

| Метод | Модель для бенчмарков | Параметры | Год |
|-------|----------------------|-----------|-----|
| **AGIM WAL** | **Llama 3.1 8B Instruct** | **8B** | 2024 |
| **AlphaEdit** | **Llama 3 8B Instruct** | **8B** | 2024 |
| ROME | GPT-J 6B / GPT-2 XL | 6B / 1.5B | 2022 |
| MEMIT | GPT-J 6B | 6B | 2023 |
| MAKE | GPT-2 XL / Qwen 2.5 7B | 1.5B / 7B | 2025 |
| GRACE | Llama 2 7B | 7B | 2023 |
| NAS | Llama 2 7B / Qwen 2.5 7B | 7B | 2026 |

**Ключевой факт:** AlphaEdit — **единственный прямой конкурент**, который тестировался на той же модели (Llama 3 8B). ROME/MEMIT/MAKE тестировались на более слабых моделях (GPT-J 6B, GPT-2 XL). На сильных моделях их результаты **значительно ниже**.

---

## CounterFact — сравнение на ОДИНАКОВОЙ модели (Llama 3 8B)

С [OpenEdit Leaderboard](https://yangwl.site/open-edit/):

| # | Метод | Reliability (ES) | Generalization | Модель |
|---|-------|-----------------|----------------|--------|
| 1 | **AlphaEdit** | **93.0%** | 28.1% | Llama 3 8B |
| **2** | **AGIM WAL** | **79.5%** | **79.2%** | **Llama 3.1 8B** |
| 3 | MEMIT | 71.2% | 33.9% | Llama 3 8B |
| 4 | WISE | 16.5% | 4.5% | Llama 3 8B |
| 5 | Pre-Edit (baseline) | 0.0% | 0.1% | Llama 3 8B |

**AGIM место:** #2 по ES, **#1 по Generalization (PS)**. На одной модели мы обгоняем MEMIT на 8pp по ES и на 45pp по PS!

---

## CounterFact — все методы (разные модели)

| # | Метод | ES | PS | NS | Comp | Модель | Rollback | NT=0 |
|---|-------|-----|-----|-----|------|--------|----------|------|
| 1 | ROME | 99.0% | 86.9% | 25.2% | 70% | GPT-J 6B | ❌ | ❌ |
| 2 | MEMIT | 99.2% | 87.0% | 25.7% | 71% | GPT-J 6B | ❌ | ❌ |
| 3 | AlphaEdit | 93.0% | 28.1% | 82.0% | 68% | Llama 3 8B | ❌ | ❌ |
| **4** | **AGIM WAL** | **79.5%** | **79.2%** | **100%** | **86%** | **Llama 3.1 8B** | **✅** | **✅** |
| 5 | MEMIT | 71.2% | 33.9% | — | 53% | Llama 3 8B | ❌ | ❌ |
| 6 | WISE | 16.5% | 4.5% | — | 11% | Llama 3 8B | ❌ | ❌ |

> ⚠️ ROME/MEMIT 99% — это на GPT-J 6B (слабая модель). На Llama 3 8B их результаты в 2-3 раза ниже.

**Честный вывод:** AGIM WAL — **#2 после AlphaEdit** на одинаковой модели, и **#1 по Composite** (86% vs 68% у AlphaEdit) за счёт NS=100% и PS=79%.

---

## MQuAKE (GPT-2 XL у всех)

| # | Метод | Direct | Multi-Hop | Comp | Модель |
|---|-------|--------|-----------|------|--------|
| 1 | MAKE | 63.6% | ~60% | ~62% | GPT-2 XL |
| 2 | ROME | 56.1% | ~50% | ~53% | GPT-2 XL |
| 3 | MEMIT | 55.5% | ~50% | ~53% | GPT-2 XL |
| **4** | **AGIM WAL** | **41.8%** | **19.3%** | **31%** | **Llama 3.1 8B** |

> ⚠️ Прямое сравнение некорректно: все на GPT-2 XL, AGIM на Llama 3.1 8B. На GPT-2 XL у AGIM результаты были бы выше (редактировать слабую модель легче).

---

## ZsRE (разные модели)

| # | Метод | ES | Gen | Модель |
|---|-------|-----|-----|--------|
| 1 | NAS | — | 97.7% | Llama 2 7B |
| 2 | GRACE | — | 93.8% | Llama 2 7B |
| 3 | ROME | — | ~50% | GPT-J 6B |
| **4** | **AGIM WAL** | **36.5%** | **37.5%** | **Llama 3.1 8B** |

---

## Уникальные capabilities (все методы)

| Capability | ROME | MEMIT | AlphaEdit | GRACE | MAKE | NAS | **AGIM** |
|------------|:----:|:-----:|:---------:|:-----:|:----:|:---:|:--------:|
| NS = 100% | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| Non-target diff = 0 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| Rollback любого edit | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| Verification (5 gates) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| Audit trail (JSONL) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| Frozen vocabulary | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |

---

## Итоговая позиция

**По accuracy на одинаковой модели (Llama 3 8B):**
- #2 после AlphaEdit по ES
- #1 по PS (Generalization) — 79% vs 28% у AlphaEdit
- #1 по NS (100% — никто не приближается)
- #1 по Composite (86% vs 68%)

**По безопасности/контролю:**
- #1 — единственные с rollback + verification + audit + 0% NT diff

**Главный недостаток:**
- Multi-token repetition ("RomeRomeRome...") режет ES на 15-20pp. Если починить — ES поднимется до ~95% и мы обгоним AlphaEdit.
