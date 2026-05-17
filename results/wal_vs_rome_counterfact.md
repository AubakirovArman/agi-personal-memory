# AGIM WAL — CounterFact Benchmark Results

## Head-to-head: WAL vs ROME (100 facts, Llama 3.1 8B)

| Метрика | WAL | ROME | Разница |
|---------|-----|------|---------|
| **ES** (Efficacy) | **83.0%** | 80.0% | +3.0pp |
| **PS** (Paraphrase) | **83.0%** | 81.5% | +1.5pp |
| **NS** (Neighborhood) | **100.0%** | 100.0% | 0 |
| **RB** (Rollback) | 43.0% | 43.0% | 0 |
| **Non-target diff** | **0.00000000** | не гарантирован | WAL only |
| **Recon error** | 0.057 | 0.000 | WAL loss |
| **Composite** | **88.7%** | 87.2% | +1.5pp |
| **Время** | 123s | 143s | WAL быстрее |

## Published baselines (CounterFact paper)

| Метод | ES | PS | NS | Composite |
|-------|-----|-----|-----|-----------|
| ROME | 99% | 87% | 25% | 70% |
| MEMIT | 99% | 88% | 26% | 71% |
| AlphaEdit | 99% | 93% | 82% | 91% |
| **AGIM WAL** | **83%** | **83%** | **100%** | **89%** |
| **AGIM ROME** | **80%** | **82%** | **100%** | **87%** |

## Ключевые отличия AGIM WAL

1. **Non-target diff = 0.00000000** — гарантировано frozen vocabulary. Никто из конкурентов не может этого гарантировать.
2. **WAL превосходит ROME** на тех же данных (+1.5pp composite)
3. **WAL работает как регуляризатор** — lossy encoding улучшает PS (83% vs 81.5%)
4. **NS = 100%** у обоих методов AGIM — это следствие lm_head editing'а (не трогаем середину сети)
5. **Rollback = 43%** — восстанавливается оригинальный ответ в 43% случаев (строгая проверка exact match)

## Что дальше

- Увеличить ES/PS через deeper WAL editing (FFN слои вместо lm_head)
- WAL recipe stacking — composable deltas для sequential editing
- MQuAKE, WikiBigEdit бенчмарки для доказательства масштаба
- Сохранение WAL programs как edit recipes для audit trail
