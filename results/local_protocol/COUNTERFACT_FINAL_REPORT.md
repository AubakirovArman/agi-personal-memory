# AGIM WAL — CounterFact Final Report (2026-05-17)

> Legacy local-protocol report. Current EasyEdit-compatible results live in
> `../easyedit_official/`. The numbers below are historical diagnostics and
> should not be presented as current official EasyEdit results.

## Models & Protocol

- **Model**: meta-llama/Llama-3.1-8B-Instruct (8B, same class as Llama 3 8B)
- **Editor**: WAL + sequence-level, lm_head editing, clamp=0.3
- **Generation**: greedy (do_sample=False)
- **Dataset**: CounterFact from rome.baulab.info

## Что мы прошли

1. **CounterFact 1000 facts** — WAL composite 86.2% (substring proto)
2. **CounterFact 200 facts** — Sequence-level composite 91.8% (substring proto)
3. **MQuAKE 100 instances** — Direct=41.8%, Multi-Hop=19.3%
4. **KnowEdit/ZsRE 200** — ES=36.5%, Gen=37.5%
5. **KnowEdit/wiki_cf 200** — ES=15.5%, Gen=14%
6. **WikiBio custom 10** — ARR=100%, Verified=90%

## Hardened Метрики (50 фактов, все фиксы)

| Протокол | ES | ES_clean | PS | NS | NT | Composite |
|----------|-----|----------|-----|-----|-----|-----------|
| AGIM (substring) | 94% | 16% | 95% | 31.5% | 0.00000000 | 73.5% |
| EasyEdit (token exact) | 70% | — | 64% | 31.5% | 0.00000000 | 55.2% |

## Ключевые Findings

### Сильные стороны
- **NT = 0.00000000** (measured, verified) — только AGIM
- **ES = 94%** на substring-протоколе
- **Rollback** работает (38% exact match recovery)
- **Frozen vocabulary** гарантирует 0% weight-level non-target diff

### Слабые стороны
- **NS = 31.5%** — lm_head editing меняет ответы на семантически близкие вопросы
- **Repetition rate = 78%** — главная проблема, даже с sequence-level
- **ES_clean = 16%** — только 16% генераций без повторов target токена

## Что дальше

1. **EOS boost** — добавить в sequence-level editing (остановка после target)
2. **TargetCompleteStoppingCriteria** — останавливать генерацию при появлении target
3. **Deeper FFN editing** — перейти от lm_head к FFN слоям для лучшего NS
4. **Temperature tuning** — проверить T=0.3 для снижения repetition

## Comparison (Llama-3-class 8B, EasyEdit protocol)

| Метод | Composite | NS | NT=0 | Rollback |
|-------|-----------|-----|------|----------|
| AlphaEdit | 67.7% | ~82% | ❌ | ❌ |
| **AGIM WAL** | **55.2%** | **31.5%** | **✅** | **✅** |
| MEMIT | 53% | — | ❌ | ❌ |
| WISE | 11% | — | ❌ | ❌ |

AGIM competitive с MEMIT, отстаёт от AlphaEdit из-за низкого NS и repetition.
Уникальные преимущества: NT=0, rollback, verification, audit trail.
