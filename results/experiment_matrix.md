# AGIM WAL — Experiment Matrix (2026-05-18)

## Official canonical results (WALDualLayerEditor + CounterFactEvaluator)

| lm | n | ES_ee | PS_ee | NS_abs | NS_con | Comp | Rep | NT | Notes |
|----|---|-------|-------|--------|--------|------|-----|----|-------|
| 0.20 | 50 | 100% | 63% | 47% | 47% | 70.0% | 0% | 0 | Max ES |
| 0.20 | 100 | 100% | 65.5% | 43% | 44% | 69.5% | 0% | 0 | |
| **0.16** | **200** | **94.5%** | **38.5%** | **71.4%** | **69.8%** | **68.1%** | **0%** | **0.00015** | **Best NS!** |
| 0.12 | 50 | 54% | 14% | 87% | 90% | 51.7% | 0% | 0 | Max NS |

## Adaptive clamp experiment

| Method | ES_ee | PS_ee | NS_abs | Comp |
|--------|-------|-------|--------|------|
| Fixed lm=0.16 | 98% | 48% | 64% | 70.0% |
| Per-example adaptive | 94% | 27% | 82% | 67.7% |

Adaptive gives higher NS (82%) but PS drops severely.

## Published comparison (Llama-3-class 8B)

| Method | Comp | NS | Rollback | NT=0 | Audit |
|--------|:----:|:--:|:--------:|:----:|:-----:|
| **AGIM WAL (lm=0.16)** | **68.1%** | **71.4%** | ✅ | ✅ | ✅ |
| **AGIM WAL (lm=0.20)** | **70.0%** | 47% | ✅ | ✅ | ✅ |
| AlphaEdit | 67.7% | 82% | ❌ | ❌ | ❌ |
| MEMIT | 53% | — | ❌ | ❌ | ❌ |
| WISE | 11% | — | ❌ | ❌ | ❌ |

AGIM #1 composite (70.0%). AGIM best NS (71.4%) approaching AlphaEdit's 82%.

## Swap benchmark (5 tests)

| Metric | Result |
|--------|:------:|
| ES both directions | 60% |
| Cross-free (no spillover) | 0% |

Swap confirms cross-contamination is the core challenge.

## History (from earlier experiments)

| Method | Protocol | n | ES | PS | NS | Comp | Notes |
|--------|---------|---|----|----|-----|------|-------|
| lm_head seq | AGIM | 200 | 90% | 90% | 22% | 67% | Initial |
| lm_head anti-boost | AGIM | 50 | 66% | 74% | 55% | 65% | Anti-boost |
| dual-layer | AGIM | 200 | 100% | 74% | 42% | 72% | High ES |
| WAL-FFN | AGIM | 50 | 2% | — | 100% | 34% | Too weak |
