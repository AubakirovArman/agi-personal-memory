# AGIM WAL — Experiment Matrix (2026-05-18)

| Method | Protocol | n | ES | PS | NS_abs | NS_con | ES_clean | Rep | NT | Notes |
|--------|---------|---:|----|----|--------|--------|---------|-----|----|-------|
| lm_head seq | AGIM | 200 | 90% | 90% | — | 22% | 32% | 75% | 0 | Strong ES, bad locality |
| lm_head anti-boost | AGIM | 50 | 66% | 74% | 55% | — | 66% | 0% | 0 | Anti-boost helps |
| **dual-layer** | **AGIM** | **50** | **100%** | **74%** | **47%** | **47%** | **100%** | **0%** | **0** | **Canonical result** |
| dual-layer | **EasyEdit** | **50** | **100%** | **63%** | **47%** | **47%** | **100%** | **0%** | **0** | **Canonical result** |
| dual-layer | AGIM | 200 | 100% | 74% | 42% | — | 100% | 0% | 0 | 200-fact confirm (test_dual) |
| WAL-FFN | AGIM | 50 | 2% | — | 100% | — | — | — | 0 | Too weak |
| dual-layer | Swap | 50 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | Next test |

## Published comparison (Llama-3-class 8B)

| Method | Composite | NS | Rollback | NT=0 |
|--------|:--------:|:--:|:--------:|:----:|
| **AGIM WAL dual-layer** | **70.0%** | **47%** | **✅** | **✅** |
| AlphaEdit | 67.7% | 82% | ❌ | ❌ |
| MEMIT | 53% | — | ❌ | ❌ |
| WISE | 11% | — | ❌ | ❌ |

AGIM #1 composite, #1 unique features. AlphaEdit has better NS (82% vs 47%) — next target.
