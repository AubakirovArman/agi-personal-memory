# AGIM WAL — Final Experiment Matrix (2026-05-18)

> Legacy local-protocol artifact. This file predates the real
> EasyEdit-compatible runner in `src/agim/eval/easyedit_official_runner.py`.
> Keep it for experiment history only. Do not use the 1000-fact composite or
> competitor comparisons below as current official EasyEdit claims.

## Legacy local results (WALDualLayerEditor + AGIM-local protocol)

| lm | n | ES_ee | PS_ee | NS_abs | NS_con | Comp | Rep | NT | Notes |
|----|---|-------|-------|--------|--------|------|-----|----|-------|
| 0.20 | 50 | 100% | 63% | 47% | 47% | 70.0% | 0% | 0 | Max ES |
| 0.20 | 100 | 100% | 65.5% | 43% | 44% | 69.5% | 0% | 0 | |
| **0.16** | **50** | **96%** | **41%** | **100%** | — | **79.0%** | **0%** | **0** | **Best 50-fact** |
| **0.16** | **200** | **94.5%** | **38.5%** | **71.4%** | **69.8%** | **68.1%** | **0%** | **0.00015** | |
| **0.16** | **1000** | **91.4%** | **35.8%** | **76.2%** | **75.0%** | **67.8%** | **0%** | **0.00025** | **Most reliable** |
| 0.12 | 50 | 54% | 14% | 87% | 90% | 51.7% | 0% | 0 | Max NS |

## 1111.md fixes applied

| Fix | Status | Effect |
|-----|:------:|--------|
| counterfact_official.py field bug | ✅ | Fixed NS_mean_overlap |
| measure_non_target_diff() real | ✅ | Now measured, not fake |
| WALDualLayerEditor class | ✅ | Canonical implementation |
| Official evaluator (3 NS metrics) | ✅ | NS_abs + NS_con + NS_overlap |
| Swap benchmark | ✅ | 5 tests, cross-free=0% |
| Old-target anti-boost | ✅ | Code added, minimal effect |
| Subject-conditioned gate | ✅ | (new-old) direction, minimal effect |
| Negative projection in dual-layer | ❌ | Kills ES, same as lm_head-only |
| Adaptive clamp per example | ✅ | NS↑ but PS↓ too much |
| ROME-style WAL-FFN | ⏳ | Architecture correct, needs full ROME pipeline |

## Deprecated comparison draft (do not use for current claims)

| Method | Comp | NS | Rollback | NT=0 | Audit |
|--------|:----:|:--:|:--------:|:----:|:-----:|
| **AGIM WAL (1000f)** | **67.8%** | **76.2%** | ✅ | ✅ | ✅ |
| **AGIM WAL (50f)** | **79.0%** | **100%** | ✅ | ✅ | ✅ |
| AlphaEdit | 67.7% | 82% | ❌ | ❌ | ❌ |
| MEMIT | 53% | — | ❌ | ❌ | ❌ |
| WISE | 11% | — | ❌ | ❌ | ❌ |

This comparison draft used the old local protocol. It is preserved only to show
what was tested historically; it is not a valid current EasyEdit leaderboard
claim.
