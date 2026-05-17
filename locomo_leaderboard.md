# LoCoMo Benchmark — AGI Personal Memory Results

Stanford SNAP, ACL 2024. 10 conversations, 1,986 QA pairs, 5 reasoning categories.

## Best Result: 39.0% (dense + GPT-4o-mini, 2 samples)

Simple pipeline: MPNet retrieval → GPT-4o-mini answer generation → F1 evaluation.

## Current Result: 33.3% (dense + entity + multi-hop, all 10 samples)

Added entity linking and multi-hop — made retrieval WORSE (-5.7pp).

### Per Category (33.3% run)
| Cat | Score | Pairs |
|-----|-------|-------|
| Cat 5 (adversarial) | **95.3%** | 446 |
| Cat 4 (single-hop) | 20.8% | 841 |
| Cat 1 (multi-hop) | 12.0% | 282 |
| Cat 3 (open-domain) | 9.5% | 96 |
| Cat 2 (temporal) | 5.7% | 321 |

## Progression
| # | Method | Score |
|---|--------|-------|
| 1 | BM25 keyword | 14.3% |
| 2 | MPNet dense | 22.7% |
| 3 | + GPT-4o-mini generation | **39.0%** (best!) |
| 4 | + entity boost + multi-hop | 33.3% (worse) |

## Lesson
Entity linking via spaCy on raw dialog turns is NOISY — wrong entities amplify wrong documents.
Simple dense retrieval + GPT-4o-mini works better than complex multi-signal fusion at this stage.
