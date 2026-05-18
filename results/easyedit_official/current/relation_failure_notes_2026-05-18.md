# Relation Failure Notes - 2026-05-18

Source artifacts:

- `current/easyedit_official_50_first42_psall_baseline.json`
- `ablations/easyedit_official_50_first42_psall_positive_prompts.json`
- `sequential/easyedit_official_50_first42_psall_seq_lm015_negx05_noeosanti_retention.json`
- `sequential/easyedit_official_50_first42_psall_seq_lm015_negx05_noeosanti_positive_prompts_retention.json`

The tables below show the weakest relation ids by exact-token locality in the
fresh n=50 artifacts. Each relation has small `n`, so this is a triage signal,
not a stable benchmark ranking.

## Single Baseline: Weakest Locality

| Relation | n | TF rewrite | TF PS@All | TF locality | Sample prompt |
| --- | ---: | ---: | ---: | ---: | --- |
| P27 | 1 | 100.0% | 100.0% | 0.0% | Mahmoud Fawzi has a citizenship from |
| P1303 | 1 | 100.0% | 50.0% | 40.0% | Toko Yasuda, the |
| P740 | 1 | 100.0% | 100.0% | 40.0% | Anaal Nathrakh, that was created in |
| P136 | 2 | 100.0% | 100.0% | 45.0% | What does Heath Brothers play? They play |
| P159 | 2 | 100.0% | 100.0% | 45.0% | The headquarter of Monell Chemical Senses Center is located in |
| P937 | 2 | 100.0% | 100.0% | 45.0% | Leonardo Balada found employment in |

## Sequential Tuned: Weakest Locality

| Relation | n | TF rewrite | TF PS@All | TF locality | Sample prompt |
| --- | ---: | ---: | ---: | ---: | --- |
| P364 | 1 | 0.0% | 0.0% | 0.0% | The original language of The Icelandic Dream was |
| P190 | 4 | 75.0% | 0.0% | 0.0% | What is the twin city of Lyon? It is |
| P27 | 1 | 100.0% | 0.0% | 0.0% | Mahmoud Fawzi has a citizenship from |
| P136 | 2 | 50.0% | 0.0% | 10.0% | What does Heath Brothers play? They play |
| P39 | 1 | 100.0% | 0.0% | 10.0% | Robert William Muench is a |
| P1412 | 3 | 66.7% | 16.7% | 10.0% | The language used by Gilad Atzmon is |

## Readout

The single-edit baseline mostly rewrites successfully even when locality is
weak, which points to over-broad target-row effects rather than missing edit
strength.

The sequential tuned run shows relation-level collapse where PS@All goes to
zero while rewrite can remain nonzero. This is consistent with accumulated
interference across prompts and target rows. Method work should therefore focus
on locality-preserving constraints and per-edit isolation before scaling to
larger n.
