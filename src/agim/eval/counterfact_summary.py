"""Summary aggregation for the local CounterFact evaluator."""
from __future__ import annotations

from typing import Any


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_protocol(results: list[dict[str, Any]], protocol: str) -> dict[str, float]:
    rows = [r["protocols"][protocol] for r in results if protocol in r["protocols"]]
    es_token = _mean([r["ES_token"] for r in rows])
    es_sub = _mean([r["ES_substring"] for r in rows])
    ps_token_2 = _mean([r["PS_token_2"] for r in rows])
    ps_sub_2 = _mean([r["PS_substring_2"] for r in rows])
    ns_abs = _mean([r["NS_absence"] for r in rows])
    ns_con = _mean([r["NS_consistency"] for r in rows])
    summary = {
        "ES_token": es_token,
        "ES_substring": es_sub,
        "ES_clean": _mean([r["ES_clean"] for r in rows]),
        "PS_token_2": ps_token_2,
        "PS_substring_2": ps_sub_2,
        "PS_token_all": _mean([r["PS_token_all"] for r in rows]),
        "PS_substring_all": _mean([r["PS_substring_all"] for r in rows]),
        "NS_absence": ns_abs,
        "NS_consistency": ns_con,
        "NS_overlap": _mean([r["NS_overlap"] for r in rows]),
        "Composite_token_absence": (es_token + ps_token_2 + ns_abs) / 3,
        "Composite_token_consistency": (es_token + ps_token_2 + ns_con) / 3,
        "Composite_substring_absence": (es_sub + ps_sub_2 + ns_abs) / 3,
        "Composite_substring_consistency": (es_sub + ps_sub_2 + ns_con) / 3,
        "repetition_rate": _mean([1.0 if r["has_repetition"] else 0.0 for r in rows]),
        "RB_old_target": _mean([r["RB_old_target"] for r in rows]),
        "RB_consistency": _mean([r["RB_consistency"] for r in rows]),
        "RB_overlap": _mean([r["RB_overlap"] for r in rows]),
    }
    return {k: round(v, 6) for k, v in summary.items()}


def summarize_nt(results: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "lm_head_non_edited_max": round(
            max((r["NT"]["lm_head_non_edited_max"] for r in results), default=0.0), 8),
        "embed_non_edited_max": round(
            max((r["NT"]["embed_non_edited_max"] for r in results), default=0.0), 8),
        "edited_lm_rows_avg": round(
            _mean([r["NT"]["edited_lm_rows_count"] for r in results]), 4),
        "edited_embed_rows_avg": round(
            _mean([r["NT"]["edited_embed_rows_count"] for r in results]), 4),
        "eos_row_changed_rate": round(
            _mean([1.0 if r["NT"]["eos_row_changed"] else 0.0 for r in results]), 6),
    }


def group_summary(results: list[dict[str, Any]], key: str,
                  protocol: str = "easyedit_strict") -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        if key == "target_token_bucket":
            group = _token_bucket(row["target_token_count"])
        elif key == "subject_token_bucket":
            group = _token_bucket(row["subject_token_count"])
        else:
            group = str(row.get(key, "unknown"))
        grouped.setdefault(group, []).append(row)
    return {
        group: {"n": len(rows), **summarize_protocol(rows, protocol)}
        for group, rows in sorted(grouped.items())
    }


def _token_bucket(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 3:
        return "2-3"
    return "4+"
