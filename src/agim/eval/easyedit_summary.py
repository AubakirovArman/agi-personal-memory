"""Summary aggregation for EasyEdit-compatible metrics."""
from __future__ import annotations

from typing import Any

import numpy as np


def mean_metric(rows: list[dict[str, Any]], phase: str, key: str) -> float | None:
    values = []
    for row in rows:
        if key in row[phase]:
            values.append(float(np.mean(row[phase][key])))
    return round(float(np.mean(values)), 6) if values else None


def summarize_official(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"pre": {}, "post": {}}
    for phase in ("pre", "post"):
        for key in ("rewrite_acc", "rephrase_acc", "rephrase_all_acc"):
            value = mean_metric(rows, phase, key)
            if value is not None:
                summary[phase][key] = value
    loc_values = []
    for row in rows:
        loc = row["post"].get("locality", {})
        if "neighborhood_acc" in loc:
            loc_values.append(float(np.mean(loc["neighborhood_acc"])))
    if loc_values:
        summary["post"]["locality"] = {
            "neighborhood_acc": round(float(np.mean(loc_values)), 6),
        }
    port_values = []
    for row in rows:
        for value in row["post"].get("portability", {}).values():
            if isinstance(value, list):
                port_values.append(float(np.mean(value)))
    if port_values:
        summary["post"]["portability"] = {
            "mean_acc": round(float(np.mean(port_values)), 6),
        }
    summary["post_generation_vanilla"] = _generation_summary(rows)
    contextual_rows = [row["contextual_generation"] for row in rows
                       if "contextual_generation" in row]
    if contextual_rows:
        summary["post_generation_contextual"] = _contextual_summary(contextual_rows)
    nt_rows = [row["NT"] for row in rows if "NT" in row]
    if nt_rows:
        summary["NT"] = _nt_summary(nt_rows)
    prob_rows = [row["probability"] for row in rows if "probability" in row]
    if prob_rows:
        summary["post_probability"] = _probability_summary(prob_rows)
    fluency_rows = [row["fluency"]["ngram_entropy"] for row in rows if "fluency" in row]
    if fluency_rows:
        summary["post_fluency"] = {
            "ngram_entropy": round(float(np.mean(fluency_rows)), 6)
        }
    relation_summary = summarize_by_relation(rows)
    if relation_summary:
        summary["metrics_by_relation_id"] = relation_summary
    return summary


def _generation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gen_summary = {
        "rewrite_acc": round(float(np.mean([
            np.mean(row["generation"]["rewrite_acc"]) for row in rows
        ])), 6)
    }
    for key in ("rephrase_acc", "rephrase_all_acc"):
        values = [float(np.mean(row["generation"][key])) for row in rows
                  if key in row["generation"]]
        if values:
            gen_summary[key] = round(float(np.mean(values)), 6)
    return gen_summary


def _contextual_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ctx_summary = {
        "rewrite_acc": round(float(np.mean([
            np.mean(row["rewrite_acc"]) for row in rows
        ])), 6)
    }
    for key in ("rephrase_acc", "rephrase_all_acc"):
        values = [float(np.mean(row[key])) for row in rows if key in row]
        if values:
            ctx_summary[key] = round(float(np.mean(values)), 6)
    return ctx_summary


def _nt_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "lm_head_non_edited_max": round(
            max(row["lm_head_non_edited_max"] for row in rows), 8),
        "embed_non_edited_max": round(
            max(row["embed_non_edited_max"] for row in rows), 8),
        "edited_lm_rows_avg": round(float(np.mean([
            row["edited_lm_rows_count"] for row in rows
        ])), 4),
        "edited_embed_rows_avg": round(float(np.mean([
            row["edited_embed_rows_count"] for row in rows
        ])), 4),
        "eos_row_changed_rate": round(float(np.mean([
            1.0 if row["eos_row_changed"] else 0.0 for row in rows
        ])), 6),
    }


def _probability_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prob_summary = {
        "rewrite_acc": round(float(np.mean([r["rewrite_acc"] for r in rows])), 6)
    }
    if any("rephrase_acc" in row for row in rows):
        prob_summary["rephrase_acc"] = round(float(np.mean([
            row.get("rephrase_acc", 0.0) for row in rows
        ])), 6)
    values = [float(np.mean(row["rephrase_all_acc"])) for row in rows
              if "rephrase_all_acc" in row]
    if values:
        prob_summary["rephrase_all_acc"] = round(float(np.mean(values)), 6)
    prob_loc = []
    for row in rows:
        for value in row.get("locality", {}).values():
            prob_loc.append(float(np.mean(value)))
    if prob_loc:
        prob_summary["locality_acc"] = round(float(np.mean(prob_loc)), 6)
    return prob_summary


def summarize_by_relation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        relation_id = row.get("relation_id")
        if relation_id is not None:
            grouped.setdefault(str(relation_id), []).append(row)

    summary: dict[str, Any] = {}
    for relation_id, rel_rows in sorted(grouped.items()):
        post: dict[str, Any] = {"n": len(rel_rows)}
        for key in ("rewrite_acc", "rephrase_acc", "rephrase_all_acc"):
            value = mean_metric(rel_rows, "post", key)
            if value is not None:
                post[key] = value
        loc_values = []
        for row in rel_rows:
            loc = row["post"].get("locality", {})
            if "neighborhood_acc" in loc:
                loc_values.append(float(np.mean(loc["neighborhood_acc"])))
        if loc_values:
            post["locality_acc"] = round(float(np.mean(loc_values)), 6)
        prob_values = [row.get("probability", {}) for row in rel_rows]
        prob_values = [row for row in prob_values if row]
        if prob_values:
            post["probability"] = _probability_summary(prob_values)
        summary[relation_id] = post
    return summary
