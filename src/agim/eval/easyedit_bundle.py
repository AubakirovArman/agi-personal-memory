"""Build per-case post-edit metric bundles."""
from __future__ import annotations

from typing import Any

from .easyedit_metrics import (
    attach_locality_acc,
    attach_rephrase_all_acc,
    contextual_generation_metrics,
    fluency_metrics,
    official_generation_metrics,
    probability_metrics,
)


def post_edit_bundle(
    *,
    model,
    tok,
    hparams,
    compute_edit_quality,
    test_prediction_acc,
    record: dict[str, Any],
    pre: dict[str, Any],
    model_name: str,
    device_id: int,
    device: str,
    probability: bool,
    fluency: bool,
) -> dict[str, Any]:
    post = compute_edit_quality(
        model, model_name, hparams, tok, record, device_id, eval_metric="token_em"
    )
    attach_locality_acc(pre, post, record)
    attach_rephrase_all_acc(model, tok, post, record, device)
    row: dict[str, Any] = {
        "pre": pre,
        "post": post,
        "generation": official_generation_metrics(
            model, tok, hparams, test_prediction_acc, record, device_id
        ),
        "contextual_generation": contextual_generation_metrics(
            model, tok, record, device
        ),
    }
    if probability:
        row["probability"] = probability_metrics(model, tok, record, device)
    if fluency:
        row["fluency"] = fluency_metrics(model, tok, [record["prompt"]], device)
    return row
