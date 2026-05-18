"""Backend matrix runner helpers for EasyEdit-compatible runs."""
from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any, Callable


def run_backend_comparison(
    *,
    args,
    model,
    tok,
    hparams,
    facts,
    records,
    dataset_sha256,
    all_facts,
    locality_limit,
    compute_edit_quality,
    test_prediction_acc,
    summary_metrics,
    device_id,
    run_backend_once: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for backend in parse_backend_list(args.compare_backends):
        reason = backend_skip_reason(args, backend)
        if reason:
            rows.append({"edit_backend": backend, "status": "skipped", "reason": reason})
            continue
        backend_args = copy(args)
        backend_args.edit_backend = backend
        backend_args.method_profile_id = None
        backend_args.output = str(backend_output_path(Path(args.output), backend))
        backend_args.failures_output = None
        payload = run_backend_once(
            args=backend_args,
            model=model,
            tok=tok,
            hparams=hparams,
            facts=facts,
            records=records,
            dataset_sha256=dataset_sha256,
            all_facts=all_facts,
            locality_limit=locality_limit,
            compute_edit_quality=compute_edit_quality,
            test_prediction_acc=test_prediction_acc,
            summary_metrics=summary_metrics,
            device_id=device_id,
        )
        rows.append({
            "edit_backend": backend,
            "status": "completed",
            "output": backend_args.output,
            "method_profile_id": payload["method_profile_id"],
            "summary": payload["summary"],
            "time_s": payload["time_s"],
        })
    return {
        "artifact_schema_version": "easyedit_backend_matrix.v1",
        "n": len(records),
        "model": args.model,
        "device": args.device,
        "sample_policy": args.sample_policy,
        "seed": args.seed,
        "sequential_edit": args.sequential_edit,
        "requested_backends": parse_backend_list(args.compare_backends),
        "rows": rows,
    }


def parse_backend_list(value: str) -> list[str]:
    valid = {"dual_row", "side_slot", "wal_rome", "wal_memit"}
    backends = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(backends) - valid)
    if unknown:
        raise ValueError(f"Unknown compare backends: {', '.join(unknown)}")
    return backends


def backend_skip_reason(args, backend: str) -> str | None:
    if backend == "side_slot" and not args.sequential_edit:
        return "side_slot comparison requires --sequential-edit"
    return None


def backend_output_path(output: Path, backend: str) -> Path:
    suffix = output.suffix or ".json"
    return output.with_name(f"{output.stem}.{backend}{suffix}")
