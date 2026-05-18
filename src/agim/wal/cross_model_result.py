from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .cross_model_artifact import run_wal_artifact_workflow
from .cross_model_common import ModelCandidate, model_workflow_result
from .cross_model_runtime import run_local_causal_lm_smoke


def controlled_model_workflow_result(
    module: str,
    name: str,
    family: str,
    candidates: list[ModelCandidate],
    near_misses: list[ModelCandidate],
) -> dict[str, object]:
    result = model_workflow_result(module, name, family, candidates, near_misses)
    if not candidates:
        return result

    selected = candidates[0]
    runtime = run_local_causal_lm_smoke(selected)
    artifact = run_wal_artifact_workflow(module, family, selected)
    passed = runtime.get("status") == "PASS" and artifact.get("status") == "PASS"
    result.update(
        {
            "status": "PASS" if passed else str(runtime.get("status", "FAIL")),
            "pass": passed,
            "reason": None if passed else runtime.get("reason", "CONTROLLED_WORKFLOW_FAILED"),
            "selected_candidate": asdict(selected),
            "runtime_smoke": runtime,
            "artifact_workflow": artifact,
            "validation_scope": [
                "local_snapshot_manifest",
                "tokenizer_load_local_files_only",
                "causal_lm_load_local_files_only",
                "forward_logits_finite",
                "deterministic_generation_executes",
                "wal_artifact_lifecycle",
                "rollback_restores_artifact_checksum",
            ],
            "semantic_edit_training": False,
            "weights_modified": False,
        }
    )
    return result


def write_result(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
