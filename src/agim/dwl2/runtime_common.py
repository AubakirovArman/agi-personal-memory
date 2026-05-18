from __future__ import annotations

import json
from pathlib import Path

from torch import nn

def _set_submodule(root: nn.Module, path: str, module: nn.Module) -> None:
    parent = root
    parts = path.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def _load_shape_runtime_policy(shape_policy_json: str | Path | None) -> dict[str, dict[str, int]]:
    if shape_policy_json is None:
        return {}
    with open(shape_policy_json) as handle:
        payload = json.load(handle)
    return {
        str(item["tensor_name"]): {
            "group_rows": int(item["group_rows"]),
            "group_cols": int(item["group_cols"]),
        }
        for item in payload.get("selected", [])
    }


def _resolve_shape_policy(
    shape_policy: dict[str, dict[str, int]],
    module_name: str,
) -> tuple[str, dict[str, int]] | None:
    candidates = [f"{module_name}.weight"]
    if module_name.startswith("model."):
        candidates.append(f"{module_name[6:]}.weight")
    else:
        candidates.append(f"model.{module_name}.weight")
    for tensor_name in candidates:
        match = shape_policy.get(tensor_name)
        if match is not None:
            return tensor_name, match
    return None


def _load_fused_allowlist(fused_allowlist_json: str | Path | None) -> set[str]:
    if fused_allowlist_json is None:
        return set()
    with open(fused_allowlist_json) as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return {str(item) for item in payload}
    if "promoted_layers" in payload:
        return {str(item) for item in payload["promoted_layers"]}
    adaptive_names = payload.get("adaptive_names")
    if isinstance(adaptive_names, dict) and "primary_enabled_layer_names" in adaptive_names:
        return {str(item) for item in adaptive_names["primary_enabled_layer_names"]}
    return set()


def _module_name_candidates(module_name: str) -> tuple[str, ...]:
    if module_name.startswith("model."):
        return (module_name, module_name[6:])
    return (module_name, f"model.{module_name}")
