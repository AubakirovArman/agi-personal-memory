"""Run metadata helpers for EasyEdit-compatible artifacts."""
from __future__ import annotations

import hashlib
import json
from typing import Any


ARTIFACT_SCHEMA_VERSION = "easyedit_official.v2"
DEFAULT_FAILURE_FAMILIES = ("tf", "ctx_gen", "prob")
FAILURE_FAMILY_CHOICES = ("tf", "ctx_gen", "prob", "vanilla_gen")


def parse_failure_families(value: str | None) -> tuple[str, ...]:
    if not value:
        return DEFAULT_FAILURE_FAMILIES
    families = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(families) - set(FAILURE_FAMILY_CHOICES))
    if unknown:
        raise ValueError(f"Unknown failure families: {', '.join(unknown)}")
    return families or DEFAULT_FAILURE_FAMILIES


def method_profile_id(args) -> str:
    explicit = getattr(args, "method_profile_id", None)
    if explicit:
        return explicit
    if not getattr(args, "wal_encode_updates", True):
        return "seq_exact_additive" if getattr(args, "sequential_edit", False) else (
            "single_exact_additive"
        )
    if getattr(args, "sequential_edit", False):
        if getattr(args, "positive_constraint_mode", "none") == "ridge":
            return "seq_positive_ridge"
        if getattr(args, "positive_constraint_mode", "none") == "projected":
            return "seq_positive_projected"
        if getattr(args, "use_positive_prompts", False):
            return "seq_positive"
        if getattr(args, "projection_mode", "sequential") == "orthogonal":
            return "seq_orthogonal"
        if getattr(args, "history_slot_mode", "global") == "relation":
            return "seq_relation_slots"
        return "seq_tuned"
    if getattr(args, "use_positive_prompts", False):
        if getattr(args, "positive_constraint_mode", "none") == "ridge":
            return "single_positive_ridge"
        if getattr(args, "positive_constraint_mode", "none") == "projected":
            return "single_positive_projected"
        return "single_positive"
    if getattr(args, "neg_prompt_limit", 0) <= 4:
        return "single_ps"
    return "single_loc"


def base_model_digest(model: Any, args) -> str:
    payload: dict[str, Any] = {
        "model": getattr(args, "model", None),
    }
    config = getattr(model, "config", None)
    if config is not None:
        if hasattr(config, "to_dict"):
            payload["config"] = config.to_dict()
        else:
            payload["config_repr"] = repr(config)
    return _sha_json(payload)


def atoms_digest(editor: Any) -> str | None:
    atoms = getattr(editor, "atoms", None)
    if atoms is None:
        return None
    tensor = atoms.detach().float().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return f"sha256:{digest.hexdigest()}"


def _sha_json(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"
