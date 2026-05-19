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
    edit_backend = getattr(args, "edit_backend", "dual_row")
    if edit_backend == "wal_rome":
        return _append_profile_id(
            "seq_wal_rome" if getattr(args, "sequential_edit", False) else "single_wal_rome",
            args,
        )
    if edit_backend == "wal_memit":
        return _append_profile_id(
            "seq_wal_memit" if getattr(args, "sequential_edit", False) else "single_wal_memit",
            args,
        )
    if not getattr(args, "wal_encode_updates", True):
        return _append_profile_id(
            "seq_exact_additive" if getattr(args, "sequential_edit", False) else "single_exact_additive",
            args,
        )
    if getattr(args, "sequential_edit", False):
        if edit_backend == "side_slot":
            slot_buckets = int(getattr(args, "relation_slot_buckets", 0) or 0)
            if slot_buckets > 0:
                return _append_profile_id(f"seq_side_slot_sharded_{slot_buckets}", args)
            return _append_profile_id("seq_side_slot", args)
        relation_mode = getattr(args, "relation_protected_mode", "none")
        if relation_mode != "none":
            return _append_profile_id(
                f"seq_relation_protected_{relation_mode}",
                args,
            )
        if getattr(args, "positive_constraint_mode", "none") == "ridge":
            return _append_profile_id("seq_positive_ridge", args)
        if getattr(args, "positive_constraint_mode", "none") == "projected":
            return _append_profile_id("seq_positive_projected", args)
        if getattr(args, "positive_constraint_mode", "none") == "constrained":
            return _append_profile_id("seq_positive_constrained", args)
        if getattr(args, "use_positive_prompts", False):
            return _append_profile_id("seq_positive", args)
        if getattr(args, "projection_mode", "sequential") == "orthogonal":
            return _append_profile_id("seq_orthogonal", args)
        if getattr(args, "history_slot_mode", "global") == "relation":
            return _append_profile_id("seq_relation_slots", args)
        return _append_profile_id("seq_tuned", args)
    if edit_backend == "side_slot":
        slot_buckets = int(getattr(args, "relation_slot_buckets", 0) or 0)
        if slot_buckets > 0:
            return _append_profile_id(f"single_side_slot_sharded_{slot_buckets}", args)
        return _append_profile_id("single_side_slot", args)
    relation_mode = getattr(args, "relation_protected_mode", "none")
    if relation_mode != "none":
        return _append_profile_id(f"single_relation_protected_{relation_mode}", args)
    if getattr(args, "positive_constraint_mode", "none") == "constrained":
        return _append_profile_id("single_positive_constrained", args)
    if getattr(args, "use_positive_prompts", False):
        if getattr(args, "positive_constraint_mode", "none") == "ridge":
            return _append_profile_id("single_positive_ridge", args)
        if getattr(args, "positive_constraint_mode", "none") == "projected":
            return _append_profile_id("single_positive_projected", args)
        if getattr(args, "positive_constraint_mode", "none") == "constrained":
            return _append_profile_id("single_positive_constrained", args)
        return _append_profile_id("single_positive", args)
    if getattr(args, "neg_prompt_limit", 0) <= 4:
        return _append_profile_id("single_ps", args)
    return _append_profile_id("single_loc", args)


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


def _append_profile_id(profile_id: str, args) -> str:
    anti_profile = getattr(args, "anti_profile", "off")
    positive_profile = getattr(args, "positive_profile", "off")

    tokens: list[str] = [profile_id]
    if anti_profile and anti_profile != "off":
        tokens.append(f"anti_{anti_profile}")
    if positive_profile and positive_profile != "off":
        tokens.append(f"pos_{positive_profile}")
    return "_".join(tokens)
