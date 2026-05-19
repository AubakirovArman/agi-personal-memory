"""EasyEdit-compatible AGIM WAL method adapter.

This module is intentionally importable from an external EasyEdit checkout:

    from agim.integrations.easyedit_agimwal import (
        AGIMWALHyperParams,
        apply_agimwal_to_model,
    )

Register those objects in EasyEdit's ``ALG_DICT`` / hparams imports to run
AGIM WAL as a regular EasyEdit method while keeping AGIM as the source of the
editing implementation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any


EASYEDIT_ALG_NAME = "AGIMWAL"


@dataclass
class AGIMWALHyperParams:
    """Minimal EasyEdit hparams surface for AGIM WAL."""

    model_name: str
    device: int | str = 0
    alg_name: str = EASYEDIT_ALG_NAME
    backend: str = "dual_row"
    K: int = 256
    lmax: int = 16
    clamp_lm: float = 0.20
    clamp_embed: float = 0.06
    clamp_eos: float = 0.0
    clamp_anti: float = 0.06
    clamp_old: float = 0.0
    target_token_mode: str = "contextual"
    use_neg_prompts: bool = True
    neg_prompt_limit: int = 10
    use_positive_prompts: bool = False
    positive_prompt_limit: int = 4
    positive_key_weight: float = 1.0
    positive_constraint_mode: str = "none"
    positive_constraint_k_pos: int = 4
    positive_constraint_k_neg: int = 4
    clamp_anti_scope: str = "both"
    neg_projection_strength: float = 0.3
    history_projection_strength: float = 0.0
    embed_history_projection_strength: float = 0.0
    projection_mode: str = "sequential"
    history_slot_mode: str = "global"
    max_history_keys: int = 128
    state_namespace: str = "easyedit"
    relation_protected_mode: str = "none"
    max_relation_protected_keys: int = 64
    wal_encode_updates: bool = True
    nt_sample_size: int = 500
    rome_target_layer: int = 7
    rome_candidate_layers: str = ""
    rome_top_rows: int = 32
    rome_clamp: float = 0.08
    rome_auto_locate: bool = False
    max_length: int = 512
    batch_size: int = 1
    model_parallel: bool = False
    fp16: bool = True

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> "AGIMWALHyperParams":
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(config) - known)
        if unknown:
            raise ValueError(f"Unknown AGIMWAL hparams: {unknown}")
        if config.get("alg_name", EASYEDIT_ALG_NAME) != EASYEDIT_ALG_NAME:
            raise ValueError("AGIMWALHyperParams requires alg_name='AGIMWAL'")
        return cls(**config)

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str) -> "AGIMWALHyperParams":
        path = hparams_name_or_path
        if not path.endswith((".yaml", ".yml")):
            path = f"{path}.yaml"
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - exercised by users only.
            raise RuntimeError("PyYAML is required to load EasyEdit YAML hparams") from exc
        with open(path, "r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream) or {}
        return cls.from_mapping(payload)

    @property
    def device_string(self) -> str:
        text = str(self.device)
        if text == "cpu" or text.startswith("cuda:"):
            return text
        return f"cuda:{text}"


def apply_agimwal_to_model(
    model,
    tok,
    request: list[dict[str, Any]] | dict[str, Any],
    hparams: AGIMWALHyperParams,
    copy: bool = False,
    return_orig_weights: bool = False,
    keep_original_weight: bool = False,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """EasyEdit algorithm entry point.

    The return shape matches EasyEdit methods: ``(edited_model, weights_copy)``.
    ``weights_copy`` uses full tensors because EasyEdit's BaseEditor restores
    edits through its generic parameter-copy rollback path.
    """
    if copy:
        model = deepcopy(model)
    weights_copy = {}
    if return_orig_weights or keep_original_weight:
        weights_copy = snapshot_easyedit_weights(model, hparams)

    editor = kwargs.pop("editor_factory", _build_editor)(model, tok, hparams)
    editor.nt_sample_size = hparams.nt_sample_size
    editor.build_vocab()
    for item in _request_list(request):
        apply_agimwal_request(editor, item, hparams)
    return model, weights_copy


def apply_agimwal_request(
    editor,
    request: dict[str, Any],
    hparams: AGIMWALHyperParams,
) -> dict[str, Any]:
    """Apply one normalized EasyEdit request through an AGIM WAL editor."""
    if hparams.backend not in {"dual_row", "wal_rome"}:
        raise ValueError("AGIMWAL backend must be dual_row or wal_rome")
    subject = str(request.get("subject", ""))
    prompt = _format_prompt(str(request.get("prompt", "")), subject)
    positive_prompts = _positive_prompts(request) if hparams.use_positive_prompts else []
    return editor.apply_edit(
        subject=subject,
        target=_target_text(request.get("target_new", "")),
        relation=str(request.get("relation_id") or request.get("relation") or ""),
        prompt=prompt,
        old_target=_target_text(
            request.get("ground_truth")
            or request.get("target_true")
            or request.get("target_old")
            or ""
        ),
        clamp_lm=hparams.clamp_lm,
        clamp_embed=hparams.clamp_embed,
        clamp_eos=hparams.clamp_eos,
        clamp_anti=hparams.clamp_anti,
        clamp_old=hparams.clamp_old,
        target_token_mode=hparams.target_token_mode,
        neg_prompts=_negative_prompts(request) if hparams.use_neg_prompts else [],
        max_neg_prompts=hparams.neg_prompt_limit,
        positive_prompts=positive_prompts,
        max_positive_prompts=hparams.positive_prompt_limit,
        positive_key_weight=hparams.positive_key_weight,
        positive_constraint_mode=hparams.positive_constraint_mode,
        positive_constraint_k_pos=hparams.positive_constraint_k_pos,
        positive_constraint_k_neg=hparams.positive_constraint_k_neg,
        neg_projection_strength=hparams.neg_projection_strength,
        clamp_anti_scope=hparams.clamp_anti_scope,
        history_projection_strength=hparams.history_projection_strength,
        embed_history_projection_strength=hparams.embed_history_projection_strength,
        projection_mode=hparams.projection_mode,
        history_slot_mode=hparams.history_slot_mode,
        max_history_keys=hparams.max_history_keys,
        state_namespace=hparams.state_namespace,
        relation_protected_mode=hparams.relation_protected_mode,
        max_relation_protected_keys=hparams.max_relation_protected_keys,
        wal_encode_updates=hparams.wal_encode_updates,
    )


def snapshot_easyedit_weights(model, hparams: AGIMWALHyperParams) -> dict[str, Any]:
    """Return full-tensor backups in EasyEdit rollback format."""
    return {
        name: _get_parameter(model, name).detach().clone()
        for name in easyedit_weight_names(model, hparams)
    }


def easyedit_weight_names(model, hparams: AGIMWALHyperParams) -> list[str]:
    if hparams.backend == "dual_row":
        return ["lm_head.weight", "model.embed_tokens.weight"]
    if hparams.backend != "wal_rome":
        raise ValueError("AGIMWAL backend must be dual_row or wal_rome")
    return [
        f"model.layers.{layer}.mlp.down_proj.weight"
        for layer in _rome_snapshot_layers(model, hparams)
    ]


def easyedit_registration_snippet() -> str:
    return (
        "from agim.integrations.easyedit_agimwal import "
        "AGIMWALHyperParams, apply_agimwal_to_model\n"
        "ALG_DICT['AGIMWAL'] = apply_agimwal_to_model\n"
    )


def _build_editor(model, tok, hparams: AGIMWALHyperParams):
    if hparams.backend == "wal_rome":
        from agim.model.wal_rome_editor import WALRomeEditor

        return WALRomeEditor(
            model,
            tok,
            K=hparams.K,
            lmax=hparams.lmax,
            device=hparams.device_string,
            target_layer=hparams.rome_target_layer,
            candidate_layers=_csv_ints(hparams.rome_candidate_layers),
            top_rows=hparams.rome_top_rows,
            clamp_rome=hparams.rome_clamp,
            auto_locate=hparams.rome_auto_locate,
        )
    if hparams.backend != "dual_row":
        raise ValueError("AGIMWAL backend must be dual_row or wal_rome")
    from agim.model.wal_dual_editor import WALDualLayerEditor

    return WALDualLayerEditor(
        model,
        tok,
        K=hparams.K,
        lmax=hparams.lmax,
        device=hparams.device_string,
    )


def _request_list(request: list[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
    return request if isinstance(request, list) else [request]


def _target_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("str", ""))
    return str(value)


def _format_prompt(prompt: str, subject: str) -> str:
    if "{}" in prompt:
        return prompt.format(subject)
    return prompt


def _negative_prompts(request: dict[str, Any]) -> list[str]:
    prompts: list[str] = []
    for value in (request.get("locality") or {}).values():
        prompt = value.get("prompt", []) if isinstance(value, dict) else []
        prompts.extend(_as_text_list(prompt))
    return [_format_prompt(prompt, str(request.get("subject", ""))) for prompt in prompts]


def _positive_prompts(request: dict[str, Any]) -> list[str]:
    prompts = request.get("rephrase_prompts") or request.get("rephrase_prompt") or []
    subject = str(request.get("subject", ""))
    return [_format_prompt(prompt, subject) for prompt in _as_text_list(prompts)]


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _get_parameter(model, path: str):
    obj = model
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _rome_snapshot_layers(model, hparams: AGIMWALHyperParams) -> list[int]:
    if not hparams.rome_auto_locate:
        return [int(hparams.rome_target_layer)]
    layers = _csv_ints(hparams.rome_candidate_layers)
    if layers:
        return layers
    try:
        return list(range(min(12, len(model.model.layers))))
    except AttributeError:
        return [int(hparams.rome_target_layer)]


def _csv_ints(value: str | list[int] | tuple[int, ...]) -> list[int] | None:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    if not str(value).strip():
        return None
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]
