"""Named EasyEdit run presets."""
from __future__ import annotations

from typing import Any


ANTI_PROFILES: dict[str, dict[str, Any]] = {
    "off": {},
    "target_low": {
        "clamp_anti": 0.02,
        "clamp_anti_scope": "target",
    },
    "subject_low": {
        "clamp_anti": 0.02,
        "clamp_anti_scope": "subject",
    },
    "both_low": {
        "clamp_anti": 0.02,
        "clamp_anti_scope": "both",
    },
}


POSITIVE_PROFILES: dict[str, dict[str, Any]] = {
    "off": {},
    "w015": {
        "use_positive_prompts": True,
        "positive_prompt_limit": 1,
        "positive_key_weight": 0.15,
    },
    "w025": {
        "use_positive_prompts": True,
        "positive_prompt_limit": 2,
        "positive_key_weight": 0.25,
    },
    "w035": {
        "use_positive_prompts": True,
        "positive_prompt_limit": 2,
        "positive_key_weight": 0.35,
    },
    "w025_ridge": {
        "use_positive_prompts": True,
        "positive_prompt_limit": 2,
        "positive_key_weight": 0.25,
        "positive_constraint_mode": "ridge",
    },
}


PRESETS: dict[str, dict[str, Any]] = {
    "random_50_seed_42": {
        "n": 50,
        "sample_policy": "random",
        "seed": 42,
        "output": "results/easyedit_official/current/random_50_seed_42.json",
    },
    "random_50_seed_43": {
        "n": 50,
        "sample_policy": "random",
        "seed": 43,
        "output": "results/easyedit_official/current/random_50_seed_43.json",
    },
    "random_50_seed_44": {
        "n": 50,
        "sample_policy": "random",
        "seed": 44,
        "output": "results/easyedit_official/current/random_50_seed_44.json",
    },
    "ablation_selective_anti_repetition_seed42": {
        "n": 50,
        "sample_policy": "random",
        "seed": 42,
        "output": "results/easyedit_official/ablations/ablation_selective_anti_repetition_seed42.json",
        "positive_profile": "off",
        "anti_profile": "target_low",
    },
    "ablation_kpos_positive_w025_seed42": {
        "n": 50,
        "sample_policy": "random",
        "seed": 42,
        "output": "results/easyedit_official/ablations/ablation_kpos_positive_w025_seed42.json",
        "positive_profile": "w025",
    },
    "ablation_kpos_kneg_ridge_seed42": {
        "n": 50,
        "sample_policy": "random",
        "seed": 42,
        "output": "results/easyedit_official/ablations/ablation_kpos_kneg_ridge_seed42.json",
        "positive_profile": "w025_ridge",
    },
}


def _apply_profile(args, name: str, profiles: dict[str, dict[str, Any]]) -> None:
    profile = profiles.get(name, {})
    for key, value in profile.items():
        setattr(args, key, value)


def apply_profiles(args) -> Any:
    anti_profile = getattr(args, "anti_profile", "off")
    if anti_profile not in ANTI_PROFILES:
        raise ValueError(f"Unknown anti profile: {anti_profile}")
    positive_profile = getattr(args, "positive_profile", "off")
    if positive_profile not in POSITIVE_PROFILES:
        raise ValueError(f"Unknown positive profile: {positive_profile}")
    _apply_profile(args, anti_profile, ANTI_PROFILES)
    _apply_profile(args, positive_profile, POSITIVE_PROFILES)
    return args


def apply_preset(args):
    if args.preset:
        preset = PRESETS[args.preset]
        for key, value in preset.items():
            setattr(args, key, value)
    return apply_profiles(args)
