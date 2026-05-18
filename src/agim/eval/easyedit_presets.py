"""Named EasyEdit run presets."""
from __future__ import annotations

from typing import Any


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
}


def apply_preset(args):
    if not args.preset:
        return args
    preset = PRESETS[args.preset]
    for key, value in preset.items():
        setattr(args, key, value)
    return args
