"""Small utilities for EasyEdit-compatible runner output."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


def parse_retention_steps(value: str, total: int) -> list[int]:
    if not value.strip():
        return []
    steps = set()
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        step = int(raw)
        if step <= 0:
            raise ValueError("retention steps must be positive")
        if step <= total:
            steps.add(step)
    return sorted(steps)


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def parse_device_id(device: str) -> int:
    if device in {"cuda", "cpu"}:
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    return int(device)
