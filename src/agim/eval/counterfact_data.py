"""CounterFact dataset and repository metadata helpers."""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import urllib.request
from pathlib import Path
from typing import Any


LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
COUNTERFACT_URL = "https://rome.baulab.info/data/dsets/counterfact.json"


def load_dataset(source: str) -> tuple[list[dict[str, Any]], str]:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source) as response:
            raw = response.read()
    else:
        raw = Path(source).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    return json.loads(raw), digest


def select_facts(data: list[dict[str, Any]], n: int, policy: str,
                 seed: int) -> list[dict[str, Any]]:
    if policy == "random":
        rng = random.Random(seed)
        return rng.sample(data, min(n, len(data)))
    return data[:n]


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None
