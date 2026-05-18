"""Helpers for loading EasyEdit metric functions without full plugin imports."""
from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import torch


DEFAULT_EASYEDIT_ROOT = Path("/mnt/hf_model_weights/arman/3bit/sites/EasyEdit")


def _module(name: str, path: str | None = None) -> types.ModuleType:
    mod = types.ModuleType(name)
    if path is not None:
        mod.__path__ = [path]  # type: ignore[attr-defined]
    sys.modules[name] = mod
    return mod


def _install_easyedit_stubs(root: Path) -> None:
    """Install tiny stubs for EasyEdit imports not needed by text metrics."""
    _module("easyeditor", str(root / "easyeditor"))
    _module("easyeditor.evaluate", str(root / "easyeditor" / "evaluate"))
    _module("easyeditor.editors", str(root / "easyeditor" / "editors"))
    util_mod = _module("easyeditor.util", str(root / "easyeditor" / "util"))
    trainer_mod = _module("easyeditor.trainer", str(root / "easyeditor" / "trainer"))
    _module("easyeditor.models", str(root / "easyeditor" / "models"))
    _module("easyeditor.models.melo", str(root / "easyeditor" / "models" / "melo"))

    class HyperParams:
        pass

    class LORA:
        pass

    util_mod.HyperParams = HyperParams
    trainer_mod.nn = torch.nn

    melo_mod = _module("easyeditor.models.melo.melo")
    melo_mod.LORA = LORA

    gen_mod = _module("easyeditor.util.generate")

    def generate_fast(*_args, **_kwargs):
        raise RuntimeError("generate_fast is not needed for this EasyEdit run")

    gen_mod.generate_fast = generate_fast

    if importlib.util.find_spec("nltk") is None:
        nltk_mod = _module("nltk")

        def word_tokenize(text: str) -> list[str]:
            return text.split()

        def ngrams(tokens: list[str], n: int):
            return zip(*[tokens[i:] for i in range(n)])

        class FreqDist(dict):
            def __init__(self, items):
                super().__init__()
                for item in items:
                    self[item] = self.get(item, 0) + 1

        nltk_mod.word_tokenize = word_tokenize
        nltk_mod.ngrams = ngrams
        nltk_mod.FreqDist = FreqDist


def _load_source_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_easyedit_official(root: Path) -> tuple[Any, Any, Any]:
    """Load official EasyEdit metric functions without importing all plugins."""
    _install_easyedit_stubs(root)
    utils = _load_source_module(
        "easyeditor.evaluate.evaluate_utils",
        root / "easyeditor" / "evaluate" / "evaluate_utils.py",
    )
    evaluate = _load_source_module(
        "easyeditor.evaluate.evaluate",
        root / "easyeditor" / "evaluate" / "evaluate.py",
    )
    editors_utils = _load_source_module(
        "easyeditor.editors.utils",
        root / "easyeditor" / "editors" / "utils.py",
    )
    return evaluate.compute_edit_quality, utils.test_prediction_acc, editors_utils.summary_metrics
