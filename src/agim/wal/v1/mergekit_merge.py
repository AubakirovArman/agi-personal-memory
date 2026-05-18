from __future__ import annotations

from typing import List

import torch.nn as nn

from .mergekit_config import MergeConfig
from .mergekit_methods import _merge_linear, _merge_slerp, _merge_soup, _merge_ties


def merge_wal_models(
    models: List[nn.Module],
    config: MergeConfig,
) -> nn.Module:
    """Merge multiple WAL-encoded models.
    
    Args:
        models: List of models with WAL layers
        config: Merge configuration
    
    Returns:
        Merged model (first model is used as base)
    """
    if not models:
        raise ValueError("Empty model list")
    if len(models) == 1:
        return models[0]
    
    base_model = models[0]
    
    if config.method == "soup":
        return _merge_soup(base_model, models, config)
    elif config.method == "linear":
        return _merge_linear(base_model, models, config)
    elif config.method == "slerp":
        return _merge_slerp(base_model, models, config)
    elif config.method == "ties":
        return _merge_ties(base_model, models, config)
    else:
        raise ValueError(f"Unknown merge method: {config.method}")
