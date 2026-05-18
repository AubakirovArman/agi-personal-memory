from __future__ import annotations

from typing import List

import torch.nn as nn

from .nn import WALCachedLinear, WALLinear, WALParameter
from .mergekit_config import MergeConfig


def merge_task_vectors(
    base_model: nn.Module,
    finetuned_models: List[nn.Module],
    config: MergeConfig,
) -> nn.Module:
    """Merge task vectors (finetuned - base) rather than raw weights.
    
    This is the standard Task Arithmetic / TIES approach.
    """
    import copy
    merged = copy.deepcopy(base_model)
    
    # Compute task vectors
    task_vectors = []
    for ft_model in finetuned_models:
        tv = copy.deepcopy(ft_model)
        for (name1, m1), (name2, m2) in zip(tv.named_modules(), base_model.named_modules()):
            if isinstance(m1, (WALLinear, WALCachedLinear)) and isinstance(m2, (WALLinear, WALCachedLinear)):
                w1 = m1.wal_weight.decode()
                w2 = m2.wal_weight.decode()
                # Store task vector as decoded diff (will re-encode later)
                if not hasattr(m1, '_task_vector'):
                    m1._task_vector = w1 - w2
        task_vectors.append(tv)
    
    # Average task vectors and apply to base
    weights = config.weights if config.weights else [1.0 / len(task_vectors)] * len(task_vectors)
    
    for name, module in merged.named_modules():
        if isinstance(module, (WALLinear, WALCachedLinear)):
            base_w = module.wal_weight.decode()
            
            # Collect task vectors
            tvs = []
            for i, tv_model in enumerate(task_vectors):
                tv_module = dict(tv_model.named_modules()).get(name)
                if tv_module is not None and hasattr(tv_module, '_task_vector'):
                    tvs.append(tv_module._task_vector * weights[i])
            
            if not tvs:
                continue
            
            merged_w = base_w + sum(tvs)
            
            # Re-encode
            from .nn import encode_linear_weight
            new_wal_param = encode_linear_weight(
                merged_w,
                K=module.wal_weight.atom_table.K0,
                C=module.wal_weight.coeffs.values.numel(),
            )
            
            module.wal_weight = WALParameter(
                prog=new_wal_param.prog,
                atom_table=new_wal_param.atom_table,
                coeffs=new_wal_param.coeffs,
                shape=module.wal_weight.shape,
                dtype=module.wal_weight.dtype,
            )
            module.clear_cache() if hasattr(module, 'clear_cache') else None
    
    return merged
