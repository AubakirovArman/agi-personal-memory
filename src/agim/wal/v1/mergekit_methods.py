from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .meta import program_soup
from .nn import WALCachedLinear, WALLinear, WALParameter
from .mergekit_config import MergeConfig


def _merge_soup(
    base_model: nn.Module,
    models: List[nn.Module],
    config: MergeConfig,
) -> nn.Module:
    """Program soup merge: merge programs at atom/coeff level."""
    import copy
    merged = copy.deepcopy(base_model)
    
    for name, module in merged.named_modules():
        if isinstance(module, (WALLinear, WALCachedLinear)):
            # Collect programs from all models
            progs = []
            for m in models:
                src_module = dict(m.named_modules()).get(name)
                if src_module is not None and isinstance(src_module, (WALLinear, WALCachedLinear)):
                    progs.append(src_module.wal_weight.prog)
            
            if len(progs) < 2:
                continue
            
            # Merge programs
            weights = config.weights if config.weights else [1.0 / len(progs)] * len(progs)
            merged_prog = program_soup(progs, weights=weights, method=config.soup_method)
            
            # Create new WALParameter with merged program
            module.wal_weight = WALParameter(
                prog=merged_prog,
                atom_table=module.wal_weight.atom_table,
                coeffs=module.wal_weight.coeffs,
                shape=module.wal_weight.shape,
                dtype=module.wal_weight.dtype,
            )
            module.clear_cache() if hasattr(module, 'clear_cache') else None
    
    return merged


def _merge_linear(
    base_model: nn.Module,
    models: List[nn.Module],
    config: MergeConfig,
) -> nn.Module:
    """Linear interpolation in decoded weight space."""
    import copy
    merged = copy.deepcopy(base_model)
    
    weights = config.weights if config.weights else [1.0 / len(models)] * len(models)
    
    for name, module in merged.named_modules():
        if isinstance(module, (WALLinear, WALCachedLinear)):
            # Decode all weights
            decoded = []
            for i, m in enumerate(models):
                src_module = dict(m.named_modules()).get(name)
                if src_module is not None and isinstance(src_module, (WALLinear, WALCachedLinear)):
                    w = src_module.wal_weight.decode().to(module.wal_weight.prog.atom_ids.device)
                    decoded.append(w * weights[i])
            
            if len(decoded) < 2:
                continue
            
            # Average decoded weights
            merged_weight = sum(decoded)
            
            # Re-encode merged weight
            from .nn import encode_linear_weight
            new_wal_param = encode_linear_weight(
                merged_weight,
                K=module.wal_weight.atom_table.K0,
                C=module.wal_weight.coeffs.values.numel(),
            )
            
            # Preserve shape/dtype
            module.wal_weight = WALParameter(
                prog=new_wal_param.prog,
                atom_table=new_wal_param.atom_table,
                coeffs=new_wal_param.coeffs,
                shape=module.wal_weight.shape,
                dtype=module.wal_weight.dtype,
            )
            module.clear_cache() if hasattr(module, 'clear_cache') else None
    
    return merged


def _merge_slerp(
    base_model: nn.Module,
    models: List[nn.Module],
    config: MergeConfig,
) -> nn.Module:
    """Spherical Linear Interpolation (SLERP) in decoded weight space.
    
    Only supports 2 models currently.
    """
    if len(models) != 2:
        raise ValueError("SLERP requires exactly 2 models")
    
    import copy
    merged = copy.deepcopy(base_model)
    
    t = config.weights[1] if config.weights else 0.5
    
    for name, module in merged.named_modules():
        if isinstance(module, (WALLinear, WALCachedLinear)):
            m0 = dict(models[0].named_modules()).get(name)
            m1 = dict(models[1].named_modules()).get(name)
            
            if m0 is None or m1 is None:
                continue
            
            w0 = m0.wal_weight.decode().flatten()
            w1 = m1.wal_weight.decode().flatten()
            
            # SLERP formula
            dot = torch.dot(w0, w1)
            dot = torch.clamp(dot, -1.0, 1.0)
            
            theta = torch.acos(dot)
            sin_theta = torch.sin(theta)
            
            if sin_theta < config.epsilon:
                # Linear interpolation for very close vectors
                merged_w = w0 * (1 - t) + w1 * t
            else:
                merged_w = w0 * (torch.sin((1 - t) * theta) / sin_theta) + \
                           w1 * (torch.sin(t * theta) / sin_theta)
            
            merged_weight = merged_w.reshape(module.wal_weight.shape)
            
            # Re-encode
            from .nn import encode_linear_weight
            new_wal_param = encode_linear_weight(
                merged_weight,
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


def _merge_ties(
    base_model: nn.Module,
    models: List[nn.Module],
    config: MergeConfig,
) -> nn.Module:
    """TIES-style merging: trim, elect sign, merge.
    
    Simplified TIES: keep top-k% of parameter changes by magnitude,
    resolve sign conflicts, then average.
    """
    import copy
    merged = copy.deepcopy(base_model)
    
    for name, module in merged.named_modules():
        if isinstance(module, (WALLinear, WALCachedLinear)):
            base_w = module.wal_weight.decode().flatten()
            
            deltas = []
            for m in models[1:]:
                src_module = dict(m.named_modules()).get(name)
                if src_module is not None and isinstance(src_module, (WALLinear, WALCachedLinear)):
                    w = src_module.wal_weight.decode().flatten()
                    deltas.append(w - base_w)
            
            if not deltas:
                continue
            
            # Stack deltas
            delta_stack = torch.stack(deltas)  # [N-1, D]
            
            # Trim: keep top-k% by magnitude
            if config.density < 1.0:
                flat_mags = delta_stack.abs().flatten()
                k = int(config.density * flat_mags.numel())
                if k > 0:
                    threshold = torch.topk(flat_mags, k, largest=True)[0][-1]
                    mask = delta_stack.abs() >= threshold
                    delta_stack = delta_stack * mask
            
            # Elect sign: majority vote
            signs = torch.sign(delta_stack)
            sign_votes = signs.sum(dim=0)
            elected_sign = torch.sign(sign_votes)
            elected_sign[elected_sign == 0] = 1  # Default to positive
            
            # Apply elected sign and average
            avg_delta = delta_stack.mean(dim=0)
            merged_delta = avg_delta.abs() * elected_sign
            
            merged_weight = base_w + merged_delta
            merged_weight = merged_weight.reshape(module.wal_weight.shape)
            
            # Re-encode
            from .nn import encode_linear_weight
            new_wal_param = encode_linear_weight(
                merged_weight,
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
