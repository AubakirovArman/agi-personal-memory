from __future__ import annotations

import torch
import torch.nn as nn

from .nn_encode import encode_linear_weight
from .nn_layers import WALCachedLinear, WALLinear, WALParameter


def replace_wal_with_linear(model: nn.Module) -> nn.Module:
    """Replace all WALLinear / WALCachedLinear layers with standard nn.Linear.
    
    Decodes WAL-encoded weights to dense tensors. Useful for the
    "edit in weight space, store in WAL space" workflow.
    
    Args:
        model: PyTorch model with WAL layers
    
    Returns:
        Modified model (in-place)
    """
    for name, module in model.named_children():
        if isinstance(module, (WALLinear, WALCachedLinear)):
            weight = module.wal_weight.decode(
                module.wal_weight._cache_device or torch.device("cpu")
            )
            bias = module.bias.data if module.bias is not None else None
            new_layer = nn.Linear(
                weight.shape[1],
                weight.shape[0],
                bias=bias is not None,
                dtype=weight.dtype,
                device=weight.device,
            )
            with torch.no_grad():
                new_layer.weight.copy_(weight)
                if bias is not None:
                    new_layer.bias.copy_(bias)
            setattr(model, name, new_layer)
        else:
            replace_wal_with_linear(module)
    return model


def replace_linear_with_wal(
    model: nn.Module,
    K: int = 256,
    C: int = 16,
    build_hier: bool = False,
    max_l1: int = 64,
    cached: bool = False,
) -> nn.Module:
    """Replace all nn.Linear layers in a model with WALLinear layers.
    
    Args:
        model: PyTorch model
        K: Number of base atoms
        C: Number of coefficients
        build_hier: Whether to build hierarchical atoms
        max_l1: Max L1 atoms
        cached: Use WALCachedLinear instead of WALLinear
    
    Returns:
        Modified model (in-place)
    """
    LinearClass = WALCachedLinear if cached else WALLinear
    
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Encode weight
            wal_param = encode_linear_weight(
                module.weight.data,
                K=K, C=C,
                build_hier=build_hier,
                max_l1=max_l1,
            )
            # Create replacement
            new_layer = LinearClass(
                wal_weight=wal_param,
                bias=module.bias.data if module.bias is not None else None,
            )
            # Replace
            setattr(model, name, new_layer)
        else:
            # Recurse
            replace_linear_with_wal(
                module, K=K, C=C,
                build_hier=build_hier,
                max_l1=max_l1,
                cached=cached,
            )
    
    return model


def wal_state_dict(model: nn.Module) -> dict:
    """Extract state dict with WAL-encoded weights.
    
    For WALLinear layers, serializes the WAL representation instead of
    dense weights.
    """
    from .format import serialize_wal_v1
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, (WALLinear, WALCachedLinear)):
            blob = serialize_wal_v1(
                module.wal_weight.prog,
                module.wal_weight.atom_table,
                module.wal_weight.coeffs,
            )
            state[f"{name}.wal_weight"] = blob
            if module.bias is not None:
                state[f"{name}.bias"] = module.bias
    return state


def wal_load_state_dict(model: nn.Module, state_dict: dict):
    """Load state dict with WAL-encoded weights.
    
    Deserializes WAL blobs and reconstructs WALLinear layers.
    """
    from .format import deserialize_wal_v1
    for name, module in model.named_modules():
        if isinstance(module, (WALLinear, WALCachedLinear)):
            blob = state_dict.get(f"{name}.wal_weight")
            if blob is not None:
                prog, atom_table, coeffs, meta = deserialize_wal_v1(blob)
                module.wal_weight = WALParameter(
                    prog=prog,
                    atom_table=atom_table,
                    coeffs=coeffs,
                    shape=tuple(meta['shape']),
                )
            bias_key = f"{name}.bias"
            if bias_key in state_dict:
                module.bias = state_dict[bias_key]
