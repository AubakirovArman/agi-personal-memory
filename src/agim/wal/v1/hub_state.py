from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .format import serialize_wal_v1
from .nn import WALCachedLinear, WALLinear, WALParameter


def extract_wal_state_dict(model: nn.Module) -> Dict[str, Any]:
    """Extract a complete WAL state dict from a model.
    
    Returns dict with:
    - 'wal_blobs': {layer_name: serialized_bytes}
    - 'biases': {layer_name: tensor}
    - 'non_wal': {param_name: tensor} for non-WAL parameters
    - 'metadata': model architecture info
    """
    wal_blobs = {}
    biases = {}
    non_wal = {}
    wal_layers = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (WALLinear, WALCachedLinear)):
            blob = serialize_wal_v1(
                module.wal_weight.prog,
                module.wal_weight.atom_table,
                module.wal_weight.coeffs,
            )
            wal_blobs[f"{name}.wal_weight"] = blob
            wal_layers[name] = {
                "shape": list(module.wal_weight.shape),
                "dtype": str(module.wal_weight.dtype).replace("torch.", ""),
                "K0": module.wal_weight.atom_table.K0,
                "K_total": module.wal_weight.atom_table.K_total,
                "C": module.wal_weight.coeffs.values.numel(),
            }
            if module.bias is not None:
                biases[f"{name}.bias"] = module.bias.cpu()
        
    # Non-WAL parameters
    for name, param in model.named_parameters():
        if not any(name.startswith(wl) for wl in wal_layers):
            non_wal[name] = param.data.cpu()
    
    for name, buf in model.named_buffers():
        if not any(name.startswith(wl) for wl in wal_layers) and name not in non_wal:
            non_wal[name] = buf.data.cpu()
    
    return {
        "wal_blobs": wal_blobs,
        "biases": biases,
        "non_wal": non_wal,
        "wal_layers": wal_layers,
    }


def load_wal_state_dict(
    state_dict: Dict[str, Any],
    target_model: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """Load a WAL state dict. If target_model provided, reconstruct in-place.
    
    Returns:
        If target_model is None: dict of reconstructed components
        If target_model is provided: loaded model
    """
    from .format import deserialize_wal_v1
    
    wal_blobs = state_dict.get("wal_blobs", {})
    biases = state_dict.get("biases", {})
    non_wal = state_dict.get("non_wal", {})
    wal_layers = state_dict.get("wal_layers", {})
    
    reconstructed = {}
    
    for name, blob in wal_blobs.items():
        prog, atom_table, coeffs, meta = deserialize_wal_v1(blob)
        layer_name = name.replace(".wal_weight", "")
        layer_info = wal_layers.get(layer_name, {})
        shape = tuple(layer_info.get("shape", prog.shape))
        dtype = getattr(torch, layer_info.get("dtype", "float32"))
        
        wal_param = WALParameter(
            prog=prog,
            atom_table=atom_table,
            coeffs=coeffs,
            shape=shape,
            dtype=dtype,
        )
        reconstructed[name] = wal_param
    
    for name, bias in biases.items():
        reconstructed[name] = bias
    
    for name, param in non_wal.items():
        reconstructed[name] = param
    
    if target_model is not None:
        # Try to load into model
        for name, module in target_model.named_modules():
            if isinstance(module, (WALLinear, WALCachedLinear)):
                wal_key = f"{name}.wal_weight"
                if wal_key in reconstructed:
                    module.wal_weight = reconstructed[wal_key]
                bias_key = f"{name}.bias"
                if bias_key in reconstructed:
                    module.bias = reconstructed[bias_key].to(module.wal_weight.prog.atom_ids.device)
        
        # Load non-WAL parameters
        missing, unexpected = target_model.load_state_dict(
            {k: v for k, v in non_wal.items()}, strict=False
        )
        return target_model
    
    return reconstructed
