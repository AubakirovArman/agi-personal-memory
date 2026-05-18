from __future__ import annotations

import struct
from typing import List, Tuple

import numpy as np
import torch

from .isa import AtomDef


MAGIC = b'WAL1'
VERSION = 1

# Flags
FLAG_HAS_RESIDUALS = 1 << 0
FLAG_PACKED_COEFFS = 1 << 1

# Op codes
OP_ADD = 0
OP_MUL = 1
OP_NEG = 2
OP_CONST = 3

_OP_ENCODE = {"ADD": OP_ADD, "MUL": OP_MUL, "NEG": OP_NEG, "CONST": OP_CONST}
_OP_DECODE = {v: k for k, v in _OP_ENCODE.items()}


def _pack_uint4(values: torch.Tensor) -> torch.Tensor:
    """Pack uint8 values (0..15) into uint4 pairs."""
    N = values.numel()
    padded = values
    if N % 2 == 1:
        padded = torch.cat([values, torch.zeros(1, dtype=torch.uint8)])
    even = padded[0::2] & 0x0F
    odd = padded[1::2] & 0x0F
    return (even << 4) | odd


def _unpack_uint4(packed: torch.Tensor, N: int) -> torch.Tensor:
    """Unpack uint4 pairs into uint8 values."""
    even = (packed >> 4) & 0x0F
    odd = packed & 0x0F
    values = torch.stack([even, odd], dim=1).flatten()[:N]
    return values


def _pack_bitmap(bool_tensor: torch.Tensor) -> bytes:
    """Pack boolean tensor into bit-packed bytes."""
    arr = bool_tensor.cpu().numpy().astype(np.uint8)
    return np.packbits(arr).tobytes()


def _unpack_bitmap(data: bytes, N: int) -> torch.Tensor:
    """Unpack bit-packed bytes into boolean tensor."""
    arr = np.unpackbits(np.frombuffer(data, dtype=np.uint8))[:N]
    return torch.from_numpy(arr).bool()


def _serialize_hierarchical(atom_defs: List[AtomDef]) -> bytes:
    """Serialize hierarchical atom definitions to bytes.
    
    Only serializes L1+ definitions (index >= K0).
    Assumes atom_defs[0:K0] are L0 base atoms.
    """
    # Count L1+ defs
    l1_defs = [(i, d) for i, d in enumerate(atom_defs) if d.level > 0]
    count = len(l1_defs)
    
    buf = struct.pack('<H', count)
    for idx, d in l1_defs:
        op_code = _OP_ENCODE.get(d.op, OP_CONST)
        n_children = len(d.children) if d.children else 0
        buf += struct.pack('<BB', op_code, n_children)
        if n_children > 0:
            buf += struct.pack('<' + 'H' * n_children, *d.children)
            buf += struct.pack('<' + 'f' * n_children, *d.coeffs)
    return buf


def _deserialize_hierarchical(data: bytes, offset: int, K0: int) -> Tuple[List[AtomDef], int]:
    """Deserialize hierarchical atom definitions from bytes.
    
    Returns (atom_defs_list, new_offset).
    """
    count = struct.unpack_from('<H', data, offset)[0]
    offset += 2
    
    defs = [AtomDef(level=0, op="CONST") for _ in range(K0)]
    
    for _ in range(count):
        op_code, n_children = struct.unpack_from('<BB', data, offset)
        offset += 2
        op = _OP_DECODE.get(op_code, "CONST")
        
        if n_children > 0:
            children = list(struct.unpack_from('<' + 'H' * n_children, data, offset))
            offset += n_children * 2
            coeffs = list(struct.unpack_from('<' + 'f' * n_children, data, offset))
            offset += n_children * 4
        else:
            children = None
            coeffs = None
        
        level = 1 if op_code != OP_CONST else 0
        defs.append(AtomDef(level=level, op=op, children=children, coeffs=coeffs))
    
    return defs, offset
