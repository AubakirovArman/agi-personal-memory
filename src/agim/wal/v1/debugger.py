#!/usr/bin/env python3
"""WAL v1 Debugger & Inspector facade/core class."""
from __future__ import annotations

from typing import Callable, List, Optional

import torch

from .isa import AtomTableV1, CoeffTable, ProgramBufferV1
from .debugger_analysis import WALDebuggerAnalysisMixin
from .debugger_print import WALDebuggerPrintMixin
from .debugger_types import Breakpoint, HeatmapStats, TraceRecord


class WALDebugger(WALDebuggerAnalysisMixin, WALDebuggerPrintMixin):
    """Debugger for WAL v1 programs.
    
    Supports:
    - Step-through execution with breakpoints
    - Hierarchical atom resolution tracing
    - Program heatmaps and statistics
    - Program diff
    """
    
    def __init__(self, atom_table: AtomTableV1, coeffs):
        self.atom_table = atom_table
        # Accept either CoeffTable or raw tensor
        if hasattr(coeffs, 'values') and callable(getattr(coeffs, 'values', None)):
            # It's a tensor — .values is a method
            self.coeff_values = coeffs
        else:
            # It's a CoeffTable
            self.coeff_values = coeffs.values
        self.breakpoints: List[Breakpoint] = []
        self.trace_log: List[TraceRecord] = []
        self._precomputed_atoms: Optional[torch.Tensor] = None
    
    def _get_precomputed_atoms(self) -> torch.Tensor:
        """Lazy precompute flat atom values."""
        if self._precomputed_atoms is None:
            from .decoder import precompute_flat_atoms
            self._precomputed_atoms = precompute_flat_atoms(self.atom_table)
        return self._precomputed_atoms
    
    def set_breakpoint(self, condition: Callable[[int, int, float], bool], name: str = "bp") -> Breakpoint:
        """Set a conditional breakpoint.
        
        Args:
            condition: Function (atom_id, coeff_id, residual) -> bool
            name: Breakpoint name for reporting
        
        Returns:
            The created breakpoint
        """
        bp = Breakpoint(condition=condition, name=name)
        self.breakpoints.append(bp)
        return bp
    
    def set_atom_breakpoint(self, atom_id: int, name: Optional[str] = None) -> Breakpoint:
        """Break when a specific atom is used."""
        return self.set_breakpoint(
            condition=lambda a, c, r: a == atom_id,
            name=name or f"atom_{atom_id}"
        )
    
    def set_coeff_breakpoint(self, coeff_id: int, name: Optional[str] = None) -> Breakpoint:
        """Break when a specific coefficient is used."""
        return self.set_breakpoint(
            condition=lambda a, c, r: c == coeff_id,
            name=name or f"coeff_{coeff_id}"
        )
    
    def set_residual_breakpoint(self, threshold: float = 0.0, name: Optional[str] = None) -> Breakpoint:
        """Break when residual exceeds threshold."""
        return self.set_breakpoint(
            condition=lambda a, c, r: abs(r) > threshold,
            name=name or f"residual_gt_{threshold}"
        )
    
    def clear_breakpoints(self):
        """Remove all breakpoints."""
        self.breakpoints.clear()
    
    def resolve_atom_tree(self, atom_id: int, max_depth: int = 10, _depth: int = 0) -> str:
        """Get hierarchical resolution tree for an atom as a string.
        
        Args:
            atom_id: Atom to resolve
            max_depth: Maximum recursion depth
            _depth: Current depth (internal)
        
        Returns:
            Formatted tree string
        """
        indent = "  " * _depth
        if atom_id < self.atom_table.K0:
            val = self.atom_table.base_atoms[atom_id].item()
            return f"{indent}ATOM {atom_id} [L0] = {val:.6f}"
        
        if _depth >= max_depth:
            return f"{indent}ATOM {atom_id} [... max depth]"
        
        d = self.atom_table.atom_defs[atom_id]
        lines = [f"{indent}ATOM {atom_id} [{d.op}]"]
        
        if d.children and d.coeffs:
            for child_id, coeff in zip(d.children, d.coeffs):
                child_tree = self.resolve_atom_tree(child_id, max_depth, _depth + 1)
                lines.append(child_tree)
                if _depth == 0:
                    lines.append(f"{indent}  * {coeff:.6f}")
        
        return "\n".join(lines)
    
    def step(self, prog: ProgramBufferV1, index: int) -> TraceRecord:
        """Execute a single step (one weight) with breakpoint checks.
        
        Args:
            prog: Program buffer
            index: Weight index to execute
        
        Returns:
            TraceRecord with full execution details
        """
        atom_id = int(prog.atom_ids[index])
        coeff_id = int(prog.coeff_ids[index])
        
        flat_atoms = self._get_precomputed_atoms()
        atom_val = flat_atoms[atom_id].item()
        coeff_val = self.coeff_values[coeff_id].item()
        product = atom_val * coeff_val
        
        residual = 0.0
        if prog.has_residual.numel() > 0 and prog.has_residual[index]:
            residual = prog.residuals[index].item()
        
        final_value = product + residual
        
        # Check breakpoints
        bp_hit = None
        for bp in self.breakpoints:
            if bp.check(atom_id, coeff_id, residual):
                bp_hit = bp.name
        
        # Build resolution tree for hierarchical atoms
        tree = None
        if atom_id >= self.atom_table.K0:
            tree = self.resolve_atom_tree(atom_id, max_depth=5)
        
        record = TraceRecord(
            index=index,
            atom_id=atom_id,
            coeff_id=coeff_id,
            atom_value=atom_val,
            coeff_value=coeff_val,
            product=product,
            residual=residual,
            final_value=final_value,
            atom_resolution_tree=tree,
            breakpoint_hit=bp_hit,
        )
        self.trace_log.append(record)
        return record
    
    def run(self, prog: ProgramBufferV1, start: int = 0, end: Optional[int] = None) -> List[TraceRecord]:
        """Run debugger over a range of weights.
        
        Args:
            prog: Program buffer
            start: Start index
            end: End index (exclusive), defaults to prog.N
        
        Returns:
            List of TraceRecords
        """
        if end is None:
            end = prog.N
        
        self.trace_log.clear()
        for i in range(start, min(end, prog.N)):
            self.step(prog, i)
        return self.trace_log


__all__ = [
    "Breakpoint",
    "CoeffTable",
    "HeatmapStats",
    "TraceRecord",
    "WALDebugger",
]
