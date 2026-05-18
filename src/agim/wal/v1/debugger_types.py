from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class Breakpoint:
    """A conditional breakpoint in WAL program execution."""
    condition: Callable[[int, int, float], bool]  # (atom_id, coeff_id, residual) -> bool
    name: str
    hit_count: int = 0
    
    def check(self, atom_id: int, coeff_id: int, residual: float) -> bool:
        """Check if breakpoint triggers for given weight."""
        if self.condition(atom_id, coeff_id, residual):
            self.hit_count += 1
            return True
        return False


@dataclass
class TraceRecord:
    """Single step trace record."""
    index: int
    atom_id: int
    coeff_id: int
    atom_value: float
    coeff_value: float
    product: float
    residual: float
    final_value: float
    atom_resolution_tree: Optional[str] = None
    breakpoint_hit: Optional[str] = None


@dataclass
class HeatmapStats:
    """Program heatmap statistics."""
    total_weights: int
    atom_frequencies: Dict[int, int] = field(default_factory=dict)
    coeff_frequencies: Dict[int, int] = field(default_factory=dict)
    top_atoms: List[Tuple[int, int, float]] = field(default_factory=list)  # (id, count, pct)
    top_coeffs: List[Tuple[int, int, float]] = field(default_factory=list)
    residual_count: int = 0
    residual_pct: float = 0.0
    atom_entropy: float = 0.0
    coeff_entropy: float = 0.0
