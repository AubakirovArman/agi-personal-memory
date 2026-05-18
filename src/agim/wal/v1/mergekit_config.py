from __future__ import annotations

from typing import List, Optional


class MergeConfig:
    """Configuration for a model merge operation."""
    
    def __init__(
        self,
        method: str = "soup",  # soup, slerp, ties, linear
        weights: Optional[List[float]] = None,
        soup_method: str = "mean",  # mean, majority, weighted
        density: float = 1.0,  # For TIES: fraction of params to keep
        epsilon: float = 1e-6,  # For SLERP
    ):
        self.method = method
        self.weights = weights
        self.soup_method = soup_method
        self.density = density
        self.epsilon = epsilon
