from __future__ import annotations

from typing import Any, Dict, Optional


class WALModelCard:
    """Metadata card for a WAL-encoded model."""
    
    def __init__(
        self,
        base_model: str,
        wal_version: str = "1.0",
        encoder_config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        adapters: Optional[list] = None,
    ):
        self.base_model = base_model
        self.wal_version = wal_version
        self.encoder_config = encoder_config or {}
        self.metrics = metrics or {}
        self.adapters = adapters or []
    
    def to_dict(self) -> dict:
        return {
            "base_model": self.base_model,
            "wal_version": self.wal_version,
            "encoder_config": self.encoder_config,
            "metrics": self.metrics,
            "adapters": self.adapters,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "WALModelCard":
        return cls(
            base_model=d.get("base_model", ""),
            wal_version=d.get("wal_version", "1.0"),
            encoder_config=d.get("encoder_config", {}),
            metrics=d.get("metrics", {}),
            adapters=d.get("adapters", []),
        )
