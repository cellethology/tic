# file: tic/features/registry.py
"""Global registry for feature extractors."""
from __future__ import annotations

from typing import Dict, Type

from .base import FeatureExtractor

__all__ = ["FeatureRegistry", "register"]


class _Registry:
    def __init__(self) -> None:  # noqa: D401
        self._map: Dict[str, Type[FeatureExtractor]] = {}

    # ------------------------------------------------------------------
    def register(self, cls: Type[FeatureExtractor]) -> Type[FeatureExtractor]:  # noqa: D401
        if cls.name in self._map:
            raise KeyError(f"Feature name '{cls.name}' already registered.")
        self._map[cls.name] = cls
        return cls

    def get(self, name: str) -> Type[FeatureExtractor]:  # noqa: D401
        if name not in self._map:
            raise KeyError(f"Unknown feature '{name}'. Registered: {list(self._map)}")
        return self._map[name]

    def names(self) -> list[str]:  # noqa: D401
        return list(self._map)


FeatureRegistry = _Registry()


def register(cls: Type[FeatureExtractor]) -> Type[FeatureExtractor]:  # noqa: D401
    """Decorator to register extractor classes."""
    return FeatureRegistry.register(cls)
