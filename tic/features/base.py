# tic/features/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence

import numpy as np
from anndata import AnnData

__all__ = ["FeatureExtractor"]


class FeatureExtractor(ABC):
    """
    Abstract base class for every feature‑extractor plugin.
    """

    #: unique key for registry / `.obsm`
    name: str = "base"

    def __init__(self, **params) -> None:  # noqa: D401
        self.params = params

    # ------------------------------------------------------------------ user API
    @abstractmethod
    def transform(
        self,
        adata: AnnData,
        *,
        centre_idx: int,
        neighbour_idx: Sequence[int],
    ) -> np.ndarray:
        """Return a **1‑D** feature vector for the given neighbourhood."""

    # ------------------------------------------------------------------ metadata
    @property
    def n_features(self) -> int:  # noqa: D401
        """Feature dimension (must be set by subclass)."""
        raise NotImplementedError

    def feature_names(self, adata: AnnData) -> List[str]:  # type: ignore[override]
        """
        Human-readable column names, length == `n_features`.

        Default fallback: ``f"{self.name}:{i}"``.
        """
        return [f"{self.name}:{i}" for i in range(self.n_features)]

    def feature_meta(self, adata: AnnData) -> Dict[str, list] | None:  # noqa: D401
        """
        Optional per-feature metadata (dict of column-name → list).

        Return *None* if not needed.
        """
        return None