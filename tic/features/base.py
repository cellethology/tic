# file: tic/features/base.py
"""Base classes shared by all feature extractors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from anndata import AnnData

__all__ = ["FeatureExtractor"]


class FeatureExtractor(ABC):
    """Abstract base class for any feature extractor plugin."""

    #: unique name used for registry / obsm key
    name: str = "base"

    def __init__(self, **params) -> None:  # noqa: D401
        self.params = params

    # ------------------------------------------------------------------
    @abstractmethod
    def transform(
        self,
        adata: AnnData,
        *,
        centre_idx: int,
        neighbour_idx: Sequence[int],
    ) -> np.ndarray:  # noqa: D401
        """Produce a 1-D feature vector for the given neighbourhood."""

    # ------------------------------------------------------------------
    @property
    def n_features(self) -> int:  # noqa: D401
        """Feature dimension (requires prior call or fixed length)."""
        raise NotImplementedError