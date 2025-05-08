# file: tic/features/neighbourhood/composition.py
"""Neighbourhood cell‑type composition."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..base import FeatureExtractor
from ..registry import register


@register
class NeighbourComposition(FeatureExtractor):
    """Counts / fractions of each cell_type inside neighbourhood."""

    name = "composition"

    def __init__(self, *, obs_key: str = "cell_type", normalize: bool = True):
        super().__init__(obs_key=obs_key, normalize=normalize)
        self.obs_key = obs_key
        self.normalize = normalize
        self._cats: Sequence[str] | None = None

    def transform(self, adata, *, centre_idx, neighbour_idx):  # noqa: D401
        cats = adata.obs[self.obs_key].astype("category")
        if self._cats is None:
            self._cats = list(cats.cat.categories)
        mapping = {c: i for i, c in enumerate(self._cats)}
        counts = np.zeros(len(self._cats), dtype=float)
        for i in neighbour_idx:
            counts[mapping[cats.iat[i]]] += 1
        if self.normalize and counts.sum() > 0:
            counts /= counts.sum()
        return counts

    @property
    def n_features(self):  # noqa: D401
        return len(self._cats) if self._cats else 0
