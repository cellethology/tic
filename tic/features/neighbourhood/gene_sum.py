# file: tic/features/neighbourhood/gene_sum.py
"""Total gene expression of neighbours."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp

from ..base import FeatureExtractor
from ..registry import register


@register
class NeighbourGeneSum(FeatureExtractor):
    """Sum of expression across all neighbour cells (including centre)."""

    name = "neighbor_gene_sum"

    def transform(self, adata, *, centre_idx, neighbour_idx: Sequence[int]):  # noqa: D401
        X = adata.X
        if sp.issparse(X):
            return np.asarray(X[neighbour_idx].sum(axis=0)).ravel()
        return np.asarray(X[neighbour_idx].sum(axis=0), dtype=float)

    @property
    def n_features(self):  # noqa: D401
        return self._cached_n

    _cached_n = 0
