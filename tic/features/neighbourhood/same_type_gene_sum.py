# file: tic/features/neighbourhood/same_type_gene_sum.py
"""Sum expression of neighbours with same cell type as centre."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp

from ..base import FeatureExtractor
from ..registry import register


@register
class SameTypeGeneSum(FeatureExtractor):
    """Sum of expression of neighbours whose cell type == centre's."""

    name = "same_type_gene_sum"

    def __init__(self, *, obs_key: str = "cell_type"):
        super().__init__(obs_key=obs_key)
        self.obs_key = obs_key

    def transform(self, adata, *, centre_idx, neighbour_idx: Sequence[int]):  # noqa: D401
        centre_type = adata.obs[self.obs_key].iat[centre_idx]
        same = [i for i in neighbour_idx if adata.obs[self.obs_key].iat[i] == centre_type]
        X = adata.X
        if not same:
            return np.zeros(adata.n_vars, dtype=float)
        if sp.issparse(X):
            return np.asarray(X[same].sum(axis=0)).ravel()
        return np.asarray(X[same].sum(axis=0), dtype=float)

    @property
    def n_features(self):  # noqa: D401
        return self._cached_n

    _cached_n = 0

@register
class SameTypeGeneAverage(FeatureExtractor):
    """Average expression of neighbours with same cell type as centre."""

    name = "same_type_gene_average"
    
    def __init__(self, *, obs_key: str = "cell_type"):
        super().__init__(obs_key=obs_key)
        self.obs_key = obs_key

    def transform(self, adata, *, centre_idx, neighbour_idx: Sequence[int]):  # noqa: D401
        centre_type = adata.obs[self.obs_key].iat[centre_idx]
        same = [i for i in neighbour_idx if adata.obs[self.obs_key].iat[i] == centre_type]
        X = adata.X
        if not same:
            return np.zeros(adata.n_vars, dtype=float)
        if sp.issparse(X):
            return np.asarray(X[same].mean(axis=0)).ravel()
        return np.asarray(X[same].mean(axis=0), dtype=float)
