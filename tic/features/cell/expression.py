"""Raw gene expression of the centre cell."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..base import FeatureExtractor
from ..registry import register


@register
class CentreGene(FeatureExtractor):
    """Raw expression vector of the centre cell (dense)."""

    name = "centre_gene"
    def __init__(self):
        super().__init__()
        self._n = 0

    def transform(self, adata, *, centre_idx, neighbour_idx):  # noqa: D401
        X = adata.X
        if self._n == 0:
            self._n = adata.n_vars
        if sp.issparse(X):
            return np.asarray(X[centre_idx].toarray()).ravel()
        return np.asarray(X[centre_idx], dtype=float)

    @property
    def n_features(self):  # noqa: D401
        return self._n

