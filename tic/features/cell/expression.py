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

    def transform(self, adata, *, centre_idx, neighbour_idx):  # noqa: D401
        X = adata.X
        if sp.issparse(X):
            return np.asarray(X[centre_idx].toarray()).ravel()
        return np.asarray(X[centre_idx], dtype=float)

    @property
    def n_features(self):  # noqa: D401
        return self._cached_n

    # we can infer from adata later; here placeholder
    _cached_n = 0
