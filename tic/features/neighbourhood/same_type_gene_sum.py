# tic/features/neighbourhood/same_type_gene_sum.py
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from anndata import AnnData
from scipy import sparse as sp

from ..base import FeatureExtractor
from ..registry import register


class _SameTypeBase(FeatureExtractor):
    """Shared logic for *sum* / *average* variants."""

    reduce: str  # "sum" or "mean"
    name: str

    def __init__(self, *, obs_key: str = "cell_type") -> None:
        super().__init__(obs_key=obs_key)
        self.obs_key: str = obs_key
        self._n: int = 0
        self._feature_names: List[str] = []

    # -----------------------------------------------
    def _reduce(self, X_subset) -> np.ndarray:
        if self.reduce == "sum":
            return X_subset.sum(axis=0)
        # mean
        return X_subset.mean(axis=0)

    # -----------------------------------------------
    def transform(  # noqa: D401
        self,
        adata: AnnData,
        *,
        centre_idx: int,
        neighbour_idx: Sequence[int],
    ) -> np.ndarray:
        if self._n == 0:
            self._n = adata.n_vars
            self._feature_names = list(adata.var_names)

        centre_type = adata.obs[self.obs_key].iat[centre_idx]
        same = [i for i in neighbour_idx if adata.obs[self.obs_key].iat[i] == centre_type]
        if not same:
            return np.zeros(self._n, dtype=float)

        X = adata.X[same]
        if sp.issparse(X):
            X = X.todense()

        return np.asarray(self._reduce(X), dtype=float).ravel()

    # -----------------------------------------------
    @property
    def n_features(self) -> int:  # noqa: D401
        return self._n

    def feature_names(self, adata: AnnData) -> List[str]:  # type: ignore[override]
        return self._feature_names

    def feature_meta(self, adata: AnnData) -> Dict[str, list]:  # type: ignore[override]
        return {
            "extractor": [self.name] * self.n_features,
            "gene": self._feature_names,
        }


@register
class SameTypeGeneSum(_SameTypeBase):
    """Sum of neighbour expression with the **same** cell‑type as centre."""
    name = "same_type_gene_sum"
    reduce = "sum"


@register
class SameTypeGeneAverage(_SameTypeBase):
    """Average neighbour expression with the **same** cell‑type as centre."""
    name = "same_type_gene_average"
    reduce = "mean"