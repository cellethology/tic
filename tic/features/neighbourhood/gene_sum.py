# tic/features/neighbourhood/gene_sum.py
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from anndata import AnnData
from scipy import sparse as sp

from ..base import FeatureExtractor
from ..registry import register


@register
class NeighbourGeneSum(FeatureExtractor):
    """Sum of expression across *all* neighbour cells **including** centre."""

    name = "neighbor_gene_sum"

    def __init__(self) -> None:
        super().__init__()
        self._n: int = 0
        self._feature_names: List[str] = []

    # ------------------------------------------------------------------
    def transform(  # noqa: D401
        self,
        adata: AnnData,
        *,
        centre_idx: int,  # noqa: ARG002
        neighbour_idx: Sequence[int],
    ) -> np.ndarray:
        if self._n == 0:
            self._n = adata.n_vars
            self._feature_names = [f"{self.name}:{i}" for i in list(adata.var_names)]

        X = adata.X[neighbour_idx]
        if sp.issparse(X):
            return np.asarray(X.sum(axis=0)).ravel()
        return np.asarray(X.sum(axis=0), dtype=float)

    # ------------------------------------------------------------------
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