# tic/features/cell/centre_gene.py
"""Raw gene‑expression vector of the *centre* cell."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from anndata import AnnData
from scipy import sparse as sp

from ..base import FeatureExtractor
from ..registry import register


@register
class CentreGene(FeatureExtractor):
    """
    Return the *dense* raw expression vector of the centre cell.

    Shape: ``(n_genes,)``.
    """

    name = "centre_gene"

    def __init__(self) -> None:
        super().__init__()
        self._n: int = 0
        self._feature_names: List[str] = []

    # ------------------------------------------------------------------ core
    def transform(  # noqa: D401
        self,
        adata: AnnData,
        *,
        centre_idx: int,
        neighbour_idx: Sequence[int],  # noqa: ARG002  未使用
    ) -> np.ndarray:
        if self._n == 0:
            self._n = adata.n_vars
            self._feature_names = [f"{self.name}:{i}" for i in list(adata.var_names)]

        X_row = adata.X[centre_idx]
        if sp.issparse(X_row):
            X_row = X_row.toarray()
        return np.asarray(X_row, dtype=float).ravel()

    # ------------------------------------------------------------------ metadata
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