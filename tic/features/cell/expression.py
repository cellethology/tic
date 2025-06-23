# tic/features/cell/centre_gene.py
"""Raw gene-expression vector of the *centre* cell."""
from __future__ import annotations

from typing import Dict, List, Sequence, Optional

import numpy as np
from anndata import AnnData
import scanpy
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

@register
class CentreGeneHVG(FeatureExtractor):
    """
    Return the dense raw expression vector of the centre cell after HVG filtering.

    First filters the data to top highly variable genes (default 2000),
    then returns the expression vector.

    Shape: ``(n_genes,)``.
    """

    name = "centre_gene_hvg"

    def __init__(self, n_top_genes: Optional[int] = 2000) -> None:
        super().__init__()
        self._n: int = 0
        self._feature_names: List[str] = []
        self.n_top_genes = n_top_genes
        self._hvg_indices: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ core
    def transform(  # noqa: D401
        self,
        adata: AnnData,
        *,
        centre_idx: int,
        neighbour_idx: Sequence[int],  # noqa: ARG002  未使用
    ) -> np.ndarray:
        if self._n == 0:
            # First run - identify HVGs
            if 'highly_variable' not in adata.var:
                scanpy.pp.highly_variable_genes(adata, n_top_genes=self.n_top_genes)
            
            self._hvg_indices = np.where(adata.var['highly_variable'])[0]
            self._n = len(self._hvg_indices)
            self._feature_names = [f"{self.name}:{i}" for i in list(adata.var_names[self._hvg_indices])]

        X_row = adata.X[centre_idx]
        if sp.issparse(X_row):
            X_row = X_row.toarray()
        
        # Select only HVG columns
        X_row = np.asarray(X_row, dtype=float).ravel()[self._hvg_indices]
        return X_row

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