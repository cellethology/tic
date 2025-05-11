# tic/features/neighbourhood/celltype_gene_count.py
from __future__ import annotations

from typing import List, Literal, Sequence

import numpy as np
from anndata import AnnData
from scipy import sparse as sp

from ..base import FeatureExtractor
from ..registry import register
from ...data.utils import get_biomarkers, get_cell_types


@register
class CelltypeGeneCount(FeatureExtractor):
    r"""
    Flattened vector of shape ``(#cell_types × #genes,)``.

    Parameters
    ----------
    agg : {"count", "mean", "sum"}, default "count"
        Aggregation across neighbours.
    threshold : float, default 0.0
        Used only when ``agg="count"`` — binary indicator `expr > threshold`.
    """

    name = "celltype_gene_count"

    def __init__(
        self,
        *,
        agg: Literal["count", "mean", "sum"] = "count",
        threshold: float = 0.0,
    ) -> None:
        super().__init__(agg=agg, threshold=threshold)
        self.agg: Literal["count", "mean", "sum"] = agg
        self.threshold: float = threshold

        # caches
        self._cell_types: List[str] | None = None
        self._genes: List[str] | None = None
        self._n: int = 0
        self._feature_names: List[str] = []

    # ------------------------------------------------------------------ core
    def transform(  # noqa: D401
        self,
        adata: AnnData,
        *,
        centre_idx: int,  # noqa: ARG002  中心索引目前不用
        neighbour_idx: Sequence[int],
    ) -> np.ndarray:
        if self._cell_types is None:
            self._cell_types = get_cell_types(adata)
        if self._genes is None:
            self._genes = get_biomarkers(adata)
        cats, genes = self._cell_types, self._genes
        n_ct, n_g = len(cats), len(genes)

        if self._n == 0:  # 第一次调用
            self._n = n_ct * n_g
            self._feature_names = [
                f"{self.name}:{self.agg}_{ct}_{g}" for ct in cats for g in genes
            ]

        # --- gather expression & labels
        X = adata.X[neighbour_idx]
        X = np.asarray(X.todense() if sp.issparse(X) else X, dtype=np.float32)
        labels = adata.obs["cell_type"].astype(str).to_numpy()[neighbour_idx]

        if self.agg == "count":
            X = (X > self.threshold).astype(np.float32)

        # --- accumulate
        out = np.zeros((n_ct, n_g), dtype=np.float32)
        ct_to_row = {ct: i for i, ct in enumerate(cats)}
        row_ids = np.vectorize(ct_to_row.get)(labels)
        for r in range(n_ct):
            rows = X[row_ids == r]
            if rows.size:
                if self.agg == "mean":
                    out[r] = rows.mean(axis=0)
                else:  # "sum" or "count"
                    out[r] = rows.sum(axis=0)
        return out.ravel()

    # ------------------------------------------------------------------ metadata
    @property
    def n_features(self) -> int:  # noqa: D401
        return self._n

    def feature_names(self, adata: AnnData) -> List[str]:  # type: ignore[override]
        if not self._feature_names:
            # lazy init for edge‑cases
            _ = self.transform(adata, centre_idx=0, neighbour_idx=[0])
        return self._feature_names

    def feature_meta(self, adata: AnnData):  # type: ignore[override]
        return {
            "extractor": [self.name] * self.n_features,
            "agg": [self.agg] * self.n_features,
            "cell_type": [ct for ct in self._cell_types for _ in self._genes],  # type: ignore[arg-type]
            "gene": self._genes * len(self._cell_types),  # type: ignore[arg-type]
        }