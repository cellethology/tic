# file: tic/features/neighbourhood/celltype_gene_count.py
"""Count the number of neighbour cells of each cell type that express each gene."""
from __future__ import annotations

from typing import Sequence
import numpy as np
from anndata import AnnData

from ..base import FeatureExtractor
from ..registry import register
from ...data.utils import get_cell_types, get_biomarkers

@register
class CelltypeGeneCount(FeatureExtractor):
    """
    Count the number of neighbour cells of each cell type that express each gene.

    Output is a flat vector of length (#cell_types * #genes), ordered by cell_type
    then gene, i.e. [count(ct0, g0), count(ct0, g1), ..., count(ct1, g0), ...].

    Configuration parameters: None.
    """
    name = "celltype_gene_count"

    def transform(
        self,
        adata: AnnData,
        *,
        centre_idx: int,
        neighbour_idx: Sequence[int],
    ) -> np.ndarray:
        cell_types = get_cell_types(adata)
        genes      = get_biomarkers(adata)

        expr = np.asarray(adata.X[neighbour_idx, :])
        ct_labels = adata.obs.iloc[neighbour_idx]["cell_type"].astype(str).to_numpy()
        
        # index mapping
        gene_to_idx = {g: i for i, g in enumerate(genes)}
        
        counts = np.zeros((len(cell_types), len(genes)), dtype=int)
        for i, ct in enumerate(cell_types):
            mask = (ct_labels == ct)
            if not mask.any():
                continue
            sub = expr[mask, :]
            for gene, j in gene_to_idx.items():
                counts[i, j] = int((sub[:, j] > 0).sum())
        
        return counts.ravel()

    @property
    def n_features(self):  # noqa: D401
        return self._cached_n

    _cached_n = 0