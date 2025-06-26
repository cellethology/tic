"""tic.features.composite.centre_gene_composition
================================================

Feature‑extractor that **concatenates** the raw gene‑expression vector of the
*centre* cell with the neighbourhood cell‑type composition (counts or
fractions).  It re‑uses two existing extractors – :class:`~tic.features.cell.centre_gene.CentreGene`
for the gene expression and :class:`~tic.features.neighbourhood.composition.NeighbourComposition`
for the composition – and simply joins their outputs into a single 1‑D feature
vector.

Example
-------
>>> from tic.features.composite.centre_gene_composition import CentreGeneComposition
>>> fx = CentreGeneComposition(obs_key="cell_type", normalize=True)
>>> v = fx.transform(adata, centre_idx=42, neighbour_idx=k_idx)
>>> v.shape  # (n_genes + n_categories,)

"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from anndata import AnnData

from ..base import FeatureExtractor
from ..registry import register
from ..cell.expression import CentreGene
from ..neighbourhood.composition import NeighbourComposition

__all__ = ["CentreGeneComposition"]


@register
class CentreGeneComposition(FeatureExtractor):
    """Centre‑gene expression **plus** neighbour composition (concatenated)."""

    name = "centre_gene_comp"

    # pylint: disable=too-many-instance-attributes
    def __init__(self, *, obs_key: str = "cell_type", normalize: bool = True):
        super().__init__(obs_key=obs_key, normalize=normalize)
        # Re‑use existing extractors
        self._centre_gene = CentreGene()
        self._composition = NeighbourComposition(obs_key=obs_key, normalize=normalize)

        self._n: int = 0
        self._feature_names: List[str] = []
        self._meta_cached: Dict[str, list] | None = None

    # ------------------------------------------------------------------ core
    def transform(  # noqa: D401
        self,
        adata: AnnData,
        *,
        centre_idx: int,
        neighbour_idx: Sequence[int],
    ) -> np.ndarray:
        # Compute sub‑features
        g_vec = self._centre_gene.transform(
            adata, centre_idx=centre_idx, neighbour_idx=neighbour_idx
        )
        c_vec = self._composition.transform(
            adata, centre_idx=centre_idx, neighbour_idx=neighbour_idx
        )

        if self._n == 0:
            # First call → cache metadata & dimensions
            self._n = g_vec.size + c_vec.size
            self._feature_names = (
                self._centre_gene.feature_names(adata) + self._composition.feature_names(adata)
            )

        return np.concatenate([g_vec, c_vec], dtype=float)

    # ------------------------------------------------------------------ metadata
    @property
    def n_features(self) -> int:  # noqa: D401
        return self._n

    def feature_names(self, adata: AnnData) -> List[str]:  # type: ignore[override]
        # If not initialised (called before *transform*), fall back to sub‑extractors
        if not self._feature_names:
            return (
                self._centre_gene.feature_names(adata) + self._composition.feature_names(adata)
            )
        return self._feature_names

    def feature_meta(self, adata: AnnData) -> Dict[str, list] | None:  # type: ignore[override]
        """Merge metadata from both sub‑extractors (column‑wise alignment)."""
        if self._meta_cached is not None:
            return self._meta_cached

        meta_g = self._centre_gene.feature_meta(adata) or {}
        meta_c = self._composition.feature_meta(adata) or {}

        # Union of keys, fill missing values with empty strings for alignment
        keys = set(meta_g).union(meta_c)
        merged: Dict[str, list] = {}
        for key in keys:
            merged[key] = meta_g.get(key, [""] * self._centre_gene.n_features) + meta_c.get(
                key, [""] * self._composition.n_features
            )

        self._meta_cached = merged
        return merged
