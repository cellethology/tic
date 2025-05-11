# tic/features/neighbourhood/composition.py
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from anndata import AnnData

from ..base import FeatureExtractor
from ..registry import register


@register
class NeighbourComposition(FeatureExtractor):
    """
    Counts (or fractions) of each category in the neighbour set.

    Parameters
    ----------
    obs_key : str, default "cell_type"
        `.obs` column holding the categorical label.
    normalize : bool, default True
        If *True* → return fractions; otherwise raw counts.
    """

    name = "composition"

    def __init__(self, *, obs_key: str = "cell_type", normalize: bool = True) -> None:
        super().__init__(obs_key=obs_key, normalize=normalize)
        self.obs_key: str = obs_key
        self.normalize: bool = normalize

        self._cats: List[str] | None = None
        self._n: int = 0
        self._feature_names: List[str] = []

    # ------------------------------------------------------------------ core
    def transform(  # noqa: D401
        self,
        adata: AnnData,
        *,
        centre_idx: int,  # noqa: ARG002
        neighbour_idx: Sequence[int],
    ) -> np.ndarray:
        col = adata.obs[self.obs_key].astype("category")
        if self._cats is None:
            self._cats = list(col.cat.categories)
            self._n = len(self._cats)
            self._feature_names = [f"{self.name}:{c}" for c in self._cats]

        mapping = {c: i for i, c in enumerate(self._cats)}
        counts = np.zeros(self._n, dtype=float)
        for i in neighbour_idx:
            counts[mapping[col.iat[i]]] += 1

        if self.normalize and counts.sum() > 0:
            counts /= counts.sum()

        return counts

    # ------------------------------------------------------------------ metadata
    @property
    def n_features(self) -> int:  # noqa: D401
        return self._n

    def feature_names(self, adata: AnnData) -> List[str]:  # type: ignore[override]
        return self._feature_names

    def feature_meta(self, adata: AnnData) -> Dict[str, list]:  # type: ignore[override]
        return {
            "extractor": [self.name] * self.n_features,
            "category": self._cats,
        }