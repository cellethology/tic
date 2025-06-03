# file: tic/pseudotime/pp/dimensionality.py
"""Dimensionality-reduction wrappers (PCA, KernelPCA, t-SNE, UMAP, MDS, Isomap, LLE)."""
from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding
from umap import UMAP

__all__ = ["DimensionalityReduction"]

# Supported methods
ReductionMethod = Literal[
    "pca",
    "kernel_pca",
    "tsne",
    "umap",
    "mds",
    "isomap",
    "lle",
]


class DimensionalityReduction:
    """Unified interface over multiple dimensionality-reduction algorithms.

    Parameters
    ----------
    method
        Which algorithm to use. Supported options are
        "pca", "kernel_pca", "tsne", "umap", "mds", "isomap", "lle".
    n_components
        Target number of dimensions.
    random_state
        Random seed (if applicable).
    **kwargs
        Extra keyword arguments passed to the underlying estimator.
    """

    def __init__(
        self,
        method: ReductionMethod = "pca",
        n_components: int = 2,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        self.method = method.lower()
        self.n_components = n_components
        self.random_state = random_state
        self.kwargs = kwargs

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the chosen model and return the transformed data.

        Parameters
        ----------
        X
            Input data of shape (n_samples, n_features).

        Returns
        -------
        X_reduced
            Transformed data of shape (n_samples, n_components).
        """
        model = self._create_model()
        return model.fit_transform(X)

    def _create_model(self):
        """Instantiate and return the underlying reduction estimator."""
        m = self.method
        if m == "pca":
            return PCA(n_components=self.n_components, random_state=self.random_state, **self.kwargs)
        if m == "kernel_pca":
            return KernelPCA(n_components=self.n_components, random_state=self.random_state, **self.kwargs)
        if m == "tsne":
            return TSNE(n_components=self.n_components, random_state=self.random_state, **self.kwargs)
        if m == "umap":
            return UMAP(n_components=self.n_components, random_state=self.random_state, **self.kwargs)
        if m == "mds":
            return MDS(n_components=self.n_components, random_state=self.random_state, **self.kwargs)
        if m == "isomap":
            return Isomap(n_components=self.n_components, **self.kwargs)
        if m == "lle":
            return LocallyLinearEmbedding(
                n_components=self.n_components,
                random_state=self.random_state,
                **self.kwargs,
            )
        raise ValueError(f"Unsupported reduction method: {self.method!r}")