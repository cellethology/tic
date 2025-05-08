# file: tic/pseudotime/pp/clustering.py
"""Clustering wrapper providing a unified interface over multiple algorithms."""
from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.cluster import (
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    KMeans,
    MeanShift,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture

__all__ = ["Clustering"]

# Supported clustering methods
ClusterMethod = Literal[
    "kmeans",
    "agg",
    "dbscan",
    "mean_shift",
    "spectral",
    "birch",
    "gmm",
]


class Clustering:
    """Unified interface over multiple clustering algorithms.

    Parameters
    ----------
    method
        Which clustering method to use. Supported options:
        - "kmeans"
        - "agg" (Agglomerative)
        - "dbscan"
        - "mean_shift"
        - "spectral"
        - "birch"
        - "gmm" (Gaussian Mixture)
    n_clusters
        Number of clusters (if applicable).
    random_state
        Random seed (if applicable).
    eps
        The maximum distance between two samples for DBSCAN (only if method=="dbscan").
    """

    def __init__(
        self,
        method: ClusterMethod = "kmeans",
        n_clusters: int = 10,
        random_state: int = 42,
        eps: float = 0.5,
    ) -> None:
        self.method = method.lower()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.eps = eps

    def _create_model(self):
        """Instantiate and return the clustering model corresponding to self.method."""
        if self.method == "kmeans":
            return KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
            )
        if self.method == "agg":
            return AgglomerativeClustering(
                n_clusters=self.n_clusters,
            )
        if self.method == "dbscan":
            return DBSCAN(
                eps=self.eps,
            )
        if self.method == "mean_shift":
            return MeanShift()
        if self.method == "spectral":
            return SpectralClustering(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                assign_labels="discretize",
            )
        if self.method == "birch":
            return Birch(
                n_clusters=self.n_clusters,
            )
        if self.method == "gmm":
            return GaussianMixture(
                n_components=self.n_clusters,
                random_state=self.random_state,
            )
        raise ValueError(f"Unsupported clustering method: {self.method!r}")

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit the chosen clustering algorithm to the data and return labels.

        Parameters
        ----------
        embeddings
            2D array of shape (n_samples, n_features).

        Returns
        -------
        labels
            Array of shape (n_samples,) with cluster assignments.
        """
        model = self._create_model()
        # Some models (e.g., GMM) use `fit_predict`, others require two steps
        if hasattr(model, "fit_predict"):
            return model.fit_predict(embeddings)  # type: ignore
        # e.g., GaussianMixture
        labels = model.fit(embeddings).predict(embeddings)  # type: ignore
        return labels