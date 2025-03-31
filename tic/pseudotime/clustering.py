"""
Module: tic.pseudotime.clustering

Provides a Clustering class to cluster embeddings using KMeans or
Agglomerative Clustering.
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering


class Clustering:
    """
    Cluster embeddings using different methods.

    Attributes
    ----------
    method : str
        Clustering method ("kmeans" or "agg").
    n_clusters : int
        Number of clusters to form.
    random_state : int
        Random state for reproducibility.
    """

    def __init__(self, method: str = "kmeans", n_clusters: int = 10, random_state: int = 42) -> None:
        """
        Initialize the clustering method.

        Parameters
        ----------
        method : str, optional
            Clustering method to use (either "kmeans" or "agg"), by default "kmeans".
        n_clusters : int, optional
            Number of clusters to form, by default 10.
        random_state : int, optional
            Random state for reproducibility, by default 42.
        """
        self.method = method
        self.n_clusters = n_clusters
        self.random_state = random_state

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Perform clustering on the embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            The reduced embeddings to cluster.

        Returns
        -------
        np.ndarray
            Cluster labels assigned to each sample.

        Raises
        ------
        ValueError
            If an unsupported clustering method is specified.
        """
        if self.method == "kmeans":
            return self._kmeans_cluster(embeddings)
        elif self.method == "agg":
            return self._agg_cluster(embeddings)
        else:
            raise ValueError(f"Unsupported clustering method: {self.method}")

    def _kmeans_cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Perform KMeans clustering on the embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            The reduced embeddings to cluster.

        Returns
        -------
        np.ndarray
            Cluster labels from KMeans.
        """
        model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        return model.fit_predict(embeddings)

    def _agg_cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Perform Agglomerative Clustering on the embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            The reduced embeddings to cluster.

        Returns
        -------
        np.ndarray
            Cluster labels from Agglomerative Clustering.
        """
        model = AgglomerativeClustering(n_clusters=self.n_clusters)
        return model.fit_predict(embeddings)