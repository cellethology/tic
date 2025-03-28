# tic.pseudotime.clustering

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering

class Clustering:
    """
    A class for clustering embeddings using different methods.

    Attributes:
        method (str): Clustering method ("kmeans" or "agg").
        n_clusters (int): Number of clusters to form.
    """
    def __init__(self, method: str = "kmeans", n_clusters: int = 10, random_state: int = 42):
        """
        Initialize the clustering method.

        Args:
            method (str): Clustering method to use (either "kmeans" or "agg").
            n_clusters (int): Number of clusters to form.
        """
        self.method = method
        self.n_clusters = n_clusters
        self.random_state = random_state

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Perform clustering on the embeddings.

        Args:
            embeddings (np.ndarray): The reduced embeddings to cluster.

        Returns:
            np.ndarray: Cluster labels assigned to each sample.
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

        Args:
            embeddings (np.ndarray): The reduced embeddings to cluster.

        Returns:
            np.ndarray: Cluster labels for KMeans.
        """
        model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        return model.fit_predict(embeddings)

    def _agg_cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Perform Agglomerative Clustering on the embeddings.

        Args:
            embeddings (np.ndarray): The reduced embeddings to cluster.

        Returns:
            np.ndarray: Cluster labels for Agglomerative Clustering.
        """
        model = AgglomerativeClustering(n_clusters=self.n_clusters)
        return model.fit_predict(embeddings)