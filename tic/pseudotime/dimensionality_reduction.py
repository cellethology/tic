# tic.pseudotime.dimensionality_reduction

import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP

class DimensionalityReduction:
    """
    A class to perform dimensionality reduction on the embeddings.

    Attributes:
        method (str): Method for dimensionality reduction ("PCA" or "UMAP").
        n_components (int): Number of components to reduce to.
        random_state (int): Random state for reproducibility.
    """
    def __init__(self, method: str = "PCA", n_components: int = 2, random_state: int = 42):
        """
        Initialize the dimensionality reduction method.

        Args:
            method (str): Method for dimensionality reduction ("PCA" or "UMAP").
            n_components (int): Number of components to reduce to.
            random_state (int): Random state for reproducibility.
        """
        self.method = method
        self.n_components = n_components
        self.random_state = random_state

    def reduce(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Perform dimensionality reduction on the embeddings.

        Args:
            embeddings (np.ndarray): The input embeddings to reduce.

        Returns:
            np.ndarray: Reduced embeddings.
        """
        if self.method == "PCA":
            return self._pca_reduce(embeddings)
        elif self.method == "UMAP":
            return self._umap_reduce(embeddings)
        else:
            raise ValueError(f"Invalid dimensionality reduction method: {self.method}")

    def _pca_reduce(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Perform PCA reduction on the embeddings.

        Args:
            embeddings (np.ndarray): The input embeddings to reduce.

        Returns:
            np.ndarray: Reduced embeddings from PCA.
        """
        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        return pca.fit_transform(embeddings)

    def _umap_reduce(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Perform UMAP reduction on the embeddings.

        Args:
            embeddings (np.ndarray): The input embeddings to reduce.

        Returns:
            np.ndarray: Reduced embeddings from UMAP.
        """
        umap = UMAP(n_components=self.n_components, random_state=self.random_state)
        return umap.fit_transform(embeddings)