"""
Module: tic.pseudotime.dimensionality_reduction

Provides a DimensionalityReduction class to reduce the dimensions of
embeddings using PCA or UMAP.
"""

import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP


class DimensionalityReduction:
    """
    Perform dimensionality reduction on embeddings.

    Attributes
    ----------
    method : str
        Dimensionality reduction method ("PCA" or "UMAP").
    n_components : int
        Number of components to reduce to.
    random_state : int
        Random state for reproducibility.
    """

    def __init__(self, method: str = "PCA", n_components: int = 2, random_state: int = 42) -> None:
        """
        Initialize the dimensionality reduction method.

        Parameters
        ----------
        method : str, optional
            Reduction method ("PCA" or "UMAP"), by default "PCA".
        n_components : int, optional
            Number of components to reduce to, by default 2.
        random_state : int, optional
            Random state for reproducibility, by default 42.
        """
        self.method = method
        self.n_components = n_components
        self.random_state = random_state

    def reduce(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Perform dimensionality reduction on the embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            The input embeddings to reduce.

        Returns
        -------
        np.ndarray
            Reduced embeddings.

        Raises
        ------
        ValueError
            If an invalid reduction method is specified.
        """
        if self.method == "PCA":
            return self._pca_reduce(embeddings)
        elif self.method == "UMAP":
            return self._umap_reduce(embeddings)
        else:
            raise ValueError(f"Invalid dimensionality reduction method: {self.method}")

    def _pca_reduce(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduce embeddings using PCA.

        Parameters
        ----------
        embeddings : np.ndarray
            The input embeddings to reduce.

        Returns
        -------
        np.ndarray
            Reduced embeddings from PCA.
        """
        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        return pca.fit_transform(embeddings)

    def _umap_reduce(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduce embeddings using UMAP.

        Parameters
        ----------
        embeddings : np.ndarray
            The input embeddings to reduce.

        Returns
        -------
        np.ndarray
            Reduced embeddings from UMAP.
        """
        umap = UMAP(n_components=self.n_components, random_state=self.random_state)
        return umap.fit_transform(embeddings)