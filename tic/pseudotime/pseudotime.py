"""
Module: tic.pseudotime.pseudotime

Provides an abstract base class for pseudotime inference methods and an
implementation using the Slingshot algorithm.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from pyslingshot import Slingshot
from anndata import AnnData
import matplotlib.pyplot as plt


class PseudotimeMethod(ABC):
    """
    Abstract base class for pseudotime inference methods.
    """

    @abstractmethod
    def analyze(
        self, labels: np.ndarray, umap_embeddings: np.ndarray, output_dir: str
    ) -> np.ndarray:
        """
        Perform pseudotime analysis on embeddings and cluster labels.

        Parameters
        ----------
        labels : np.ndarray
            Cluster labels for the embeddings.
        umap_embeddings : np.ndarray
            UMAP-reduced embeddings.
        output_dir : str
            Directory to save results and plots.

        Returns
        -------
        np.ndarray
            Pseudotime values for each cell.
        """
        pass


class SlingshotMethod(PseudotimeMethod):
    """
    Perform pseudotime analysis using the Slingshot algorithm.

    Attributes
    ----------
    start_node : Optional[int]
        The starting node for the analysis. If None, defaults to cluster 0.
    """

    def __init__(self, start_node: Optional[int] = None) -> None:
        """
        Initialize the Slingshot pseudotime inference method.

        Parameters
        ----------
        start_node : Optional[int], optional
            The starting node for the analysis. Defaults to None.
        """
        self.start_node = start_node

    def analyze(
        self, labels: np.ndarray, umap_embeddings: np.ndarray, output_dir: Optional[str] = None
    ) -> np.ndarray:
        """
        Perform pseudotime analysis using Slingshot.

        Parameters
        ----------
        labels : np.ndarray
            Cluster labels for the cells.
        umap_embeddings : np.ndarray
            UMAP-reduced embeddings.
        output_dir : Optional[str], optional
            Directory to save results and plots, by default None.

        Returns
        -------
        np.ndarray
            Pseudotime values for each cell.
        """
        # Prepare AnnData for Slingshot
        ad = self._prepare_AnnData(umap_embeddings, labels)

        # Initialize and fit Slingshot
        slingshot = self._initialize_slingshot(ad)

        # Extract pseudotime and save plots if required
        pseudotime = slingshot.unified_pseudotime
        if output_dir:
            self._save_plots(slingshot, output_dir)

        return pseudotime

    def _prepare_AnnData(self, umap_embeddings: np.ndarray, labels: np.ndarray) -> AnnData:
        """
        Prepare an AnnData object for Slingshot.

        Parameters
        ----------
        umap_embeddings : np.ndarray
            UMAP-reduced embeddings.
        labels : np.ndarray
            Cluster labels for the cells.

        Returns
        -------
        AnnData
            The prepared AnnData object.
        """
        ad = AnnData(X=umap_embeddings)
        ad.obs["celltype"] = labels
        ad.obsm["X_umap"] = umap_embeddings
        return ad

    def _initialize_slingshot(self, ad: AnnData) -> Slingshot:
        """
        Initialize and fit the Slingshot algorithm.

        Parameters
        ----------
        ad : AnnData
            The AnnData object with embeddings and labels.

        Returns
        -------
        Slingshot
            A fitted Slingshot object.
        """
        start = self.start_node if self.start_node is not None else 0
        slingshot = Slingshot(ad, celltype_key="celltype", obsm_key="X_umap", start_node=start)
        slingshot.fit(num_epochs=10)
        return slingshot

    def _save_plots(self, slingshot: Slingshot, output_dir: str) -> None:
        """
        Save cluster and pseudotime plots to the specified directory.

        Parameters
        ----------
        slingshot : Slingshot
            Fitted Slingshot object.
        output_dir : str
            Directory to save the plots.
        """
        os.makedirs(output_dir, exist_ok=True)
        fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
        axes[0].set_title("Clusters")
        axes[1].set_title("Pseudotime")

        slingshot.plotter.clusters(axes[0], labels=np.arange(slingshot.num_clusters), s=4, alpha=0.5)
        slingshot.plotter.curves(axes[0], slingshot.curves)
        slingshot.plotter.clusters(axes[1], color_mode="pseudotime", s=5)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pseudotime_visualization.svg"))
        plt.close()