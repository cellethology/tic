# tic.pseudotime.pseudotime

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
    def analyze(self, labels: np.ndarray, umap_embeddings: np.ndarray, output_dir: str) -> np.ndarray:
        """
        Perform pseudotime analysis on the given embeddings and cluster labels.

        Args:
            labels (np.ndarray): Cluster labels for the embeddings.
            umap_embeddings (np.ndarray): UMAP-reduced embeddings.
            output_dir (str): Directory to save results and plots.

        Returns:
            np.ndarray: Pseudotime values for each cell.
        """
        pass

class SlingshotMethod(PseudotimeMethod):
    """
    Pseudotime analysis using the Slingshot algorithm.

    Attributes:
        start_node (Optional[int]): The starting node for the pseudotime analysis. If None, defaults to all clusters.
    """
    def __init__(self, start_node: Optional[int] = None):
        """
        Initialize the Slingshot pseudotime inference method.

        Args:
            start_node (Optional[int]): The starting node for the pseudotime analysis. If None, defaults to all clusters.
        """
        self.start_node = start_node

    def analyze(self, labels: np.ndarray, umap_embeddings: np.ndarray, output_dir: Optional[str] = None) -> np.ndarray:
        """
        Perform pseudotime analysis using Slingshot.

        Args:
            labels (np.ndarray): Cluster labels for the cells.
            umap_embeddings (np.ndarray): UMAP-reduced embeddings.
            output_dir (Optional[str]): Directory to save results and plots.

        Returns:
            np.ndarray: Pseudotime values for each cell.
        """
        # Convert to AnnData format for Slingshot
        ad = self._prepare_AnnData(umap_embeddings, labels)

        # Initialize Slingshot and fit the model
        slingshot = self._initialize_slingshot(ad)

        # Extract pseudotime and plot results if output directory is provided
        pseudotime = slingshot.unified_pseudotime
        if output_dir:
            self._save_plots(slingshot, output_dir)

        return pseudotime

    def _prepare_AnnData(self, umap_embeddings: np.ndarray, labels: np.ndarray) -> AnnData:
        """
        Prepare the AnnData object for Slingshot.

        Args:
            umap_embeddings (np.ndarray): UMAP-reduced embeddings.
            labels (np.ndarray): Cluster labels for the cells.

        Returns:
            AnnData: The prepared AnnData object.
        """
        ad = AnnData(X=umap_embeddings)
        ad.obs['celltype'] = labels
        ad.obsm['X_umap'] = umap_embeddings
        return ad

    def _initialize_slingshot(self, ad: AnnData) -> Slingshot:
        """
        Initialize the Slingshot algorithm with the given AnnData object.

        Args:
            ad (AnnData): The AnnData object containing embeddings and labels.

        Returns:
            Slingshot: A fitted Slingshot object.
        """
        slingshot = Slingshot(ad, celltype_key="celltype", obsm_key="X_umap", start_node=self.start_node if self.start_node is not None else 0)
        slingshot.fit(num_epochs=10)
        return slingshot

    def _save_plots(self, slingshot: Slingshot, output_dir: str):
        """
        Save plots of clusters and pseudotime to the output directory.

        Args:
            slingshot (Slingshot): Fitted Slingshot object.
            output_dir (str): Directory to save the plots.
        """
        os.makedirs(output_dir, exist_ok=True)
        fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
        axes[0].set_title('Clusters')
        axes[1].set_title('Pseudotime')

        slingshot.plotter.clusters(axes[0], labels=np.arange(slingshot.num_clusters), s=4, alpha=0.5)
        slingshot.plotter.curves(axes[0], slingshot.curves)
        slingshot.plotter.clusters(axes[1], color_mode='pseudotime', s=5)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/pseudotime_visualization.svg")
        plt.close()
