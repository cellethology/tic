# file: tic/pseudotime/tl/slingshot.py
"""Wrapper around *pyslingshot* for pseudotime inference."""
from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from pyslingshot import Slingshot

from .base import PseudotimeMethod

__all__ = ["SlingshotMethod"]


class SlingshotMethod(PseudotimeMethod):
    """Pseudotime inference using the Slingshot algorithm."""

    def __init__(self, *, start_node: Optional[int] = None, epochs: int = 10) -> None:
        """
        Parameters
        ----------
        start_node
            The starting cluster/node for Slingshot; defaults to None (cluster 0).
        epochs
            Number of training epochs for Slingshot.
        """
        self.start_node = start_node
        self.epochs = epochs

    def fit_predict(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        *,
        output_dir: Optional[str] = None,
    ) -> np.ndarray:
        """
        Run Slingshot pseudotime inference.

        Parameters
        ----------
        embeddings
            Array of reduced embeddings (n_cells × n_dims).
        labels
            Cluster labels for each cell. (n_cells,)
        output_dir
            If provided, save diagnostic plots in this directory.

        Returns
        -------
        np.ndarray
            Pseudotime values for each cell.
        """
        # build AnnData
        ad = self._to_anndata(embeddings, labels)

        # ── Catch the string‐raise bug from pyslingshot/principal curve ──
        try:
            sl = self._fit_slingshot(ad)
        except TypeError as e:
            # this almost certainly means "points" had <2 rows
            raise RuntimeError(
                "Slingshot failed (too few points/clusters for principal curve)"
            ) from None

        # save plots if asked
        if output_dir:
            self._save_plots(sl, output_dir)

        return sl.unified_pseudotime

    # ------------------------------------------------------------------
    def _to_anndata(self, emb: np.ndarray, lbl: np.ndarray) -> AnnData:
        """
        Construct an AnnData object for Slingshot input.
        """
        ad = AnnData(X=emb)
        ad.obs["celltype"] = lbl.astype(str)
        ad.obsm["X_umap"] = emb
        return ad

    def _fit_slingshot(self, ad: AnnData) -> Slingshot:
        """
        Instantiate and fit the Slingshot model.
        """
        start = self.start_node if self.start_node is not None else 0
        sl = Slingshot(ad, celltype_key="celltype", obsm_key="X_umap", start_node=start)
        sl.fit(num_epochs=self.epochs)
        return sl

    def _save_plots(self, sl: Slingshot, out_dir: str) -> None:
        """
        Save cluster and pseudotime diagnostic plots.
        """
        os.makedirs(out_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].set_title("Clusters")
        axes[1].set_title("Pseudotime")

        # plot clusters and curves
        sl.plotter.clusters(axes[0], labels=np.arange(sl.num_clusters), s=4, alpha=0.6)
        sl.plotter.curves(axes[0], sl.curves)
        # plot pseudotime
        sl.plotter.clusters(axes[1], color_mode="pseudotime", s=5)

        plt.tight_layout()
        fig_path = os.path.join(out_dir, "cluster_and_ptime.png")
        plt.savefig(fig_path)
        plt.close(fig)
