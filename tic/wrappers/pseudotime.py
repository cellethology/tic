# tic/wrappers/pseudotime.py
"""
Wrapper for end-to-end pseudotime inference.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Literal, Mapping, Sequence
import numpy as np
from anndata import AnnData
from matplotlib import pyplot as plt
import pandas as pd

from ..constant import DEFAULT_KEY
from ..pseudotime.pp.clustering import ClusterMethod
from ..pseudotime.pp.dimensionality import ReductionMethod

from ..plotting.pseudotime import scatter_embedding
from ..config import PseudotimeConfig
from ..pipeline.pseudotime import PseudotimePipeline
from .base import BaseWrapper


class PseudotimeWrapper(BaseWrapper[PseudotimeConfig, AnnData]):
    """User-friendly façade around :class:`PseudotimePipeline`."""

    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        rep_key: str = "centre_gene",
        cluster_method: str | ClusterMethod = "kmeans",
        n_clusters: int = 2,
        dr_method: str | ReductionMethod = "pca",
        n_components: int = 2,
        start_node: int | None = None,
        output_dir: Path | None = None,
        random_state: int | None = 42,
        step_order: Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            PseudotimeConfig(
                rep_key=rep_key,
                dr_method=dr_method,
                n_components=n_components,
                cluster_method=cluster_method,
                n_clusters=n_clusters,
                start_node=start_node,
                output_dir=output_dir,
                random_state=random_state,
                step_order=step_order,
            )
        )
        if self.cfg.output_dir:
            os.makedirs(self.cfg.output_dir, exist_ok=True)
            self.save_params(os.path.join(self.cfg.output_dir, 'params.json'))
    
    def _check_reprensation(self, adata: AnnData): 
        """
        Check if the representation obsm data is valid: 
            the data should be a 2D array-like object and can be converted to a numpy array.
        """
        if not isinstance(adata.obsm[self.cfg.rep_key], np.ndarray):
            adata.obsm[self.cfg.rep_key] = adata.obsm[self.cfg.rep_key].toarray()
        return adata
    # ------------------------------------------------------------------
    def _fit_impl(self, adata: AnnData, *, copy: bool = True) -> AnnData:  # noqa: D401
        adata = self._check_reprensation(adata)
        pipeline = PseudotimePipeline(self.cfg, steps=self.cfg.step_order)
        return pipeline.run(adata, copy=copy)

    # ── Convenience accessors -------------------------------------------------
    @property
    def embedding(self) -> np.ndarray:  # noqa: D401 – property docstring below
        """Return the reduced embedding *after* a successful ``fit`` call."""
        return self.result.obsm[DEFAULT_KEY.get('rp_reduced')]

    @property
    def clusters(self) -> np.ndarray:  # noqa: D401
        """Cluster labels returned by the clustering step."""
        return self.result.obs[DEFAULT_KEY.get('cluster')].to_numpy()

    @property
    def pseudotime(self) -> np.ndarray:  # noqa: D401
        """Continuous pseudotime values (zeros if Slingshot skipped)."""
        return self.result.obs[DEFAULT_KEY.get('pseudotime')].to_numpy()

    @property
    def metrics(self) -> pd.DataFrame:  # noqa: D401
        """
        Monotonicity / trend metrics (only if the *metrics* step was run).

        Returns: dataframe with the following columns:
            biomarker	mono_method	mono_correlation	mono_p_value	trend_method	trend_trend	trend_statistic	trend_p_value

            biomarker: the name of the biomarker(str)
            mono_method: the method used to calculate the monotonicity(str)
            mono_correlation: the correlation coefficient of the monotonicity(float)
            mono_p_value: the p-value of the monotonicity(float)
            trend_method: the method used to calculate the trend(str)
            trend_trend: the trend of the biomarker(str)
            trend_statistic: the statistic of the trend(float)
            trend_p_value: the p-value of the trend(float)
        """
        return self.result.uns["pseudotime_metrics"]

    # ------------------------------------------------------------------
    def plot(
        self,
        kind: Literal["pseudotime", "cluster"] = "pseudotime",
        *,
        ax: "plt.Axes | None" = None,
        palette: Mapping[int, str] | None = None,
        save_path: str | None = None,
    ) -> "plt.Axes":  # noqa: D401 – returns ax
        """Scatter the embedding coloured by *pseudotime* or *cluster*."""
        from matplotlib import pyplot as plt  # local import (optional dependency)

        ax = scatter_embedding(
            self.embedding,
            self.clusters,
            self.pseudotime,
            kind=kind,
            ax=ax,
            palette=palette,
            save_path=save_path,
        )
        return ax

