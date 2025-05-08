
# tic/wrappers/graph.py
"""
Wrapper for building and querying cell-cell graphs.

This wrapper handles:
  1. Graph construction (KNN or radius).
  2. Subgraph extraction around a centre cell to represent the microenvironment.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Sequence, Union

import numpy as np
from anndata import AnnData
from matplotlib.axes import Axes

from ..config import GraphConfig
from ..graph.io import to_networkx, to_pyg
from ..graph.pp import compute_neighbors
from ..graph.tl import extract_subgraph
from ..graph.tl.subgraph import Strategy
from ..plotting.graph import plot_graph
from .base import BaseWrapper


class GraphWrapper(BaseWrapper[GraphConfig, Optional[AnnData]]):
    """
    Wrapper for constructing and querying spatial graphs stored in AnnData.

    Parameters
    ----------
    method : {'knn', 'radius'}
        Graph construction method.
    k : int
        Number of neighbors for KNN.
    radius : float, optional
        Distance threshold for radius graph.
    metric : str
        Distance metric for neighbor search.
    key : str
        Key under which adjacency is stored in `adata.obsp`.
    """

    def __init__(
        self,
        *,
        method: Literal['knn', 'radius'] = 'knn',
        k: int = 10,
        radius: Optional[float] = None,
        metric: str = 'euclidean',
        key: str = 'connectivities',
    ) -> None:
        super().__init__(
            GraphConfig(method=method, k=k, radius=radius, metric=metric, key=key)
        )

    def _fit_impl(
        self,
        adata: AnnData,
        *,
        copy: bool = False,
    ) -> Optional[AnnData]:
        """
        Build and store adjacency in `adata.obsp[key]`.
        """
        return compute_neighbors(
            adata,
            method=self.cfg.method,
            k=self.cfg.k,
            radius=self.cfg.radius,
            metric=self.cfg.metric,
            key_added=self.cfg.key,
            copy=copy,
        )

    def subgraph_indices(
        self,
        adata: AnnData,
        centre: Union[int, str],
        *,
        strategy: Optional[Strategy] = None,
        k: Optional[int] = None,
        radius: Optional[float] = None,
        hop: int = 2,
        metric: Optional[Literal['euclidean','cosine']] = 'euclidean',
    ) -> np.ndarray:
        """
        Extract neighbor indices around a centre cell.

        Parameters
        ----------
        adata : AnnData
            Annotated data with stored graph in `adata.obsp[key]`.
        centre : int or str
            Index or observation name of the centre cell.
        strategy : ["knn", "radius", "k_hop_shortest", "k_hop_pyg", "bfs_python", "slice_adj"], optional
            Extraction strategy; defaults to `self.cfg.method`.
        k : int, optional
            Number of neighbors for KNN; defaults to `self.cfg.k`.
        radius : float, optional
            Radius for 'radius' strategy; defaults to `self.cfg.radius`.
        hop : int
            Number of hops for 'k_hop' strategy.
        metric : str, optional
            Distance metric; defaults to `self.cfg.metric`.

        Returns
        -------
        np.ndarray
            Array of neighbor cell indices.
        """
        strat = strategy or self.cfg.method
        return extract_subgraph(
            adata,
            centre,
            strategy=strat,
            k=k or self.cfg.k,
            radius=radius if radius is not None else self.cfg.radius,
            hop=hop,
            metric=metric or self.cfg.metric,
            return_type='indices',
            graph_key=self.cfg.key,
        )

    def plot(
        self,
        adata: AnnData,
        *,
        indices: Optional[Sequence[int]] = None,
        color_by: str = 'cell_type',
        ax: Optional[Axes] = None,
        palette: Optional[dict[str, str]] = None,
        show: bool = True,
    ) -> Axes:
        """
        Plot the spatial graph, optionally highlighting a subgraph.
        """
        return plot_graph(
            adata,
            indices=indices,
            color_by=color_by,
            graph_key=self.cfg.key,
            ax=ax,
            palette=palette,
            show=show,
        )

    def as_networkx(self, adata: AnnData) -> Any:
        """
        Convert the stored graph to a NetworkX Graph.
        """
        return to_networkx(adata, graph_key=self.cfg.key)

    def as_pyg(self, adata: AnnData) -> Any:
        """
        Convert the stored graph to a PyTorch Geometric Data object.
        """
        return to_pyg(adata, graph_key=self.cfg.key)
