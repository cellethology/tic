# tic/wrappers/feature.py
"""
Facade for micro-environment feature extraction using named recipes.

Users can select from built-in recipes, specify which cell types act as “centres”,
and override key parameters for graph construction and subgraph extraction.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence, Optional
from anndata import AnnData


from ..config import FeatureConfig
from .base import BaseWrapper
from ..features import extract

class FeatureWrapper(BaseWrapper[FeatureConfig, AnnData]):
    """
    Extract micro-environment features according to a named recipe.

    Parameters
    ----------
    recipe : str
        Name of the feature recipe to apply. Use `FeatureWrapper.recipes()`
        to list available options.
    centre_types : Optional[Sequence[str]]
        Categories in `adata.obs['cell_type']` to treat as micro-environment centres.
        If `None`, all cells are used as centres.
    graph_params : Optional[Dict[str, Any]] see `tic.graph.pp.compute_neighbors` for more details
        Overrides for the recipe's graph-building extractor.
        graph_params will be passed to `tic.graph.pp.compute_neighbors` and saved in `adata.uns['graph_params']`
        Common keys:
          - method: str (e.g. 'knn' or 'radius' or 'voronoi')
          - k: int (only for knn)
          - radius: float (only for radius)
          - metric: str
          - key: Optional[str]
    subgraph_params : Optional[Dict[str, Any]] see `tic.graph.tl.extract_subgraph` for more details
        Overrides for the recipe's subgraph-extraction extractor.
        subgraph_params will be passed to `tic.graph.tl.extract_subgraph`
        Common keys:
          - strategy: str (e.g. "knn", "radius", "k_hop_shortest", "k_hop_pyg", "bfs_python", "slice_adj" )
          - hop: int
          - radius: float 
            for 'radius' strategy: this is the radius of the circle
            for other strategies: this will cut off the edges longer than the radius 

    Examples
    --------
    >>> from tic.wrappers.feature import FeatureWrapper
    >>> fw = FeatureWrapper(
    ...     recipe='tme_default',
    ...     centre_types=['Tumor'],
    ...     graph_params={'method':'voronoi'},
    ...     subgraph_params={'strategy':'k_hop_pyg','hop':2},
    ... )
    >>> fea_adata = fw.fit(adata)
    >>> fea_adata.obs.head()
    """

    def __init__(
        self,
        *,
        recipe: str = "tme_default",
        centre_types: Optional[Sequence[str]] = None,
        graph_params: Optional[Dict[str, Any]] = None,
        subgraph_params: Optional[Dict[str, Any]] = None,
        build_graph_if_missing: bool = True,
    ) -> None:
        super().__init__(
            FeatureConfig(
                recipe=recipe,
                centre_types=list(centre_types) if centre_types else None,
                graph_params=dict(graph_params) if graph_params else None,
                subgraph_params=dict(subgraph_params) if subgraph_params else None,
                build_graph_if_missing=build_graph_if_missing,
            )
        )

    def _fit_impl(self, adata: AnnData, *, copy: bool = True) -> AnnData:
        try:
            fea = extract(
                adata,
                recipe=self.cfg.recipe,
                centre_types=self.cfg.centre_types,
                graph_params=self.cfg.graph_params or {},
                subgraph_params=self.cfg.subgraph_params or {},
            )
        except KeyError as error:
            raise ValueError(
                f"Feature extraction failed: {error}. "
                f"Available recipes: {self.recipes()}"
            ) from error

        return fea.copy() if copy else fea

    @staticmethod
    def recipes() -> list[str]:
        """
        Return the list of built-in feature recipe names.
        """
        from ..features.recipes import _BUILTIN  # lazy import

        return list(_BUILTIN)

    @property
    def config_dict(self) -> Dict[str, Any]:
        """
        Return the wrapper’s initialization parameters as a dictionary.
        """
        return self.cfg.to_dict()