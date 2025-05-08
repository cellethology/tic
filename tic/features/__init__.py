# tic/features/__init__.py
"""
High-level API for feature extraction.

Core entry points
-----------------
list_available
    List names of all registered extractors.
describe
    Return the docstring of a particular extractor.
extract
    Run a *recipe* to build feature vectors with user-specified graph/subgraph params.
register
    Add a custom FeatureExtractor class to the registry.
"""

from __future__ import annotations

__all__ = [
    "list_available",
    "describe",
    "extract",
    "register",
]

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from ..graph.pp import compute_neighbors
from ..graph.tl import extract_subgraph

from .recipes import get_recipe
from .registry import FeatureRegistry, register


def list_available() -> list[str]:
    """Return all registered extractor names."""
    return FeatureRegistry.names()


def describe(name: str) -> str:
    """Return the docstring for a registered extractor."""
    cls = FeatureRegistry.get(name)
    return cls.__doc__ or "No description available."


def extract(
    adata: AnnData,
    *,
    recipe: str | Mapping[str, Mapping[str, Any]] = "cell_basic",
    centre_types: Sequence[str] | None = None,
    graph_params: Mapping[str, Any] | None = None,
    subgraph_params: Mapping[str, Any] | None = None,
    build_graph_if_missing: bool = True
    
) -> AnnData:
    """
    Apply a recipe and return a new AnnData with original X/var_names preserved,
    and extractor outputs stored in .obsm.

    Parameters
    ----------
    adata
        Tissue-level AnnData (`.obsm['spatial']` required).
    recipe
        Built-in recipe name or mapping of extractor_name → params.
    centre_types
        Which cell types to center on; `obs['cell_type']`.
    graph_params
        Parameters for compute_neighbors if no graph exists.
    subgraph_params
        Parameters for extract_subgraph (strategy, hop, etc.).
    build_graph_if_missing
        Whether to auto-build a default KNN graph.

    Returned AnnData
    -----------------

    """
    # 1) determine centre indices
    if centre_types is not None:
        centres = np.where(adata.obs['cell_type'].isin(centre_types))[0]
    else:
        centres = np.arange(adata.n_obs)

    # 2) ensure graph
    if build_graph_if_missing and not adata.obsp:
        params = {'method': 'knn', 'k': 10}
        params.update(graph_params or {})
        compute_neighbors(adata, **params)

    # 3) instantiate extractors
    recipe_dict = get_recipe(recipe)  # extractor_name -> default params
    extractors = [FeatureRegistry.get(name)(**params) for name, params in recipe_dict.items()]

    # 4) prepare subgraph kwargs
    sub_defaults = {'strategy': 'radius', 'radius': 100}
    sub_kwargs = {**sub_defaults, **(subgraph_params or {})}
    sub_kwargs.pop('return_type', None)

    # 5) initialize obsm containers
    obsm_data: dict[str, list[np.ndarray]] = {ext.name: [] for ext in extractors}
    obs_rows = []

    # 6) for each centre, compute and cache
    for c in centres:
        neigh = extract_subgraph(adata, c, return_type='indices', **sub_kwargs)
        obs_rows.append(adata.obs.iloc[[c]])
        for ext in extractors:
            vec = ext.transform(adata, centre_idx=c, neighbour_idx=neigh)
            obsm_data[ext.name].append(vec)

    # 7) build new AnnData preserving original X and var_names for centres
    obs = pd.concat(obs_rows, ignore_index=True)
    out = AnnData(
        X=adata.X[centres],
        obs=obs,
        var=adata.var.copy(),
        uns={},
    )

    # 8) assign extractor outputs to obsm as matrices
    for name, mats in obsm_data.items():
        out.obsm[name] = np.vstack(mats)

    # 9) record metadata
    out.uns['feature_modes'] = list(recipe_dict.keys())
    out.uns['graph_params'] = graph_params or {}
    out.uns['subgraph_params'] = sub_kwargs

    return out


# ----------------------------------------------------------------------
# Ensure all extractor modules are imported so their @register decorator runs
import importlib
import pkgutil

def _import_all_submodules(package_name):
    """Recursively import *all* sub-modules inside ``package_name``."""
    package = importlib.import_module(package_name)
    for mod_info in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}."):
        mod_name = mod_info.name
        # Skip meta‑modules that never contain extractors
        if mod_name.rsplit(".", 1)[-1] in {"base", "recipes", "registry"}:
            continue
        importlib.import_module(mod_name)

_import_all_submodules(__name__)
