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
from concurrent.futures import ThreadPoolExecutor
import os

from tic.graph.utils import estimate_radius

__all__ = [
    "list_available",
    "describe",
    "extract",
    "register",
]

from typing import Any, Mapping, Sequence
import logging
import time

import numpy as np
import pandas as pd
from anndata import AnnData
from tqdm import tqdm

from ..graph.pp import compute_neighbors
from ..graph.tl import extract_subgraph

from .recipes import get_recipe
from .registry import FeatureRegistry, register

# set up logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


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
    build_graph_if_missing: bool = True,
    n_jobs: int | None = None,
    test_mode: bool = False,
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
        Whether to auto-build a default KNN graph (skipped for radius strategy).
    n_jobs
        Number of threads for subgraph extraction (None=auto).
    test_mode
        If True, logs timing for graph build and subgraph extraction.
    """
    # 1) Determine centre indices
    if centre_types is not None:
        if 'cell_type' not in adata.obs.columns:
            raise ValueError("`adata.obs['cell_type']` is missing.")
        missing = set(centre_types) - set(adata.obs['cell_type'].unique())
        if missing:
            raise ValueError(f"Centre types not in adata.obs['cell_type']: {missing}.")
        centres = np.where(adata.obs['cell_type'].isin(centre_types))[0]
    else:
        centres = np.arange(adata.n_obs)

    # 2) Graph build only for non-radius strategies
    sub_defaults = {'strategy': 'radius', 'radius': estimate_radius(adata, n_samples=1000)}
    sub_kwargs = {**sub_defaults, **(subgraph_params or {})}
    sub_kwargs.pop('return_type', None)
    strat = sub_kwargs.get('strategy', 'radius')
    if build_graph_if_missing and strat != 'radius':
        params = {'method': 'knn', 'k': 10}
        params.update(graph_params or {})
        if test_mode:
            t0 = time.time()
            compute_neighbors(adata, **params)
            logger.info(f"compute_neighbors time: {time.time() - t0:.3f}s")
        else:
            compute_neighbors(adata, **params)

    # 3) Instantiate extractors
    recipe_dict = get_recipe(recipe)
    extractors = [FeatureRegistry.get(name)(**params) for name, params in recipe_dict.items()]

    # 4) Containers
    obsm_data: dict[str, list[np.ndarray]] = {ext.name: [] for ext in extractors}
    obs_rows: list[pd.DataFrame] = []

    # 5) Parallel subgraph extract & transform
    def _process(c: int):
        neigh = extract_subgraph(adata, c, return_type='indices', **sub_kwargs)
        feats = [ext.transform(adata, centre_idx=c, neighbour_idx=neigh) for ext in extractors]
        return c, feats

    max_workers = n_jobs or os.cpu_count() or 1
    if test_mode:
        t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = list(exe.map(_process, centres))
    if test_mode:
        logger.info(f"parallel extract time: {time.time() - t0:.3f}s")

    for c, feats in tqdm(futures, total=len(centres), desc="Extracting subgraphs"):
        obs_rows.append(adata.obs.iloc[[c]])
        for ext, vec in zip(extractors, feats):
            obsm_data[ext.name].append(vec)

    # 6) Build output AnnData
    obs = pd.concat(obs_rows, ignore_index=True)
    out = AnnData(X=adata.X[centres], obs=obs, var=adata.var.copy(), uns={})
    for name, mats in obsm_data.items():
        out.obsm[name] = np.vstack(mats)

    # 7) Metadata & predictors
    out.uns['feature_modes'] = list(recipe_dict.keys())
    out.uns['graph_params'] = graph_params or {}
    out.uns['subgraph_params'] = sub_kwargs
    blocks = [out.obsm[ext.name] for ext in extractors]
    out.obsm['X_predictors'] = np.hstack(blocks)
    out.uns['X_predictors_names'] = sum((ext.feature_names(adata) for ext in extractors), [])
    meta_rows = [pd.DataFrame(ext.feature_meta(adata) or {'name': ext.feature_names(adata)}) for ext in extractors]
    out.uns['X_predictors_meta'] = pd.concat(meta_rows, ignore_index=True)

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
