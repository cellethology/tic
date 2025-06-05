# tic/annotation/api.py
"""
tic.annotation.api
====================

API for cell-type annotation.

Functions
---------
- annotate_adata: end-to-end pipeline for cell-type annotation (now with automatic caching).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import scanpy as sc
from anndata import AnnData

from ..constant import DEFAULT_DATACACHE_DIR
from .annotation import (
    assign_cell_clusters,
    return_top_genes,
    transform_cluster_annotation,
)
from .llm_caller import LLMPredictor

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def annotate_adata(
    adata: Union[str, Path, AnnData],
    *,
    assign_params: Dict[str, Any] | None = None,
    llm_model_name: str = "openai",
    llm_kwargs: Dict[str, Any] | None = None,
    dataset_description: str = "",
    groupby: str = "leiden",
    rank_genes_groups_method: str = "t-test",
    n_genes: int = 5,
    added_key: str = "pred_cell_type",
    copy: bool = False,
    return_cluster_annotation: bool = False,
) -> AnnData | Tuple[AnnData, Dict[str, Any]]:
    """
    End-to-end pipeline for cell-type annotation based on clustering and LLM predictions.
    If `adata.uns['tissue_id']` is set, automatically caches the new cell-type column to:
        {DEFAULT_DATACACHE_DIR}/{tissue_id}/{tissue_id}_llm_cell_types.parquet

    Parameters
    ----------
    adata : str | Path | AnnData
        The input data as an AnnData object or a path to an .h5ad file.
    assign_params : dict, optional
        Parameters passed to the `assign_cell_clusters` function (e.g. {"cluster_key": "leiden"}).
    llm_model_name : str, default="openai"
        The name of the LLM backend to use.
    llm_kwargs : dict, optional
        Keyword arguments passed to the LLM model (e.g., {"api_key": "XXX"}).
    dataset_description : str, optional
        Description of the dataset to include in the LLM prompt.
    groupby : str, default="leiden"
        The key in `.obs` to group cells by when ranking marker genes.
    rank_genes_groups_method : str, default="t-test"
        Method used to rank marker genes.
    n_genes : int, default=5
        Number of top marker genes per cluster to retrieve.
    added_key : str, default="pred_cell_type"
        The key in `.obs` to store the predicted cell type.
    copy : bool, default=False
        Whether to copy the AnnData object before processing.
    return_cluster_annotation : bool, default=False
        Whether to return the raw cluster‐to‐cell‐type annotation dictionary.

    Returns
    -------
    AnnData or (AnnData, dict)
        The annotated AnnData object (with `adata.obs[added_key]` set), and optionally
        the cluster-to-cell-type annotation dict.
    """
    # 1) Load AnnData if input is a file path
    if isinstance(adata, (str, Path)):
        adata = sc.read_h5ad(str(adata))
    if copy:
        adata = adata.copy()

    # 2) Perform clustering (e.g. Leiden) if needed
    assign_params = assign_params or {}
    adata = assign_cell_clusters(adata=adata, **assign_params)

    # 3) Rank marker genes for each cluster
    top_genes = return_top_genes(
        adata,
        groupby=groupby,
        rank_genes_groups_method=rank_genes_groups_method,
        n_genes=n_genes,
    )

    # 4) LLM-based annotation
    llm = LLMPredictor(model_name=llm_model_name, model_kwargs=llm_kwargs or {})
    cluster_annotation = llm.annotate_clusters(
        top_genes,
        dataset_description=dataset_description,
        return_json=True,
    )

    # 5) Map cluster annotations back to AnnData
    mapping = transform_cluster_annotation(cluster_annotation)
    adata.obs[added_key] = adata.obs[groupby].map(mapping).astype("category")

    # 6) If requested, return the cluster‐to‐cell‐type dict as well
    if return_cluster_annotation:
        # Before returning, attempt to cache
        _maybe_cache_cell_types(adata, added_key)
        return adata, cluster_annotation

    # Otherwise, just cache and return the single AnnData
    _maybe_cache_cell_types(adata, added_key)
    return adata


def _maybe_cache_cell_types(adata: AnnData, added_key: str) -> None:
    """
    If `adata.uns['tissue_id']` is set, write out a Parquet file:
        {DEFAULT_DATACACHE_DIR}/{tissue_id}/{tissue_id}_llm_cell_types.parquet

    The file will contain two columns:
       ["cell_id", added_key]

    If `adata.uns['tissue_id']` is missing, do nothing.
    """
    tissue_id = adata.uns.get("tissue_id", None)
    if tissue_id is None:
        # No tissue_id means we don't know where to save a cache
        return

    try:
        # Build cache folder path
        cache_folder = Path(DEFAULT_DATACACHE_DIR) / tissue_id
        cache_folder.mkdir(parents=True, exist_ok=True)

        # Name the file exactly: "{tissue_id}_llm_cell_types.parquet"
        cache_file = cache_folder / f"{tissue_id}_llm_cell_types.parquet"

        # Build a small DataFrame of [cell_id, added_key]
        df_cache = adata.obs[[added_key]].copy()
        df_cache["cell_id"] = df_cache.index
        df_cache = df_cache.reset_index(drop=True)

        # Write Parquet
        df_cache.to_parquet(cache_file)
        logger.info("Saved cell-type cache to %s", cache_file)
    except Exception as e:
        logger.warning("Failed to cache cell types for tissue '%s': %s", tissue_id, e)