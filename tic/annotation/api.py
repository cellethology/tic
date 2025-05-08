# tic/annotation/api.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import scanpy as sc
from anndata import AnnData

from .annotation import (
    assign_cell_clusters,
    return_top_genes,
    transform_cluster_annotation,
)
from .llm_caller import LLMPredictor


def annotate_adata(
    adata: str | Path | AnnData,
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

    Parameters
    ----------
    adata : str | Path | AnnData
        The input data as an AnnData object or a path to an .h5ad file.
    assign_params : dict, optional
        Parameters passed to the `assign_cell_clusters` function.
    llm_model_name : str, default="openai"
        The name of the LLM backend to use.
    llm_kwargs : dict, optional
        Keyword arguments passed to the LLM model (e.g., API key).
    dataset_description : str, optional
        Description of the dataset to include in the LLM prompt.
    groupby : str, default="leiden"
        The key in `.obs` to group cells by when ranking marker genes.
    rank_genes_groups_method : str, default="t-test"
        Method used to rank marker genes.
    n_genes : int, default=5
        Number of top marker genes per cluster to retrieve.
    copy : bool, default=False
        Whether to copy the AnnData object before processing.
    return_cluster_annotation : bool, default=False
        Whether to return the cluster-to-cell-type annotation dictionary.

    Returns
    -------
    AnnData or (AnnData, dict)
        The annotated AnnData object, optionally with the annotation dictionary.
        added_key: str, default="pred_cell_type"
            The key in `.obs` to store the predicted cell type.
        
    Examples
    --------
    >>> from tic.annotation import annotate_adata
    >>> adata = annotate_adata(adata, added_key='pred_cell_type')
    """
    # Load AnnData if input is a file path
    if isinstance(adata, (str, Path)):
        adata = sc.read_h5ad(str(adata))
    if copy:
        adata = adata.copy()

    # Perform clustering
    assign_params = assign_params or {}
    adata = assign_cell_clusters(adata=adata, **assign_params)

    # Rank marker genes
    top_genes = return_top_genes(
        adata,
        groupby=groupby,
        rank_genes_groups_method=rank_genes_groups_method,
        n_genes=n_genes,
    )

    # Perform LLM-based annotation
    llm = LLMPredictor(model_name=llm_model_name, model_kwargs=llm_kwargs or {})
    cluster_annotation = llm.annotate_clusters(
        top_genes,
        dataset_description=dataset_description,
        return_json=True,
    )

    # Map cluster annotations back to AnnData
    mapping = transform_cluster_annotation(cluster_annotation)
    adata.obs[added_key] = adata.obs[groupby].map(mapping).astype("category")

    if return_cluster_annotation:
        return adata, cluster_annotation
    return adata