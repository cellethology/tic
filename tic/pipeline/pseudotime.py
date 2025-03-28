from typing import Optional
import os
import numpy as np
import anndata

from tic.pseudotime.clustering import Clustering
from tic.pseudotime.dimensionality_reduction import DimensionalityReduction
from tic.pseudotime.pseudotime import SlingshotMethod

def run_pseudotime_pipeline(
    adata: anndata.AnnData,
    rep_key: str = "raw_expression",
    copy: bool = True,
    dr_method: str = "UMAP",
    n_components: int = 2,
    cluster_method: str = "kmeans",
    n_clusters: int = 2,
    start_node: Optional[int] = None,
    output_dir: Optional[str] = None
) -> Optional[anndata.AnnData]:
    """
    Run a high-level pseudotime inference pipeline on an AnnData object.
    
    The pipeline consists of:
      1. Dimensionality reduction on the representation specified by `rep_key` from adata.obsm.
         If `rep_key` is not found, the pipeline falls back to using adata.X.
      2. Clustering on the reduced embeddings.
      3. Pseudotime inference using the Slingshot algorithm.
         - If start_node is provided, Slingshot uses that candidate.
         - Otherwise, the pipeline iterates over all unique cluster labels and aggregates pseudotime.
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing center cell representations.
        Expected to have a representation stored under obsm[rep_key] or, as a fallback, in X.
    rep_key : str, optional
        Key in adata.obsm to use for dimensionality reduction, by default "raw_expression".
    copy : bool, optional
        If True, operate on a copy of adata and return a new AnnData object; if False, modify adata in place and return None.
    dr_method : str, optional
        Dimensionality reduction method ("PCA" or "UMAP"). Default is "UMAP".
    n_components : int, optional
        Number of dimensions to reduce to. Default is 2.
    cluster_method : str, optional
        Clustering method ("kmeans" or "agg"). Default is "kmeans".
    n_clusters : int, optional
        Number of clusters to form. Default is 2.
    start_node : Optional[int], optional
        Starting node for pseudotime inference. If None, the pipeline iterates over all unique cluster labels
        and aggregates the resulting pseudotime values.
    output_dir : Optional[str], optional
        Directory to save pseudotime plots. If None, plots are not saved.
    
    Returns
    -------
    Optional[anndata.AnnData]
        If copy is True, returns a new AnnData object with updated fields:
            - obsm["rp_reduced"]: Reduced representation.
            - obs["cluster"]: Cluster labels.
            - obs["pseudotime"]: Inferred pseudotime values.
        If copy is False, modifies adata in place and returns None.
    """
    # Work on a copy if required.
    if copy:
        adata = adata.copy()
    
    # Step 1: Dimensionality Reduction.
    if rep_key in adata.obsm:
        embeddings = adata.obsm[rep_key]
    else:
        embeddings = adata.X
    dr = DimensionalityReduction(method=dr_method, n_components=n_components)
    reduced_embeddings = dr.reduce(embeddings)
    adata.obsm["rp_reduced"] = reduced_embeddings

    # Step 2: Clustering.
    clusterer = Clustering(method=cluster_method, n_clusters=n_clusters)
    cluster_labels = clusterer.cluster(reduced_embeddings)
    adata.obs["cluster"] = cluster_labels

    # Step 3: Pseudotime Inference using Slingshot.
    if start_node is not None:
        slingshot = SlingshotMethod(start_node=start_node)
        pseudotime = slingshot.analyze(cluster_labels, reduced_embeddings, output_dir=output_dir or "")
    else:
        unique_clusters = np.unique(cluster_labels)
        pt_candidates = []
        for candidate in unique_clusters:
            slingshot = SlingshotMethod(start_node=candidate)
            pt_candidate = slingshot.analyze(cluster_labels, reduced_embeddings, output_dir="")
            pt_candidates.append(pt_candidate)
        pseudotime = np.min(np.vstack(pt_candidates), axis=0)
    
    adata.obs["pseudotime"] = pseudotime

    return adata if copy else None