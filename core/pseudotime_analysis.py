# -*- coding: utf-8 -*-
"""
Created on Monday Dec 23 21:58 2024

@author: Jiahao Zhang
@Description: Pseudotime analysis for tumor embeddings using UMAP, KNN clustering, and Slingshot algorithm
"""

import os
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from pyslingshot import Slingshot
from anndata import AnnData
from spacegm.embeddings_analysis import dimensionality_reduction_combo
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from umap import UMAP
from sklearn.cluster import AgglomerativeClustering
# ---------------------
# data class
# ---------------------
class CellEmbedding(BaseModel):
    """
    A class to represent embedding data and additional attributes.
    
    Attributes:
        identifiers (List[List[Any]]): List of [region_id, cell_id] for each cell.
        embeddings (Dict[str, List[Any]]): Dictionary of embeddings keyed by type, e.g., 'node_embeddings', 'graph_embeddings'.
        attributes (Optional[Dict[str, List[Any]]]): Optional dictionary to store additional attributes like pseudotime, cluster labels, or any derived feature.
    """
    identifiers: List[List[Any]] = Field(..., description="List of [region_id, cell_id] for each cell.")
    embeddings: Dict[str, List[Any]] = Field(..., description="Dictionary of embeddings keyed by type, e.g., 'node_embeddings', 'graph_embeddings'.")
    attributes: Optional[Dict[str, List[Any]]] = Field(
        default_factory=dict,
        description="Optional dictionary for additional attributes, e.g., 'pseudotime', 'cluster_labels', etc."
    )

def add_attributes_from_raw_data(cell_embedding: CellEmbedding, raw_dir: str) -> CellEmbedding:
    """
    Add attributes (e.g., X, Y, SIZE, CELL_TYPE, biomarkers) to a CellEmbedding object 
    by retrieving data from the raw directory.

    Args:
        cell_embedding (CellEmbedding): The CellEmbedding object containing identifiers.
        raw_dir (str): The directory containing raw data files:
                       {cell_data}.csv, {expression}.csv, {cell_features}.csv, {cell_types}.csv.

    Returns:
        CellEmbedding: Updated CellEmbedding object with attributes populated.
    """
    attributes = {
        "X": [],
        "Y": [],
        "SIZE": [],
        "CELL_TYPE": []
    }
    # Initialize a dictionary to store biomarker expressions
    biomarker_data = {}

    for region_id, cell_id in cell_embedding.identifiers:
        # Load required raw data files for the given region
        cell_data_path = os.path.join(raw_dir, f"{region_id}.cell_data.csv")
        expression_path = os.path.join(raw_dir, f"{region_id}.expression.csv")
        cell_features_path = os.path.join(raw_dir, f"{region_id}.cell_features.csv")
        cell_types_path = os.path.join(raw_dir, f"{region_id}.cell_types.csv")

        # Read the raw data
        cell_data = pd.read_csv(cell_data_path, index_col="CELL_ID")
        expression = pd.read_csv(expression_path, index_col="CELL_ID")
        cell_features = pd.read_csv(cell_features_path, index_col="CELL_ID")
        cell_types = pd.read_csv(cell_types_path, index_col="CELL_ID")

        # Extract X and Y coordinates
        if cell_id in cell_data.index:
            attributes["X"].append(cell_data.loc[cell_id, "X"])
            attributes["Y"].append(cell_data.loc[cell_id, "Y"])
        else:
            attributes["X"].append(None)
            attributes["Y"].append(None)

        # Extract SIZE
        if cell_id in cell_features.index:
            attributes["SIZE"].append(cell_features.loc[cell_id, "SIZE"])
        else:
            attributes["SIZE"].append(None)

        # Extract CELL_TYPE (renamed from CLUSTER_LABEL)
        if "CLUSTER_LABEL" in cell_types.columns:
            cell_types = cell_types.rename(columns={"CLUSTER_LABEL": "CELL_TYPE"})
            
        if cell_id in cell_types.index:
            attributes["CELL_TYPE"].append(cell_types.loc[cell_id, "CELL_TYPE"])
        else:
            attributes["CELL_TYPE"].append(None)

        # Extract biomarker expressions
        if cell_id in expression.index:
            for biomarker in expression.columns:
                if biomarker not in biomarker_data:
                    biomarker_data[biomarker] = []
                biomarker_data[biomarker].append(expression.loc[cell_id, biomarker])
        else:
            for biomarker in expression.columns:
                if biomarker not in biomarker_data:
                    biomarker_data[biomarker] = []
                biomarker_data[biomarker].append(None)

    # Combine attributes and biomarker data
    attributes.update(biomarker_data)
    cell_embedding.attributes = attributes
    return cell_embedding

# -----------------------
# reduce dimension & assign cluster labels
# -----------------------

def dimensionality_reduction_and_clustering(
    embeddings: np.ndarray,
    n_pca_components: int = 20,
    cluster_method: str = "kmeans",
    n_clusters: int = 10,
    seed: int = 42,
    tool_saves: Optional[Tuple[Any, Any, Any]] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Tuple[Any, Any, Any]]:
    """
    Perform dimensionality reduction and clustering on given embeddings.

    Args:
        embeddings (np.ndarray): Input embeddings (e.g., node, graph, or composition vectors).
        n_pca_components (int): Number of PCA components to retain.
        cluster_method (str): Clustering method to use ('kmeans' or 'agg').
        n_clusters (int): Number of clusters for the clustering algorithm.
        seed (int): Random seed for reproducibility.
        tool_saves (Optional[Tuple[Any, Any, Any]]): Pre-initialized PCA, UMAP, and clustering objects. If None, new ones are created.

    Returns:
        Tuple: 
            - np.ndarray: PCA-reduced embeddings.
            - Optional[np.ndarray]: UMAP-reduced embeddings, if UMAP is installed.
            - np.ndarray: Cluster labels for each data point.
            - Tuple[Any, Any, Any]: Tuple of PCA, UMAP (if used), and clustering objects.
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize or use provided tools
    pca, reducer, clustering = tool_saves if tool_saves else (None, None, None)

    # Step 1: PCA
    if pca is None:
        pca = PCA(n_components=n_pca_components, random_state=seed)
        pca_embeddings = pca.fit_transform(embeddings)
    else:
        pca_embeddings = pca.transform(embeddings)

    # Step 2: UMAP
    if reducer is None:
        try:
            import umap
            reducer = umap.UMAP(random_state=seed)
            umap_embeddings = reducer.fit_transform(pca_embeddings)
        except ImportError:
            print("UMAP is not installed. Skipping UMAP dimensionality reduction.")
            reducer, umap_embeddings = None, None
    else:
        umap_embeddings = reducer.transform(pca_embeddings)

    # Step 3: Clustering
    if clustering is None:
        if cluster_method == "agg":
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
        elif cluster_method == "kmeans":
            clustering = KMeans(n_clusters=n_clusters, random_state=seed)
        else:
            raise ValueError(f"Unsupported clustering method: {cluster_method}")
        cluster_labels = clustering.fit_predict(pca_embeddings)
    else:
        cluster_labels = clustering.predict(pca_embeddings)

    return pca_embeddings, umap_embeddings, cluster_labels, (pca, reducer, clustering)
# -----------------------
# Pseudotime Analysis with Slingshot
# -----------------------
def perform_pseudotime_analysis(labels, umap_embs, output_dir, start=None, show_plots=False):
    """
    Perform pseudotime analysis using Slingshot.

    Args:
        labels (np.ndarray): Cluster labels for the embeddings.
        umap_embs (np.ndarray): UMAP-reduced embeddings.
        output_dir (str): Directory to save results and plots.
        start_nodes (list of int, optional): Starting clusters for pseudotime analysis. Defaults to all clusters.
        show_plots (bool): Whether to show plots or not. Defaults to False.

    Returns:
        dict: A dictionary mapping each start node to its pseudotime values.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert to AnnData format for Slingshot
    ad = AnnData(X=umap_embs)
    ad.obs['celltype'] = labels
    ad.obsm['X_umap'] = umap_embs

    print(f"Performing pseudotime analysis with start node {start}...")

    # Initialize Slingshot
    slingshot = Slingshot(
        ad,
        celltype_key="celltype",
        obsm_key="X_umap",
        start_node=start
    )

    # Fit the pseudotime model
    slingshot.fit(num_epochs=10)

    # Extract pseudotime
    pseudotime = slingshot.unified_pseudotime

    # Plot clusters and pseudotime
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
    axes[0].set_title('Clusters')
    axes[1].set_title('Pseudotime')

    slingshot.plotter.clusters(axes[0], labels=np.arange(slingshot.num_clusters), s=4, alpha=0.5)
    slingshot.plotter.curves(axes[0], slingshot.curves)
    slingshot.plotter.clusters(axes[1], color_mode='pseudotime', s=5)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "pseudotime_visualization.png")
    plt.savefig(plot_path)

    if show_plots:
        plt.show()
    plt.close()

    print(f"Pseudotime analysis results for start node {start} saved in {output_dir}")

    return pseudotime