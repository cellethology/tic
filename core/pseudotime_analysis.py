# -*- coding: utf-8 -*-
"""
Created on Monday Dec 23 21:58 2024

@author: Jiahao Zhang
@Description: Pseudotime analysis for tumor embeddings using UMAP, KNN clustering, and Slingshot algorithm
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyslingshot import Slingshot
from anndata import AnnData

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

def visualize_expression_vs_pseudotime(
    raw_data_root, 
    pseudotime_csv_path, 
    selected_biomarkers=["CD14", "aSMA", "PanCK"], 
    num_bins=200, 
    use_bins=True
):
    """
    Visualize biomarker expression across pseudotime, optionally with binning.

    Parameters:
        raw_data_root (str): Root directory where the expression data files are located.
        pseudotime_csv_path (str): Path to the pseudotime CSV file.
        selected_biomarkers (list): List of biomarkers to visualize.
        num_bins (int): Number of bins to divide pseudotime into (if use_bins is True).
        use_bins (bool): Whether to use pseudotime binning.

    Returns:
        None: Displays a plot of biomarker expression vs pseudotime.
    """
    # Read the pseudotime data
    pseudotime_df = pd.read_csv(pseudotime_csv_path)

    # Add pseudotime bins if required
    if use_bins:
        pseudotime_df['pseudotime_bin'] = pd.qcut(
            pseudotime_df['pseudotime'], 
            num_bins, 
            labels=False, 
            duplicates="drop"
        )
    else:
        pseudotime_df['pseudotime_bin'] = pseudotime_df['pseudotime']

    # Initialize a dictionary to store aggregated expression levels
    binned_expression = {"pseudotime_bin": []}

    # Iterate through unique region IDs and process expression data
    for region_id in pseudotime_df['region_id'].unique():
        # Load the corresponding raw expression file
        expression_file_path = os.path.join(raw_data_root, f"{region_id}.expression.csv")
        if not os.path.exists(expression_file_path):
            print(f"Expression file for region_id {region_id} not found, skipping.")
            continue

        expression_df = pd.read_csv(expression_file_path)
        expression_df = expression_df.set_index(expression_df.columns[0])  # Set CELL_ID as the index

        # Normalize each biomarker
        expression_df = (expression_df - expression_df.min()) / (expression_df.max() - expression_df.min())

        # Merge expression data with pseudotime data
        merged_df = pseudotime_df[pseudotime_df['region_id'] == region_id].merge(
            expression_df, left_on='center_node_idx', right_index=True, how='inner'
        )

        # Group by pseudotime_bin and calculate the mean expression levels
        grouped = merged_df.groupby('pseudotime_bin').mean(numeric_only=True)
        for biomarker in expression_df.columns:
            if biomarker not in binned_expression:
                binned_expression[biomarker] = []
            binned_expression[biomarker].extend(grouped[biomarker].tolist())

        binned_expression["pseudotime_bin"].extend(grouped.index.tolist())

    # Convert to DataFrame for visualization
    binned_expression_df = pd.DataFrame(binned_expression).dropna()

    # Plot pseudotime vs expression levels
    plt.figure(figsize=(12, 6))
    for biomarker in selected_biomarkers:
        if biomarker in binned_expression_df.columns:
            plt.plot(binned_expression_df['pseudotime_bin'], binned_expression_df[biomarker], label=biomarker)

    plt.xlabel('Pseudotime (binned)' if use_bins else 'Pseudotime')
    plt.ylabel('Normalized Average Expression Level')
    plt.title('Pseudotime vs Normalized Biomarker Expression Levels')
    plt.legend()
    plt.show()

def generate_composition_vector_df(dataset, cell_type, n_samples=1000, normalize=True):
    """
    Generate a DataFrame with composition vectors as embeddings for pseudotime analysis.

    Parameters:
        dataset: CellularGraphDataset object containing the graph data.
        cell_type (int): Target cell type as the center node.
        n_samples (int): Number of subgraphs to sample.
        normalize (bool): Whether to normalize the composition vectors.

    Returns:
        pd.DataFrame: DataFrame with the following columns:
            - embedding: Composition vector.
            - region_id: Region identifier (inferred from dataset).
            - center_node_idx: Index of the center node (placeholder or actual).
            - cell_type: Target cell type.
    """
    from spacegm.embeddings_analysis import get_composition_vector, sample_subgraphs_by_cell_type

    # Sample subgraphs with the specified cell type
    reference_subgraph_list = sample_subgraphs_by_cell_type(dataset, cell_type=cell_type, n_samples=n_samples)
    
    # Extract composition vectors
    ref_composition_vectors = [
        get_composition_vector(data, n_cell_types=len(dataset.cell_type_mapping))
        for data in reference_subgraph_list
    ]

    # Normalize composition vectors if required
    if normalize:
        ref_composition_vectors = [
            (vec - np.min(vec)) / (np.max(vec) - np.min(vec) + 1e-8) for vec in ref_composition_vectors
        ]

    # Build DataFrame
    data = []
    for idx, (vec, subgraph) in enumerate(zip(ref_composition_vectors, reference_subgraph_list)):
        data.append({
            "embedding": vec,
            "region_id": subgraph.region_id if hasattr(subgraph, "region_id") else f"region_{idx}",
            "center_node_idx": subgraph.original_center_node if hasattr(subgraph, "original_center_node") else idx,
            "cell_type": cell_type
        })
    
    return pd.DataFrame(data)

def aggregate_biomarker_by_pseudotime(sampled_subgraphs, biomarkers, num_bins=200, use_bins=True):
    """
    Aggregate biomarker data by pseudotime.

    Args:
        sampled_subgraphs (list): List of sampled subgraphs with pseudotime and biomarker data.
        biomarkers (list): List of biomarker keys to aggregate.
        num_bins (int): Number of bins for pseudotime (if use_bins is True).
        use_bins (bool): Whether to bin pseudotime values.

    Returns:
        dict: Aggregated biomarker data by pseudotime.
    """
    biomarker_data = {biomarker: [] for biomarker in biomarkers}
    pseudotime_values = [subgraph["pseudotime"] for subgraph in sampled_subgraphs]

    # Bin pseudotime if required
    if use_bins:
        bins = np.linspace(min(pseudotime_values), max(pseudotime_values), num_bins)
        bin_indices = np.digitize(pseudotime_values, bins)
    else:
        bin_indices = np.array(pseudotime_values)

    # Aggregate biomarker data
    for subgraph, bin_idx in zip(sampled_subgraphs, bin_indices):
        for biomarker in biomarkers:
            biomarker_value = subgraph["node_info"].get("biomarker_expression", {}).get(biomarker, np.nan)
            biomarker_data[biomarker].append((bin_idx, biomarker_value))

    # Compute average biomarker values per bin
    aggregated_data = {}
    for biomarker, values in biomarker_data.items():
        values = pd.DataFrame(values, columns=["bin", "value"]).dropna()
        aggregated = values.groupby("bin")["value"].mean().reset_index()
        aggregated_data[biomarker] = aggregated

    return aggregated_data

# Biomarker Normalization
def normalize_biomarker_values(aggregated_data):
    """
    Normalize biomarker values for better visualization.

    Args:
        aggregated_data (dict): Aggregated biomarker data by pseudotime.

    Returns:
        dict: Normalized biomarker data by pseudotime.
    """
    normalized_data = {}
    for biomarker, data in aggregated_data.items():
        values = data["value"]
        min_val = values.min()
        max_val = values.max()
        normalized_values = (values - min_val) / (max_val - min_val)
        normalized_data[biomarker] = pd.DataFrame({
            "bin": data["bin"],
            "value": normalized_values
        })
    return normalized_data

def smooth_biomarker_values(aggregated_data, window_size=5):
    """
    Smooth biomarker values using a rolling average for better visualization.

    Args:
        aggregated_data (dict): Aggregated biomarker data by pseudotime.
        window_size (int): Window size for smoothing.

    Returns:
        dict: Smoothed biomarker data by pseudotime.
    """
    smoothed_data = {}
    for biomarker, data in aggregated_data.items():
        values = data["value"].rolling(window=window_size, min_periods=1).mean()
        smoothed_data[biomarker] = pd.DataFrame({
            "bin": data["bin"],
            "value": values
        })
    return smoothed_data

def plot_biomarker_vs_pseudotime(aggregated_data, output_dir=None, method=None, transform=None, use_bins=True):
    """
    Plot biomarker expression across pseudotime.

    Args:
        aggregated_data (dict): Aggregated biomarker data by pseudotime.
        output_dir (str, optional): Directory to save the output PNG file.
        method (str, optional): Method for pseudotime (used in labels).
        transform (str, optional): Transformation to apply ('normalize', 'smooth', etc.).
        use_bins (bool): Whether pseudotime is binned.
    """
    if transform == "normalize":
        aggregated_data = normalize_biomarker_values(aggregated_data)
    elif transform == "smooth":
        aggregated_data = smooth_biomarker_values(aggregated_data)

    plt.figure(figsize=(12, 6))
    for biomarker, data in aggregated_data.items():
        plt.plot(data["bin"], data["value"], label=biomarker)

    # Use a default value for method if it's None
    method_label = method + "Pseudotime" if method is not None else "Pseudotime"
    plt.xlabel(f"{method_label} (binned)" if use_bins else f"{method_label}")
    plt.ylabel("Normalized Average Expression Level" if transform == "normalize" else "Smoothed Expression Level")
    plt.title("Pseudotime vs Biomarker Expression Levels")
    plt.legend()
    plt.show()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "biomarker_vs_pseudotime.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Biomarker trends saved to {plot_path}")

