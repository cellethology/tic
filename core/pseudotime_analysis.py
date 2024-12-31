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

def aggregate_biomarker_by_pseudotime_with_overlap(sampled_subgraphs, biomarkers, num_bins=200, overlap=0.2, use_bins=True):
    """
    Aggregate biomarker data by pseudotime with optional overlapping bins.

    Args:
        sampled_subgraphs (list): List of sampled subgraphs with pseudotime and biomarker data.
        biomarkers (list): List of biomarker keys to aggregate.
        num_bins (int): Number of bins for pseudotime (if use_bins is True).
        overlap (float): Proportion of overlap between bins (0.0 to 1.0). Defaults to 0.2 (20% overlap).
        use_bins (bool): Whether to bin pseudotime values.

    Returns:
        dict: Aggregated biomarker data by pseudotime.
    """
    if overlap < 0 or overlap >= 1:
        raise ValueError("Overlap must be between 0 and 1 (exclusive).")

    biomarker_data = {biomarker: [] for biomarker in biomarkers}
    pseudotime_values = [subgraph["pseudotime"] for subgraph in sampled_subgraphs]

    # Calculate bin edges and apply overlap if required
    if use_bins:
        min_pt, max_pt = min(pseudotime_values), max(pseudotime_values)
        bin_width = (max_pt - min_pt) / num_bins
        step_size = bin_width * (1 - overlap)  # Overlapping step size
        bin_edges = np.arange(min_pt, max_pt + step_size, step_size)
    else:
        bin_edges = np.unique(pseudotime_values)  # No binning, use unique pseudotime values

    # Aggregate biomarker data into bins
    for subgraph in sampled_subgraphs:
        pt_value = subgraph["pseudotime"]
        for i in range(len(bin_edges) - 1):
            bin_start, bin_end = bin_edges[i], bin_edges[i + 1]
            if bin_start <= pt_value < bin_end:
                for biomarker in biomarkers:
                    biomarker_value = subgraph["node_info"].get("biomarker_expression", {}).get(biomarker, np.nan)
                    biomarker_data[biomarker].append((i, biomarker_value))
                break

    # Compute average biomarker values per bin
    aggregated_data = {}
    for biomarker, values in biomarker_data.items():
        values = pd.DataFrame(values, columns=["bin", "value"]).dropna()
        aggregated = values.groupby("bin")["value"].mean().reset_index()
        aggregated["bin_center"] = aggregated["bin"].apply(lambda b: (bin_edges[b] + bin_edges[b + 1]) / 2)
        aggregated_data[biomarker] = aggregated

    return aggregated_data

def aggregate_data_by_pseudotime(
    sampled_subgraphs,
    pseudotime,
    feature_extractor,
    feature_keys,
    num_bins=100,
    overlap=0.2,
    use_bins=True
):
    """
    Generalized function to aggregate features by pseudotime with optional overlapping bins.

    Args:
        sampled_subgraphs (list): List of sampled subgraphs (dict).
        pseudotime (list): Pseudotime values for subgraphs.
        feature_extractor (callable): Function to extract features from a subgraph.
        feature_keys (list): List of feature keys to aggregate (e.g., biomarkers or cell types).
        num_bins (int): Number of bins for pseudotime (if use_bins is True).
        overlap (float): Proportion of overlap between bins (0.0 to 1.0). Defaults to 0.2 (20% overlap).
        use_bins (bool): Whether to bin pseudotime values.

    Returns:
        pd.DataFrame: Aggregated data with pseudotime or bin centers as the index and features as columns.
    """
    if overlap < 0 or overlap >= 1:
        raise ValueError("Overlap must be between 0 and 1 (exclusive).")

    # Extract feature values
    feature_data = {key: [] for key in feature_keys}
    pseudotime_values = pseudotime

    for subgraph, pt in zip(sampled_subgraphs, pseudotime):
        features = feature_extractor(subgraph)
        for key in feature_keys:
            feature_data[key].append({"pseudotime": pt, "value": features.get(key, np.nan)})

    # Calculate bin edges and apply overlap if required
    if use_bins:
        min_pt, max_pt = min(pseudotime_values), max(pseudotime_values)
        bin_width = (max_pt - min_pt) / num_bins
        step_size = bin_width * (1 - overlap)  # Overlapping step size
        bin_edges = np.arange(min_pt, max_pt + step_size, step_size)
    else:
        bin_edges = np.unique(pseudotime_values)  # No binning, use unique pseudotime values

    # Aggregate feature data into bins
    all_aggregated = []
    for i in range(len(bin_edges) - 1):
        bin_start, bin_end = bin_edges[i], bin_edges[i + 1]
        bin_center = (bin_start + bin_end) / 2

        bin_data = {}
        for key, values in feature_data.items():
            values_df = pd.DataFrame(values).dropna()
            bin_values = values_df[
                (values_df["pseudotime"] >= bin_start) & (values_df["pseudotime"] < bin_end)
            ]
            avg_value = bin_values["value"].mean() if not bin_values.empty else np.nan
            bin_data[key] = avg_value

        # Add bin_center as index
        bin_data["bin_center"] = bin_center
        all_aggregated.append(pd.DataFrame([bin_data]).set_index("bin_center"))

    # Combine all aggregated data
    aggregated_df = pd.concat(all_aggregated, axis=0)  
    return aggregated_df.reset_index()
