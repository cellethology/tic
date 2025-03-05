import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from argparse import Namespace

import streamlit as st

from core.constant import ALL_BIOMARKERS
from utils.extract_representation import extract_center_cells, save_center_cells
from utils.pseudotime_inference import (run_pseudotime_analysis, 
                                        select_subset, 
                                        extract_embeddings, 
                                        attach_reduced_embedding)
from core.pseduotime.clustering import Clustering
from core.pseduotime.dimensionality_reduction import DimensionalityReduction

# --- UI: Parameter Setup ---
st.title("Interactive Pseudotime Inference Pipeline")

# Sidebar: Basic parameters
dataset_root = st.sidebar.text_input("Dataset Root", "data/example")
center_cells_file = st.sidebar.text_input("Center Cells File", "center_cells.pt")
exp_dir = st.sidebar.text_input("Experiment Directory", "results/experiment")
representation_key = st.sidebar.selectbox("Representation Key", 
                                            options=["raw_expression", "neighbor_composition", "nn_embedding"],
                                            # options=["raw_expression", "neighbor_composition", "nn_embedding"], Not Support nn_embedding this version
                                            index=0)
dr_method = st.sidebar.selectbox("Dimensionality Reduction Method", options=["PCA", "UMAP"], index=0)
n_components = st.sidebar.number_input("Number of Components", min_value=1, value=2, step=1)
cluster_method = st.sidebar.selectbox("Clustering Method", options=["kmeans", "agg"], index=0)
n_clusters = int(st.sidebar.number_input("Number of Clusters", min_value=1, value=5, step=1))
num_cells = st.sidebar.number_input("Number of Cells (optional)", min_value=1, value=1000, step=1)

# Additional plot parameters
x_bins = st.sidebar.number_input("Number of Bins", min_value=1, value=100, step=1)
biomarkers = st.sidebar.multiselect("Biomarkers", options=ALL_BIOMARKERS, default=["PanCK", "aSMA"])
neighbor_types = st.sidebar.multiselect("Neighbor Types", 
                                        options=["Immune", "Tumor", "Stromal", "Vascular"],
                                        default=["Immune", "Tumor", "Stromal", "Vascular"])
transform_option = st.sidebar.selectbox("Transformation Function", 
                                          options=["none", "moving_average", "normalize", "my_y_transform"],
                                          index=3)
# Map transform option to function (assuming these are imported from utils.plot)
from utils.plot import moving_average, normalize, my_y_transform
transform_mapping = {
    "none": None,
    "moving_average": moving_average,
    "normalize": normalize,
    "my_y_transform": my_y_transform
}
selected_transform = transform_mapping[transform_option]

# Step 1: Run Clustering and Visualize Clusters
if st.sidebar.button("Run Clustering"):
    st.write("Running clustering...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_exp_dir = os.path.join(exp_dir, f"experiment_{timestamp}")
    os.makedirs(current_exp_dir, exist_ok=True)
    st.session_state["current_exp_dir"] = current_exp_dir

    center_cells_path = os.path.join(dataset_root, center_cells_file)
    if not os.path.exists(center_cells_path):
        st.write("Extracting center cells representations...")
        center_cells = extract_center_cells(root=dataset_root)
        save_center_cells(center_cells, center_cells_path)
    else:
        st.write("Using existing center cells file.")
        center_cells = torch.load(center_cells_path)

    # Optionally, select a subset
    cells = center_cells
    if num_cells < len(cells):
        cells = select_subset(cells, num_cells)

    # Extract embeddings from cells
    embeddings, valid_cells = extract_embeddings(cells, representation_key)
    dr = DimensionalityReduction(method=dr_method, n_components=n_components, random_state=42)
    reduced_emb = dr.reduce(embeddings)
    attach_reduced_embedding(valid_cells, reduced_emb)

    # Run clustering
    clusterer = Clustering(method=cluster_method, n_clusters=n_clusters)
    cluster_labels = clusterer.cluster(reduced_emb)
    for cell, label in zip(valid_cells, cluster_labels):
        cell.add_feature("cluster_label", label)
    st.session_state["valid_cells"] = valid_cells
    st.session_state["reduced_emb"] = reduced_emb
    st.session_state["cluster_labels"] = cluster_labels

    # If using KMeans, retrieve cluster centers
    if cluster_method == "kmeans" and hasattr(clusterer, "cluster_centers_"):
        cluster_centers = clusterer.cluster_centers_
    else:
        cluster_centers = None

    # Plot clustering result with cluster centers and labels
    fig, ax = plt.subplots()
    sc = ax.scatter(reduced_emb[:, 0], reduced_emb[:, 1], c=cluster_labels, cmap="viridis", s=10)

    if cluster_centers is not None:
        # Plot cluster centers as red "X" markers and label them
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c="red", marker="x", s=100, label="Cluster Center")
        for i, center in enumerate(cluster_centers):
            ax.text(center[0], center[1], f"Cluster {i}", color="red", fontsize=12, fontweight="bold")

    # Create a legend for the clusters based on the scatter plot colors
    # Use legend_elements to get handles corresponding to the cluster colors.
    handles, _ = sc.legend_elements(prop="colors")
    unique_labels = np.unique(cluster_labels)
    # Ensure labels are in sorted order for consistency
    cluster_labels_sorted = sorted(unique_labels)
    legend_labels = [f"Cluster {label}" for label in cluster_labels_sorted]
    legend1 = ax.legend(handles, legend_labels, title="Clusters", loc="upper right")
    ax.add_artist(legend1)

    # Save the cluster plot to a file in the output directory
    cluster_image_path = os.path.join(current_exp_dir, "cluster_result.svg")
    fig.savefig(cluster_image_path, bbox_inches="tight")
    st.write(f"Cluster plot saved to: {cluster_image_path}")

    # Display the figure in the Streamlit UI
    st.pyplot(fig)

# Step 2: Specify Start Node from Clusters
if "cluster_labels" in st.session_state:
    st.sidebar.header("Specify Start Node")
    # The valid start node is an integer from 0 to n_clusters - 1.
    start_node = st.sidebar.number_input("Select Start Node (cluster index)", 
                                           min_value=0, max_value=n_clusters - 1, 
                                           value=0, step=1)
    st.session_state["start_node"] = start_node

    # Option to run pseudotime inference
    if st.sidebar.button("Run Pseudotime Inference"):
        # Prepare arguments for pseudotime inference
        center_cells_path = os.path.join(dataset_root, center_cells_file)
        pseudo_args = Namespace(
            cells_input=center_cells_path,
            cells_output=os.path.join(st.session_state["current_exp_dir"], "cells_with_pseudotime.pt"),
            representation_key=representation_key,
            dr_method=dr_method,
            n_components=n_components,
            random_state=42,
            cluster_method=cluster_method,
            n_clusters=n_clusters,
            start_node=start_node,  # user-specified start node
            num_cells=num_cells,
            output_dir=st.session_state["current_exp_dir"]
        )
        from utils.pseudotime_inference import run_pseudotime_analysis
        run_pseudotime_analysis(pseudo_args)
        st.write("Pseudotime inference completed.")

        # Load updated cells and generate additional plots as before.
        cells_with_pt = torch.load(pseudo_args.cells_output)
        st.write(f"Loaded {len(cells_with_pt)} cells with pseudotime.")
        
        # Display pseudotime-related plots (biomarker, neighbor, distribution, etc.)
        plot_paths = []
        biomarker_plot_path = os.path.join(st.session_state["current_exp_dir"], "pseudo_vs_biomarkers.svg")
        from utils.plot import plot_pseudotime_vs_feature
        plot_pseudotime_vs_feature(cells_with_pt,
                                   x_bins=x_bins,
                                   biomarkers=biomarkers,
                                   y_transform=selected_transform,
                                   save_path=biomarker_plot_path)
        plot_paths.append(biomarker_plot_path)
        
        neighbor_plot_path = os.path.join(st.session_state["current_exp_dir"], "pseudo_vs_neighbor.svg")
        plot_pseudotime_vs_feature(cells_with_pt,
                                   x_bins=x_bins,
                                   neighbor_types=neighbor_types,
                                   y_transform=selected_transform,
                                   save_path=neighbor_plot_path)
        plot_paths.append(neighbor_plot_path)
        
        # Pseudotime distribution plot
        pseudotime_values = [cell.get_feature("pseudotime") for cell in cells_with_pt if cell.get_feature("pseudotime") is not None]
        if pseudotime_values:
            fig, ax = plt.subplots()
            ax.hist(pseudotime_values, bins=50, color="blue", edgecolor="black", alpha=0.7)
            ax.set_title("Pseudotime Distribution")
            ax.set_xlabel("Pseudotime")
            ax.set_ylabel("Frequency")
            pseudotime_distribution_path = os.path.join(st.session_state["current_exp_dir"], "pseudotime_distribution.svg")
            fig.savefig(pseudotime_distribution_path, bbox_inches="tight")
            plt.close(fig)
            plot_paths.append(pseudotime_distribution_path)
        
        # Also display the pseudotime visualization plot saved by Slingshot
        pseudotime_vis_path = os.path.join(st.session_state["current_exp_dir"], "pseudotime_visualization.svg")
        if os.path.exists(pseudotime_vis_path):
            plot_paths.append(pseudotime_vis_path)
        
        st.subheader("Output Plots")
        for path in plot_paths:
            if os.path.exists(path):
                st.image(path, caption=os.path.basename(path))
            else:
                st.write(f"Plot not found: {path}")