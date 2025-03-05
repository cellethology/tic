import os
import time
import yaml
from datetime import datetime
from argparse import Namespace

import streamlit as st
import torch
import matplotlib.pyplot as plt

# Import constants and functions from our modules
from core.constant import ALL_BIOMARKERS
from utils.extract_representation import extract_center_cells, save_center_cells
from utils.pseudotime_inference import run_pseudotime_analysis
from utils.plot import plot_pseudotime_vs_feature, my_y_transform, moving_average, normalize

# --- UI: Parameter Setup ---
st.title("Pseudotime Inference Pipeline")

st.sidebar.header("Set Pipeline Parameters")

# Part A: Paths & Files
dataset_root = st.sidebar.text_input("Dataset Root", "data/example")
center_cells_file = st.sidebar.text_input("Center Cells File", "center_cells.pt")
exp_dir = st.sidebar.text_input("Experiment Directory", "results/experiment")

# Part B: Pseudotime Inference Parameters
# Part B: Pseudotime Inference Parameters
representation_key = st.sidebar.selectbox(
    "Representation Key", 
    options=["raw_expression", "neighbor_composition", "nn_embedding"],
    index=0
)
dr_method = st.sidebar.selectbox("Dimensionality Reduction Method", options=["PCA", "UMAP"], index=0)
n_components = st.sidebar.number_input("Number of Components", min_value=1, value=2, step=1)
cluster_method = st.sidebar.selectbox("Clustering Method", options=["kmeans", "agg"], index=0)
n_clusters = int(st.sidebar.number_input("Number of Clusters", min_value=1, value=2, step=1))

# Optionally specify a start node; valid range is 0 to (n_clusters - 1)
specify_start_node = st.sidebar.checkbox("Specify Start Node?", value=False)
if specify_start_node:
    start_node = int(st.sidebar.number_input("Start Node (optional)", 
                                               min_value=0, 
                                               max_value=n_clusters - 1, 
                                               value=0, 
                                               step=1))
else:
    start_node = 0
num_cells = st.sidebar.number_input("Number of Cells (optional)", min_value=1, value=1000, step=1)

# Part C: Plot Parameters
x_bins = st.sidebar.number_input("Number of Bins", min_value=1, value=100, step=1)
# Use centralized biomarker options from core.constant.ALL_BIOMARKERS
biomarkers = st.sidebar.multiselect("Biomarkers", options=ALL_BIOMARKERS, default=["PanCK", "aSMA"])
neighbor_types = st.sidebar.multiselect(
    "Neighbor Types", 
    options=["Immune", "Tumor", "Stromal", "Vascular"],
    default=["Immune", "Tumor", "Stromal", "Vascular"]
)

# Part D: Transformation Function Selection
transform_option = st.sidebar.selectbox(
    "Transformation Function", 
    options=["none", "moving_average", "normalize", "my_y_transform"],
    index=3
)

# Map transform option to function
transform_mapping = {
    "none": None,
    "moving_average": moving_average,
    "normalize": normalize,
    "my_y_transform": my_y_transform
}
selected_transform = transform_mapping[transform_option]

# Part E: Plot Choice
plot_choice = st.sidebar.radio("Plot Option", options=["Biomarkers", "Neighbor Types", "Both"], index=2)

# --- Run Pipeline Button ---
if st.sidebar.button("Run Pipeline"):
    st.write("Starting pipeline...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Create a unique experiment output directory with a timestamp.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_exp_dir = os.path.join(exp_dir, f"experiment_{timestamp}")
    os.makedirs(current_exp_dir, exist_ok=True)

    # Save experiment configuration to a YAML file.
    config = {
        "dataset_root": dataset_root,
        "center_cells_file": center_cells_file,
        "exp_dir": current_exp_dir,
        "representation_key": representation_key,
        "dr_method": dr_method,
        "n_components": n_components,
        "cluster_method": cluster_method,
        "n_clusters": n_clusters,
        "start_node": start_node,
        "num_cells": num_cells if num_cells > 0 else None,
        "x_bins": x_bins,
        "biomarkers": biomarkers,
        "neighbor_types": neighbor_types,
        "transform": transform_option
    }
    config_path = os.path.join(current_exp_dir, "experiment_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    st.write("Configuration saved to:", config_path)
    progress_bar.progress(10)
    status_text.text("Step 1/6: Checking center cells file...")

    # Construct the center cells file path.
    center_cells_path = os.path.join(dataset_root, center_cells_file)

    # If the center cells file does not exist, run extraction.
    if not os.path.exists(center_cells_path):
        status_text.text("Extracting center cells representations...")
        center_cells = extract_center_cells(root=dataset_root)
        save_center_cells(center_cells, center_cells_path)
    else:
        status_text.text("Using existing center cells file...")
    progress_bar.progress(20)

    # Prepare arguments for pseudotime inference.
    pseudo_args = Namespace(
        cells_input=center_cells_path,
        cells_output=os.path.join(current_exp_dir, "cells_with_pseudotime.pt"),
        representation_key=representation_key,
        dr_method=dr_method,
        n_components=n_components,
        random_state=42,  # Fixed seed for reproducibility
        cluster_method=cluster_method,
        n_clusters=n_clusters,
        start_node=start_node,
        num_cells=num_cells if num_cells > 0 else None,
        output_dir=current_exp_dir  # This directory is used for pseudotime plots!
    )
    status_text.text("Step 2/6: Running pseudotime inference...")
    run_pseudotime_analysis(pseudo_args)
    progress_bar.progress(40)

    # Load the updated cells with pseudotime.
    cells_with_pt = torch.load(pseudo_args.cells_output)
    status_text.text("Step 3/6: Pseudotime inference completed.")
    progress_bar.progress(50)

    # Generate plot(s) for biomarkers/neighbor types.
    status_text.text("Step 4/6: Generating plot(s)...")
    plot_paths = []
    if plot_choice in ["Biomarkers", "Both"]:
        biomarker_plot_path = os.path.join(current_exp_dir, "pseudo_vs_biomarkers.svg")
        plot_pseudotime_vs_feature(cells_with_pt,
                                   x_bins=x_bins,
                                   biomarkers=biomarkers,
                                   y_transform=selected_transform,
                                   save_path=biomarker_plot_path)
        plot_paths.append(biomarker_plot_path)
    if plot_choice in ["Neighbor Types", "Both"]:
        neighbor_plot_path = os.path.join(current_exp_dir, "pseudo_vs_neighbor.svg")
        plot_pseudotime_vs_feature(cells_with_pt,
                                   x_bins=x_bins,
                                   neighbor_types=neighbor_types,
                                   y_transform=selected_transform,
                                   save_path=neighbor_plot_path)
        plot_paths.append(neighbor_plot_path)
    progress_bar.progress(70)

    # Generate an additional plot: pseudotime distribution.
    status_text.text("Step 5/6: Generating pseudotime distribution plot...")
    pseudotime_values = [cell.get_feature("pseudotime") for cell in cells_with_pt if cell.get_feature("pseudotime") is not None]
    if pseudotime_values:
        fig, ax = plt.subplots()
        ax.hist(pseudotime_values, bins=50, color="blue", edgecolor="black", alpha=0.7)
        ax.set_title("Pseudotime Distribution")
        ax.set_xlabel("Pseudotime")
        ax.set_ylabel("Frequency")
        pseudotime_distribution_path = os.path.join(current_exp_dir, "pseudotime_distribution.svg")
        fig.savefig(pseudotime_distribution_path, bbox_inches="tight")
        plt.close(fig)
        plot_paths.append(pseudotime_distribution_path)
    progress_bar.progress(80)

    # Display pseudotime visualization plot saved by Slingshot.
    status_text.text("Step 6/6: Displaying pseudotime visualization plot...")
    pseudotime_vis_path = os.path.join(current_exp_dir, "pseudotime_visualization.svg")
    if os.path.exists(pseudotime_vis_path):
        plot_paths.append(pseudotime_vis_path)
    progress_bar.progress(100)
    status_text.text("Pipeline completed successfully.")
    st.success("Pipeline completed!")

    # --- Display the Output Plot(s) ---
    st.subheader("Output Plot(s)")
    for plot_path in plot_paths:
        if os.path.exists(plot_path):
            st.image(plot_path, caption=os.path.basename(plot_path))
        else:
            st.write(f"Plot not found: {plot_path}")