import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from argparse import Namespace

import streamlit as st

# Utility imports 
from tic.constant import ALL_BIOMARKERS
from utils.extract_representation import extract_center_cells, save_center_cells
from utils.pseudotime_inference import (
    run_pseudotime_analysis,
    select_subset,
    extract_embeddings,
    attach_reduced_embedding
)
from tic.pseduotime.clustering import Clustering
from tic.pseduotime.dimensionality_reduction import DimensionalityReduction
from utils.plot import (
    plot_pseudotime_vs_feature,
    moving_average,
    normalize,
    my_y_transform
)


def save_params_to_yaml(params_dict: dict, yaml_path: str) -> None:
    """
    Save a dictionary of parameters to a YAML file.

    Args:
        params_dict (dict): The dictionary of parameters to be saved.
        yaml_path (str): The path to the output YAML file.
    """
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(params_dict, f, allow_unicode=True)


# ---------------------------
# Streamlit App Starts Here
# ---------------------------
st.title("Step-by-Step Interactive Pseudotime Inference Pipeline")

# ----------------------------------------
# Step 1: General Parameter Setup
# ----------------------------------------
st.header("Step 1: General Parameters")

dataset_root = st.text_input("Dataset Root", "data/example")
center_cells_file = st.text_input("Center Cells Filename", "center_cells.pt")
exp_dir = st.text_input("Experiment Output Directory", "results/experiment")

# ----------------------------------------
# Step 2: Clustering Parameters & Execution
# ----------------------------------------
st.header("Step 2: Clustering Parameters")

representation_key = st.selectbox(
    "Representation Key",
    options=["raw_expression", "neighbor_composition"],  
    index=0
)
dr_method = st.selectbox("Dimensionality Reduction Method", options=["PCA", "UMAP"], index=0)
n_components = st.number_input("Number of Components", min_value=1, value=2, step=1)
cluster_method = st.selectbox("Clustering Method", options=["kmeans", "agg"], index=0)
n_clusters = int(st.number_input("Number of Clusters (n_clusters)", min_value=1, value=5, step=1))
num_cells = st.number_input("Number of Sampled Cells (optional)", min_value=1, value=1000, step=1)

# Button to run clustering
if st.button("Run Clustering"):
    st.write("Running clustering...")

    # Create a timestamped experiment folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_exp_dir = os.path.join(exp_dir, f"experiment_{timestamp}")
    os.makedirs(current_exp_dir, exist_ok=True)
    st.session_state["current_exp_dir"] = current_exp_dir

    # Save clustering parameters as YAML
    clustering_params = {
        "dataset_root": dataset_root,
        "center_cells_file": center_cells_file,
        "representation_key": representation_key,
        "dr_method": dr_method,
        "n_components": n_components,
        "cluster_method": cluster_method,
        "n_clusters": n_clusters,
        "num_cells": num_cells,
        "exp_dir": current_exp_dir,
    }
    clustering_yaml_path = os.path.join(current_exp_dir, "clustering_params.yaml")
    save_params_to_yaml(clustering_params, clustering_yaml_path)

    # Load or generate center_cells
    center_cells_path = os.path.join(dataset_root, center_cells_file)
    if not os.path.exists(center_cells_path):
        st.write("Extracting center cells...")
        center_cells = extract_center_cells(root=dataset_root)
        save_center_cells(center_cells, center_cells_path)
    else:
        st.write("Using existing center cells file...")
        center_cells = torch.load(center_cells_path)

    # Subsample cells if requested
    cells = center_cells
    if num_cells < len(cells):
        cells = select_subset(cells, num_cells)

    # Extract embeddings and reduce dimensions
    embeddings, valid_cells = extract_embeddings(cells, representation_key)
    dr = DimensionalityReduction(method=dr_method, n_components=n_components, random_state=42)
    reduced_emb = dr.reduce(embeddings)
    attach_reduced_embedding(valid_cells, reduced_emb)

    # Perform clustering
    clusterer = Clustering(method=cluster_method, n_clusters=n_clusters)
    cluster_labels = clusterer.cluster(reduced_emb)
    for cell, label in zip(valid_cells, cluster_labels):
        cell.add_feature("cluster_label", label)

    st.session_state["valid_cells"] = valid_cells
    st.session_state["reduced_emb"] = reduced_emb
    st.session_state["cluster_labels"] = cluster_labels

    # Retrieve cluster centers if available
    if cluster_method == "kmeans" and hasattr(clusterer, "cluster_centers_"):
        cluster_centers = clusterer.cluster_centers_
    else:
        cluster_centers = None

    # Visualization of clustering
    fig, ax = plt.subplots()
    sc = ax.scatter(
        reduced_emb[:, 0],
        reduced_emb[:, 1],
        c=cluster_labels,
        cmap="viridis",
        s=10
    )
    if cluster_centers is not None:
        ax.scatter(
            cluster_centers[:, 0],
            cluster_centers[:, 1],
            c="red",
            marker="x",
            s=100,
            label="Cluster Center"
        )
        for i, center in enumerate(cluster_centers):
            ax.text(
                center[0], center[1],
                f"Cluster {i}",
                color="red", fontsize=12, fontweight="bold"
            )

    handles, _ = sc.legend_elements(prop="colors")
    unique_labels = np.unique(cluster_labels)
    cluster_labels_sorted = sorted(unique_labels)
    legend_labels = [f"Cluster {label}" for label in cluster_labels_sorted]
    legend1 = ax.legend(handles, legend_labels, title="Clusters", loc="upper right")
    ax.add_artist(legend1)

    cluster_image_path = os.path.join(current_exp_dir, "cluster_result.svg")
    fig.savefig(cluster_image_path, bbox_inches="tight")
    st.write(f"Clustering figure saved to: {cluster_image_path}")
    st.pyplot(fig)

# ----------------------------------------
# Step 3: Pseudotime Inference
# ----------------------------------------
st.header("Step 3: Pseudotime Inference")

if "cluster_labels" in st.session_state:
    start_node = st.number_input(
        "Select the Start Node (cluster index)",
        min_value=0,
        max_value=n_clusters - 1,
        value=0,
        step=1
    )

    # Button: Run Pseudotime Inference
    if st.button("Run Pseudotime Inference"):
        current_exp_dir = st.session_state["current_exp_dir"]
        center_cells_path = os.path.join(dataset_root, center_cells_file)
        pseudo_args = Namespace(
            cells_input=center_cells_path,
            cells_output=os.path.join(current_exp_dir, "cells_with_pseudotime.pt"),
            representation_key=representation_key,
            dr_method=dr_method,
            n_components=n_components,
            random_state=42,
            cluster_method=cluster_method,
            n_clusters=n_clusters,
            start_node=start_node,
            num_cells=num_cells,
            output_dir=current_exp_dir,
        )

        # Save pseudotime parameters as YAML
        pseudo_params = {
            "cells_input": pseudo_args.cells_input,
            "cells_output": pseudo_args.cells_output,
            "representation_key": pseudo_args.representation_key,
            "dr_method": pseudo_args.dr_method,
            "n_components": pseudo_args.n_components,
            "cluster_method": pseudo_args.cluster_method,
            "n_clusters": pseudo_args.n_clusters,
            "start_node": pseudo_args.start_node,
            "num_cells": pseudo_args.num_cells,
            "output_dir": pseudo_args.output_dir,
        }
        pseudo_yaml_path = os.path.join(current_exp_dir, "pseudotime_params.yaml")
        save_params_to_yaml(pseudo_params, pseudo_yaml_path)

        # Run the pseudotime inference
        run_pseudotime_analysis(pseudo_args)
        st.write("Pseudotime inference completed.")

        # Load updated cells
        cells_with_pt = torch.load(pseudo_args.cells_output)
        st.session_state["cells_with_pt"] = cells_with_pt
        st.write(f"Loaded {len(cells_with_pt)} cells with pseudotime.")

        # Display the pseudotime visualization if it exists
        pseudotime_vis_path = os.path.join(current_exp_dir, "pseudotime_visualization.svg")
        if os.path.exists(pseudotime_vis_path):
            st.subheader("Pseudotime Visualization (e.g., Slingshot)")
            st.image(pseudotime_vis_path, caption="pseudotime_visualization.svg")
        else:
            st.info("pseudotime_visualization.svg not found. Check your pseudotime analysis output.")
else:
    st.warning("Please run Clustering first to obtain cluster labels.")

# ----------------------------------------
# Step 4: Visualization
# ----------------------------------------
st.header("Step 4: Visualization")

if "cells_with_pt" in st.session_state:
    x_bins = st.number_input("Number of Bins", min_value=1, value=100, step=1)
    biomarkers = st.multiselect(
        "Biomarkers",
        options=ALL_BIOMARKERS,
        default=["PanCK", "aSMA"]
    )
    neighbor_types = st.multiselect(
        "Neighbor Types",
        options=["Immune", "Tumor", "Stromal", "Vascular"],
        default=["Immune", "Tumor", "Stromal", "Vascular"]
    )
    transform_option = st.selectbox(
        "Transformation Function",
        options=["none", "moving_average", "normalize", "my_y_transform"],
        index=3
    )
    transform_mapping = {
        "none": None,
        "moving_average": moving_average,
        "normalize": normalize,
        "my_y_transform": my_y_transform,
    }
    selected_transform = transform_mapping[transform_option]

    if st.button("Generate Visualization"):
        cells_with_pt = st.session_state["cells_with_pt"]
        vis_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vis_subdir = os.path.join(st.session_state["current_exp_dir"], f"vis_{vis_timestamp}")
        os.makedirs(vis_subdir, exist_ok=True)

        # Save current visualization parameters as YAML
        visualization_params = {
            "x_bins": x_bins,
            "biomarkers": biomarkers,
            "neighbor_types": neighbor_types,
            "transform_option": transform_option,
            "vis_subdir": vis_subdir,
        }
        vis_yaml_path = os.path.join(vis_subdir, "visualization_params.yaml")
        save_params_to_yaml(visualization_params, vis_yaml_path)

        # 1) Biomarker curves
        biomarker_plot_path = os.path.join(vis_subdir, "pseudo_vs_biomarkers.svg")
        plot_pseudotime_vs_feature(
            cells_with_pt,
            x_bins=x_bins,
            biomarkers=biomarkers,
            y_transform=selected_transform,
            save_path=biomarker_plot_path
        )

        # 2) Neighbor types curves
        neighbor_plot_path = os.path.join(vis_subdir, "pseudo_vs_neighbor.svg")
        plot_pseudotime_vs_feature(
            cells_with_pt,
            x_bins=x_bins,
            neighbor_types=neighbor_types,
            y_transform=selected_transform,
            save_path=neighbor_plot_path
        )

        # 3) Pseudotime distribution
        pseudotime_values = [
            cell.get_feature("pseudotime")
            for cell in cells_with_pt
            if cell.get_feature("pseudotime") is not None
        ]
        if pseudotime_values:
            fig, ax = plt.subplots()
            ax.hist(pseudotime_values, bins=50, color="blue", edgecolor="black", alpha=0.7)
            ax.set_title("Pseudotime Distribution")
            ax.set_xlabel("Pseudotime")
            ax.set_ylabel("Frequency")
            pseudotime_distribution_path = os.path.join(vis_subdir, "pseudotime_distribution.svg")
            fig.savefig(pseudotime_distribution_path, bbox_inches="tight")
            plt.close(fig)

        # Display the generated images in Streamlit
        st.subheader("Visualization Results")
        for fname in os.listdir(vis_subdir):
            if fname.endswith(".svg") or fname.endswith(".png"):
                path = os.path.join(vis_subdir, fname)
                st.image(path, caption=fname)
else:
    st.warning("No pseudotime results found. Please complete the previous steps first.")