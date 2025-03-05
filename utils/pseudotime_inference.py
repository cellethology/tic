"""
pseudotime_inference.py

This script performs pseudotime analysis on a list of Cell objects that already
contain a chosen representation (stored in their additional_features, e.g. "raw_expression").
The pipeline is as follows:
    1. Extract the specified representation vector from each cell.
    2. Apply dimensionality reduction (e.g., UMAP or PCA) to obtain a low-dimensional embedding.
    3. Cluster the reduced embeddings (e.g., using KMeans).
    4. Run pseudotime inference (e.g., using Slingshot) on the clustered, reduced embeddings.
    5. Attach the inferred pseudotime value to each Cell's additional_features.
    6. Save the updated list of Cells to disk.

Users can modify the parameters (representation, reduction method, clustering method, etc.)
by editing the variables below.
"""

import os
import torch
import numpy as np

from core.pseduotime.clustering import Clustering
from core.pseduotime.dimensionality_reduction import DimensionalityReduction
from core.pseduotime.pseudotime import SlingshotMethod



# Configuration
# Specify which representation key to use (must be present in cell.additional_features)
# For example, "raw_expression", "neighbor_composition", or "nn_embedding".
REPRESENTATION_KEY = "raw_expression"  # change as needed

# Dimensionality reduction parameters
DR_METHOD = "PCA"   # or "PCA"
N_COMPONENTS = 2
RANDOM_STATE = 42

# Clustering parameters
CLUSTER_METHOD = "kmeans"  # or "agg"
N_CLUSTERS = 2

# Pseudotime parameters
# For SlingshotMethod, you may specify a starting node if needed (None means auto-detect)
START_NODE = 1

# File path for the input cells (assumed to be a .pt file saved previously)
CELLS_INPUT_PATH = "/Users/zhangjiahao/Project/tic/data/example/center_cells.pt"   # This file should contain a list of Cell objects with representations.

# File path to save the updated cells with pseudotime attached.
CELLS_OUTPUT_PATH = "/Users/zhangjiahao/Project/tic/data/example/cells_with_pseudotime.pt"

def main():
    # 1. Load the list of Cell objects
    if not os.path.exists(CELLS_INPUT_PATH):
        raise FileNotFoundError(f"Input file {CELLS_INPUT_PATH} not found.")
    cells = torch.load(CELLS_INPUT_PATH)
    print(f"Loaded {len(cells)} cells for pseudotime analysis.")

    # 2. Extract the chosen representation from each cell.
    #    We assume each cell.additional_features[REPRESENTATION_KEY] is a 1D numpy array.
    embeddings = []
    valid_cells = []
    for cell in cells:
        rep = cell.get_feature(REPRESENTATION_KEY)
        if rep is None:
            print(f"Warning: Cell {cell.cell_id} has no representation '{REPRESENTATION_KEY}'; skipping.")
            continue
        embeddings.append(rep)
        valid_cells.append(cell)
    embeddings = np.array(embeddings)
    print(f"Extracted embeddings for {len(valid_cells)} cells.")

    # 3. Dimensionality Reduction: reduce embeddings to 2D for pseudotime analysis.
    dim_reducer = DimensionalityReduction(method=DR_METHOD, n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    reduced_emb = dim_reducer.reduce(embeddings)
    print(f"Reduced embeddings shape: {reduced_emb.shape}")

    # Attach the reduced embedding to each valid cell.
    for cell, emb in zip(valid_cells, reduced_emb):
        cell.add_feature("embedding", emb)

    # 4. Clustering: cluster the reduced embeddings.
    clusterer = Clustering(method=CLUSTER_METHOD, n_clusters=N_CLUSTERS)
    cluster_labels = clusterer.cluster(reduced_emb)
    print(f"Cluster labels: {np.unique(cluster_labels)}")
    
    for cell, label in zip(valid_cells, cluster_labels):
        cell.add_feature("cluster_label", label)

    # 5. Pseudotime inference: run the Slingshot method.
    pseudotime_method = SlingshotMethod(start_node=START_NODE)
    # The Slingshot method expects 2D embeddings and cluster labels.
    pseudotime_values = pseudotime_method.analyze(cluster_labels, reduced_emb, output_dir="./pseudotime_plots")
    print(f"Inferred pseudotime range: {pseudotime_values.min()} - {pseudotime_values.max()}")
    
    for cell, pt in zip(valid_cells, pseudotime_values):
        cell.add_feature("pseudotime", float(pt))
    
    # 6. Save the updated cells to disk.
    torch.save(valid_cells, CELLS_OUTPUT_PATH)
    print(f"Saved updated cells with pseudotime to {CELLS_OUTPUT_PATH}")

if __name__ == "__main__":
    main()