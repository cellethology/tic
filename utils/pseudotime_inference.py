"""
pseudotime_inference.py

A user-friendly script to perform pseudotime analysis on a list of Cell objects that
already contain a chosen representation (e.g. "raw_expression") in their additional_features.

Pipeline:
    1. Randomly select a subset of cells (optional).
    2. Extract the specified representation vector from each cell.
    3. Apply dimensionality reduction (PCA/UMAP).
    4. Cluster the reduced embeddings (KMeans/Agglomerative).
    5. Run pseudotime inference (e.g., Slingshot) on the reduced, clustered embeddings.
    6. Attach the inferred pseudotime to each Cell's additional_features.
    7. Save the updated list of Cells to disk.

Example Usage:
    python pseudotime_inference.py 
      --cells_input /path/to/center_cells.pt 
      --cells_output /path/to/cells_with_pseudotime.pt 
      --representation_key raw_expression 
      --dr_method PCA --n_components 2 
      --cluster_method kmeans --n_clusters 5 
      --start_node 1
      --num_cells 10000
"""

import os
import argparse
import numpy as np
import torch

from core.model.train_gnn import set_seed
from core.pseduotime.clustering import Clustering
from core.pseduotime.dimensionality_reduction import DimensionalityReduction
from core.pseduotime.pseudotime import SlingshotMethod

def parse_arguments():
    parser = argparse.ArgumentParser(description="Pseudotime Analysis Script")

    # I/O paths
    parser.add_argument("--cells_input", type=str, required=True,
                        help="Path to the input .pt file containing a list of Cell objects.")
    parser.add_argument("--cells_output", type=str, required=True,
                        help="Path to save the updated Cell objects with pseudotime attached.")

    # Representation config
    parser.add_argument("--representation_key", type=str, default="raw_expression", choices=['raw_expression','neighbor_composition','nn_embedding'],
                        help="Key in cell.additional_features that holds the representation vector. Default: raw_expression")

    # Dimension reduction config
    parser.add_argument("--dr_method", type=str, default="PCA",
                        choices=["PCA", "UMAP"],
                        help="Dimensionality reduction method. Default: PCA")
    parser.add_argument("--n_components", type=int, default=2,
                        help="Number of components for DR. Default: 2")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random state for reproducibility. Default: 42")

    # Clustering config
    parser.add_argument("--cluster_method", type=str, default="kmeans",
                        choices=["kmeans", "agg"],
                        help="Clustering method: 'kmeans' or 'agg'. Default: kmeans")
    parser.add_argument("--n_clusters", type=int, default=2,
                        help="Number of clusters to form. Default: 2")

    # Pseudotime config
    parser.add_argument("--start_node", type=int, default=None,
                        help="Starting node for Slingshot pseudotime. Default: None (auto-detect).")

    # Subset config
    parser.add_argument("--num_cells", type=int, default=None,
                        help="Randomly select this many cells for analysis. If None, use all. Default: None")

    # Plots / optional
    parser.add_argument("--output_dir", type=str, default="./pseudotime_plots",
                        help="Directory to save Slingshot plots. Default: ./pseudotime_plots")

    return parser.parse_args()

def main():
    args = parse_arguments()
    set_seed(args.random_state)

    # 1) Load the list of Cell objects
    if not os.path.exists(args.cells_input):
        raise FileNotFoundError(f"Input file '{args.cells_input}' not found.")
    cells = torch.load(args.cells_input)
    print(f"[Info] Loaded {len(cells)} cells from {args.cells_input}")

    # Optionally randomly select a subset
    if args.num_cells is not None and args.num_cells < len(cells):
        idx = np.random.choice(len(cells), args.num_cells, replace=False)
        cells = [cells[i] for i in idx]
        print(f"[Info] Randomly selected {args.num_cells} cells from the total.")

    # 2) Extract embeddings
    embeddings = []
    valid_cells = []
    for cell in cells:
        rep = cell.get_feature(args.representation_key)
        if rep is None:
            # skip cells that don't have the representation
            continue
        embeddings.append(rep)
        valid_cells.append(cell)
    embeddings = np.array(embeddings)
    print(f"[Info] Extracted embeddings for {len(valid_cells)} cells using key '{args.representation_key}'.")

    # 3) Dimensionality reduction
    dim_reducer = DimensionalityReduction(method=args.dr_method, 
                                          n_components=args.n_components,
                                          random_state=args.random_state)
    reduced_emb = dim_reducer.reduce(embeddings)
    print(f"[Info] Reduced embeddings to shape: {reduced_emb.shape}")

    # Attach the reduced embedding to each valid cell
    for cell, emb in zip(valid_cells, reduced_emb):
        cell.add_feature("embedding", emb)

    # 4) Clustering
    clusterer = Clustering(method=args.cluster_method, n_clusters=args.n_clusters)
    cluster_labels = clusterer.cluster(reduced_emb)
    print(f"[Info] Unique cluster labels found: {np.unique(cluster_labels)}")

    for cell, label in zip(valid_cells, cluster_labels):
        cell.add_feature("cluster_label", label)

    # 5) Pseudotime
    pseudotime_method = SlingshotMethod(start_node=args.start_node)
    pseudotime_values = pseudotime_method.analyze(cluster_labels, reduced_emb, output_dir=args.output_dir)

    # Attach pseudotime to each cell
    for cell, pt in zip(valid_cells, pseudotime_values):
        cell.add_feature("pseudotime", float(pt))

    print(f"[Info] Pseudotime range: [{pseudotime_values.min():.2f}, {pseudotime_values.max():.2f}]")

    # 6) Save updated cells
    torch.save(valid_cells, args.cells_output)
    print(f"[Info] Saved updated cells with pseudotime to {args.cells_output}")

if __name__ == "__main__":
    main()