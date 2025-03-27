# utils/pseudotime_inference.py
"""
A user-friendly script for performing pseudotime analysis on a list of Cell objects
that already contain a specified representation (e.g., "raw_expression") in their additional_features.

Pipeline:
    1. Optionally randomly select a subset of cells.
    2. Extract the specified representation vector from each cell.
    3. Apply dimensionality reduction (PCA or UMAP).
    4. Cluster the reduced embeddings (KMeans or Agglomerative).
    5. Run pseudotime inference (e.g., Slingshot) on the reduced, clustered embeddings.
    6. Attach the inferred pseudotime to each Cell's additional_features.
    7. Save the updated list of Cells to disk.

Example usage:
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

from tic.model.train_gnn import set_seed
from tic.pseduotime.clustering import Clustering
from tic.pseduotime.dimensionality_reduction import DimensionalityReduction
from tic.pseduotime.pseudotime import SlingshotMethod

def parse_arguments():
    """
    Parse command line arguments for pseudotime analysis.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Pseudotime Analysis Script")
    # I/O paths
    parser.add_argument("--cells_input", type=str, required=True,
                        help="Path to the input .pt file containing a list of Cell objects.")
    parser.add_argument("--cells_output", type=str, required=True,
                        help="Path to save the updated Cell objects with pseudotime attached.")
    # Representation configuration
    parser.add_argument("--representation_key", type=str, default="raw_expression",
                        choices=['raw_expression','neighbor_composition','nn_embedding'],
                        help="Key in cell.additional_features that holds the representation vector. Default: raw_expression")
    # Dimensionality reduction configuration
    parser.add_argument("--dr_method", type=str, default="PCA",
                        choices=["PCA", "UMAP"],
                        help="Dimensionality reduction method. Default: PCA")
    parser.add_argument("--n_components", type=int, default=2,
                        help="Number of components for dimensionality reduction. Default: 2")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random state for reproducibility. Default: 42")
    # Clustering configuration
    parser.add_argument("--cluster_method", type=str, default="kmeans",
                        choices=["kmeans", "agg"],
                        help="Clustering method: 'kmeans' or 'agg'. Default: kmeans")
    parser.add_argument("--n_clusters", type=int, default=2,
                        help="Number of clusters to form. Default: 2")
    # Pseudotime configuration
    parser.add_argument("--start_node", type=int, default=None,
                        help="Starting node for Slingshot pseudotime. Default: None (auto-detect).")
    # Subset configuration (optional)
    parser.add_argument("--num_cells", type=int, default=None,
                        help="Randomly select this many cells for analysis. If None, use all. Default: None")
    # Plot output (optional)
    parser.add_argument("--output_dir", type=str, default="./pseudotime_plots",
                        help="Directory to save Slingshot plots. Default: ./pseudotime_plots")
    return parser.parse_args()

def load_cells(file_path: str) -> list:
    """
    Load the list of Cell objects from a .pt file.

    Args:
        file_path (str): Path to the input .pt file.

    Returns:
        list: List of Cell objects.
    
    Raises:
        FileNotFoundError: If the input file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file '{file_path}' not found.")
    cells = torch.load(file_path)
    print(f"[Info] Loaded {len(cells)} cells from {file_path}")
    return cells

def select_subset(cells: list, num_cells: int = None) -> list:
    """
    Randomly select a subset of cells if num_cells is specified.

    Args:
        cells (list): List of Cell objects.
        num_cells (int, optional): Number of cells to select. Defaults to None.

    Returns:
        list: Subset of Cell objects (or original list if num_cells is None).
    """
    if num_cells is not None and num_cells < len(cells):
        idx = np.random.choice(len(cells), num_cells, replace=False)
        cells = [cells[i] for i in idx]
        print(f"[Info] Randomly selected {num_cells} cells for analysis.")
    return cells

def extract_embeddings(cells: list, key: str) -> (np.ndarray, list): # type: ignore
    """
    Extract representation embeddings from cells using the given key.

    Args:
        cells (list): List of Cell objects.
        key (str): Key in cell.additional_features containing the representation.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Array of extracted embeddings.
            - list: List of cells with valid embeddings.
    """
    embeddings = []
    valid_cells = []
    for cell in cells:
        rep = cell.get_feature(key)
        if rep is None:
            continue
        embeddings.append(rep)
        valid_cells.append(cell)
    embeddings = np.array(embeddings)
    print(f"[Info] Extracted embeddings for {len(valid_cells)} cells using key '{key}'.")
    return embeddings, valid_cells

def attach_reduced_embedding(cells: list, reduced_emb: np.ndarray) -> None:
    """
    Attach the reduced embedding to each cell.

    Args:
        cells (list): List of Cell objects.
        reduced_emb (np.ndarray): Reduced embedding array.
    """
    for cell, emb in zip(cells, reduced_emb):
        cell.add_feature("embedding", emb)

def attach_cluster_labels(cells: list, cluster_labels: np.ndarray) -> None:
    """
    Attach cluster labels to each cell.

    Args:
        cells (list): List of Cell objects.
        cluster_labels (np.ndarray): Cluster labels.
    """
    for cell, label in zip(cells, cluster_labels):
        cell.add_feature("cluster_label", label)

def attach_pseudotime(cells: list, pseudotime_values: np.ndarray) -> None:
    """
    Attach pseudotime values to each cell.

    Args:
        cells (list): List of Cell objects.
        pseudotime_values (np.ndarray): Pseudotime values.
    """
    for cell, pt in zip(cells, pseudotime_values):
        cell.add_feature("pseudotime", float(pt))

def run_pseudotime_analysis(args) -> list:
    """
    Run the pseudotime analysis pipeline.

    The pipeline includes:
        1. Loading cells.
        2. (Optional) Selecting a subset of cells.
        3. Extracting embeddings.
        4. Dimensionality reduction.
        5. Clustering.
        6. Pseudotime inference.
        7. Attaching pseudotime to cells and saving the updated cells.

    Args:
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        list: Updated list of Cell objects with pseudotime attached.
    """
    set_seed(args.random_state)
    
    # Load cells from file
    cells = load_cells(args.cells_input)
    cells = select_subset(cells, args.num_cells)
    
    # Extract embeddings
    embeddings, valid_cells = extract_embeddings(cells, args.representation_key)
    
    # Dimensionality reduction
    dr = DimensionalityReduction(method=args.dr_method,
                                 n_components=args.n_components,
                                 random_state=args.random_state)
    reduced_emb = dr.reduce(embeddings)
    print(f"[Info] Reduced embeddings shape: {reduced_emb.shape}")
    attach_reduced_embedding(valid_cells, reduced_emb)
    
    # Clustering
    clusterer = Clustering(method=args.cluster_method, n_clusters=args.n_clusters)
    cluster_labels = clusterer.cluster(reduced_emb)
    print(f"[Info] Cluster labels: {np.unique(cluster_labels)}")
    attach_cluster_labels(valid_cells, cluster_labels)
    
    # Pseudotime inference
    pseudotime_method = SlingshotMethod(start_node=args.start_node)
    pseudotime_values = pseudotime_method.analyze(cluster_labels, reduced_emb, output_dir=args.output_dir)
    attach_pseudotime(valid_cells, pseudotime_values)
    print(f"[Info] Pseudotime range: [{pseudotime_values.min():.2f}, {pseudotime_values.max():.2f}]")
    
    # Save updated cells
    torch.save(valid_cells, args.cells_output)
    print(f"[Info] Saved updated cells with pseudotime to {args.cells_output}")
    
    return valid_cells

def main():
    """
    Main function to parse arguments and run the pseudotime analysis pipeline.
    """
    args = parse_arguments()
    run_pseudotime_analysis(args)

if __name__ == "__main__":
    main()