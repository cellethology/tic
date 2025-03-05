"""
unified_pseudotime_pipeline.py

A single-file script that:

  Part A) Extracts center-cell representations from MicroEDataset (only needed once).
  Part B) Performs pseudo-time inference on the extracted cells with flexible parameters (dimension reduction, clustering, Slingshot).
  Part C) Plots the results (embedding by cluster, pseudotime vs. biomarkers, etc.).

User can run each part in a single script by toggling flags or adjusting parameters.
"""

import os
import argparse
import numpy as np
import torch

from torch.utils.data import DataLoader

# ========== PART A: Representation Extraction Imports ==========
from core.data.dataset import MicroEDataset, MicroEWrapperDataset, collate_microe

# ========== PART B: Pseudotime Inference Imports ==========
from core.pseduotime.clustering import Clustering
from core.pseduotime.dimensionality_reduction import DimensionalityReduction
from core.pseduotime.pseudotime import SlingshotMethod

# ========== PART C: Plotting Imports ==========
from utils.plot import my_y_transform, plot_pseudotime_vs_feature

# ========== MAIN SCRIPT: Unify All Steps ==========

def parse_arguments():
    parser = argparse.ArgumentParser(description="Unified Pseudotime Pipeline")
    # 1) Representation extraction
    parser.add_argument("--extract_representation", action="store_true",
                        help="If set, extract center cells with representations from Tissue/MicroE.")
    parser.add_argument("--root", type=str, default="./data",
                        help="Root directory containing 'Raw' and 'Cache' subfolders. Used for representation extraction.")
    
    # 2) Pseudotime analysis
    parser.add_argument("--do_pseudotime", action="store_true",
                        help="If set, run the pseudo-time inference on center cells.")
    parser.add_argument("--cells_input", type=str, default="./center_cells.pt",
                        help="Path to the .pt file containing center cells with representation.")
    parser.add_argument("--cells_output", type=str, default="./cells_with_pseudotime.pt",
                        help="Where to save the updated cells after pseudotime.")
    parser.add_argument("--representation_key", type=str, default="raw_expression",
                        help="Which representation to use from each cell.")
    
    # DR, Clustering, Pseudotime
    parser.add_argument("--dr_method", type=str, default="PCA", choices=["PCA","UMAP"],
                        help="Dimensionality reduction method.")
    parser.add_argument("--n_components", type=int, default=2,
                        help="Number of components.")
    parser.add_argument("--cluster_method", type=str, default="kmeans", choices=["kmeans","agg"],
                        help="Clustering method.")
    parser.add_argument("--n_clusters", type=int, default=5,
                        help="Number of clusters.")
    parser.add_argument("--start_node", type=int, default=0,
                        help="Start node for Slingshot.")
    parser.add_argument("--num_cells", type=int, default=None,
                        help="Randomly select this many cells from center_cells. If None, use all.")
    parser.add_argument("--output_dir", type=str, default="./pseudotime_plots",
                        help="Directory to save Slingshot plots.")
    
    # 3) Plotting
    parser.add_argument("--plot_embedding", action="store_true",
                        help="If set, plot the 2D embedding colored by cluster.")
    parser.add_argument("--plot_pseudotime_features", action="store_true",
                        help="If set, plot pseudotime vs. features (biomarkers or neighbor types).")
    parser.add_argument("--plot_biomarkers", nargs="+", default=None,
                        help="List of biomarker names to plot. Provide space-separated if multiple.")
    parser.add_argument("--plot_neighbor_types", nargs="+", default=None,
                        help="List of neighbor type (or general type) to plot. Provide space-separated if multiple.")
    parser.add_argument("--pseudotime_plot_bins", type=int, default=100,
                        help="Number of bins for the pseudotime vs. feature plot.")
    parser.add_argument("--pseudotime_plot_save", type=str, default=None,
                        help="If set, save the pseudotime vs. feature plot to this path.")
    parser.add_argument("--embedding_plot_save", type=str, default=None,
                        help="If set, save the embedding plot to this path.")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # ========== PART A: Representation Extraction if requested ==========
    if args.extract_representation:
        print("[Step A] Extracting representations from Tissue -> MicroE -> center cells.")
        # Build region ids from raw_dir
        raw_dir = os.path.join(args.root, "Raw")
        if not os.path.exists(raw_dir):
            raise FileNotFoundError(f"No raw folder: {raw_dir}")
        # Example: gather region_ids by .cell_data.csv filenames
        region_ids = list({fname.split('.')[0] for fname in os.listdir(raw_dir) if fname.endswith(".cell_data.csv")})
        
        print(f"Found {len(region_ids)} region IDs:", region_ids)
        
        dataset = MicroEDataset(
            root=args.root,
            region_ids=region_ids,
            k=3,
            microe_neighbor_cutoff=200.0,
            subset_cells=False,
            pre_transform=None,
            transform=None
        )
        wrapper_ds = MicroEWrapperDataset(dataset)
        loader = DataLoader(wrapper_ds, batch_size=1, shuffle=False, collate_fn=collate_microe)
        
        center_cells = []
        for batch in loader:
            for microe in batch:
                center_cell = microe.export_center_cell_with_representations()
                center_cells.append(center_cell)
        
        out_path = os.path.join(args.root, "center_cells.pt")
        torch.save(center_cells, out_path)
        print(f"[Step A] Saved {len(center_cells)} center cells to {out_path}")
    
    # ========== PART B: Pseudotime Analysis ==========
    if args.do_pseudotime:
        print("[Step B] Performing pseudotime inference with user-specified parameters.")
        
        if not os.path.exists(args.cells_input):
            raise FileNotFoundError(f"cells_input not found: {args.cells_input}")
        
        all_cells = torch.load(args.cells_input)
        print(f"[Info] Loaded {len(all_cells)} cells from {args.cells_input}")
        
        # Optionally sample a subset
        if args.num_cells is not None and args.num_cells < len(all_cells):
            idx = np.random.choice(len(all_cells), args.num_cells, replace=False)
            all_cells = [all_cells[i] for i in idx]
            print(f"[Info] Sampled {len(all_cells)} cells from total.")
        
        # 1) Extract embeddings from cell additional_features
        embeddings = []
        valid_cells = []
        for cell in all_cells:
            rep = cell.get_feature(args.representation_key)
            if rep is None:
                continue
            embeddings.append(rep)
            valid_cells.append(cell)
        embeddings = np.array(embeddings)
        print(f"[Info] Using representation '{args.representation_key}' from {len(valid_cells)} cells.")
        
        # 2) Dimension Reduction
        dim_reducer = DimensionalityReduction(method=args.dr_method, 
                                              n_components=args.n_components,
                                              random_state=42)
        reduced_emb = dim_reducer.reduce(embeddings)
        print(f"[Info] Reduced embedding shape = {reduced_emb.shape}")
        
        for cell, emb in zip(valid_cells, reduced_emb):
            cell.add_feature("embedding", emb)
        
        # 3) Clustering
        clusterer = Clustering(method=args.cluster_method, n_clusters=args.n_clusters)
        cluster_labels = clusterer.cluster(reduced_emb)
        for cell, label in zip(valid_cells, cluster_labels):
            cell.add_feature("cluster_label", label)
        
        # 4) Pseudotime
        pseudotime_method = SlingshotMethod(start_node=args.start_node)
        pseudotime_values = pseudotime_method.analyze(
            labels=cluster_labels,
            umap_embeddings=reduced_emb,
            output_dir=args.output_dir
        )
        for cell, pt in zip(valid_cells, pseudotime_values):
            cell.add_feature("pseudotime", float(pt))
        print(f"[Info] Pseudotime range: [{pseudotime_values.min():.2f}, {pseudotime_values.max():.2f}]")
        
        # 5) Save updated cells
        torch.save(valid_cells, args.cells_output)
        print(f"[Step B] Saved updated cells with pseudotime to {args.cells_output}")
    
    # ========== PART C: Plotting ==========
    # Let's do it only if user requests (plot_embedding or plot_pseudotime_features).
    if args.plot_embedding or args.plot_pseudotime_features:
        if not os.path.exists(args.cells_output):
            raise FileNotFoundError(f"Cannot plot because {args.cells_output} does not exist.")
        cells_for_plot = torch.load(args.cells_output)
        print(f"[Step C] Loaded {len(cells_for_plot)} cells from {args.cells_output} for plotting.")
        
        # Plot pseudotime vs. features (biomarkers or neighbor types)
        if args.plot_pseudotime_features:
            save_path_pt = args.pseudotime_plot_save
            # If user provided biomarkers:
            if args.plot_biomarkers:
                plot_pseudotime_vs_feature(
                    cells=cells_for_plot,
                    x_bins=args.pseudotime_plot_bins,
                    biomarkers=args.plot_biomarkers,
                    y_transform=my_y_transform,
                    save_path=save_path_pt
                )
                print(f"[Step C] Plotted pseudotime vs. biomarkers {args.plot_biomarkers}.")
            
            # If user provided neighbor types:
            elif args.plot_neighbor_types:
                plot_pseudotime_vs_feature(
                    cells=cells_for_plot,
                    x_bins=args.pseudotime_plot_bins,
                    neighbor_types=args.plot_neighbor_types,
                    y_transform=my_y_transform,
                    save_path=save_path_pt
                )
                print(f"[Step C] Plotted pseudotime vs. neighbor types {args.plot_neighbor_types}.")

if __name__ == "__main__":
    main()