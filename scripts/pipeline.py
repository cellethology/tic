#!/usr/bin/env python
"""
Pseudotime Inference Pipeline

This pipeline performs the following steps:
    1. (Optional) Extract Representation:
         - If the center cells file does not exist, extract the center cell representations
           from the raw dataset and save to a .pt file.
    2. Pseudotime Inference:
         - Load the center cells, perform dimensionality reduction, clustering, and
           pseudotime inference. The updated cells with pseudotime are saved.
    3. Plot:
         - Generate plots (e.g. pseudotime vs. biomarkers or pseudotime vs. neighbor types)
           based on the updated cells.
    4. Save Experiment Configuration:
         - All experiment parameters are recorded in a YAML file inside the experiment folder.

Usage example:
    python scripts/pipeline.py --dataset_root data/example \
                       --exp_dir results/experiments \
                       --representation_key raw_expression \
                       --dr_method PCA \
                       --n_components 2 \
                       --cluster_method kmeans \
                       --n_clusters 5 \
                       --start_node 1 \
                       --num_cells 10000 \
                       --x_bins 100 \
                       --biomarkers PanCK aSMA \
                       --transform my_y_transform
"""

import os
import argparse
import yaml
from datetime import datetime
from argparse import Namespace
import torch

from utils.extract_representation import extract_center_cells, save_center_cells
from utils.pseudotime_inference import run_pseudotime_analysis
from utils.plot import plot_pseudotime_vs_feature, my_y_transform, moving_average, normalize

def parse_arguments():
    """
    Parse command-line arguments for the pseudotime inference pipeline.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    # Part A: Extract Representation of MicroE center cells
    parser = argparse.ArgumentParser(description="Pseudotime Inference Pipeline")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root directory of the dataset (should contain Raw and Cache folders).")
    parser.add_argument("--center_cells_file", type=str, default="center_cells.pt",
                        help="Filename for the extracted center cells (default: center_cells.pt).")
    parser.add_argument("--exp_dir", type=str, default="experiment",
                        help="Base directory to save pseudotime inference and plot outputs.")

    # Part B: Pseudotime inference parameters
    parser.add_argument("--representation_key", type=str, default="raw_expression",
                        choices=['raw_expression', 'neighbor_composition', 'nn_embedding'],
                        help="Representation key in cell.additional_features.")
    parser.add_argument("--dr_method", type=str, default="PCA",
                        choices=["PCA", "UMAP"],
                        help="Dimensionality reduction method.")
    parser.add_argument("--n_components", type=int, default=2,
                        help="Number of components for dimensionality reduction.")
    parser.add_argument("--cluster_method", type=str, default="kmeans",
                        choices=["kmeans", "agg"],
                        help="Clustering method.")
    parser.add_argument("--n_clusters", type=int, default=2,
                        help="Number of clusters to form.")
    parser.add_argument("--start_node", type=int, default=None,
                        help="Starting node for Slingshot pseudotime inference (default: auto-detect).")
    parser.add_argument("--num_cells", type=int, default=None,
                        help="Randomly select this many cells for pseudotime inference (if provided).")

    # Part C: Plot parameters
    parser.add_argument("--x_bins", type=int, default=100,
                        help="Number of bins for the pseudotime plot.")
    parser.add_argument("--biomarkers", nargs="+", default=["PanCK", "aSMA"],
                        help="List of biomarkers to plot. Only used if neighbor_types is not provided.")
    parser.add_argument("--neighbor_types", nargs="+", default=["Immune", "Tumor", "Stromal", "Vascular"],
                        help="List of neighbor types to plot. Only used if biomarkers is not provided.")
    # Part D: Transform function selection
    parser.add_argument("--transform", type=str, default="my_y_transform",
                        choices=["none", "moving_average", "normalize", "my_y_transform"],
                        help="Transformation function to apply to the binned averages. "
                             "Options: none, moving_average, normalize, my_y_transform.")
    return parser.parse_args()

def save_config(config: dict, output_dir: str) -> None:
    """
    Save the experiment configuration to a YAML file in the output directory.

    Args:
        config (dict): Experiment configuration parameters.
        output_dir (str): Output directory where the YAML file will be saved.
    """
    config_path = os.path.join(output_dir, "experiment_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    print(f"[Info] Experiment configuration saved to {config_path}")

def main():
    """
    Main function for the pseudotime inference pipeline.
    """
    args = parse_arguments()

    # Create a unique experiment output directory with timestamp.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.exp_dir, f"experiment_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Record and save experiment parameters to YAML.
    config = vars(args)
    config["exp_dir"] = exp_dir  # update with the actual output directory
    save_config(config, exp_dir)

    # Construct the path to the center cells file.
    center_cells_path = os.path.join(args.dataset_root, args.center_cells_file)

    # If the center cells file does not exist, perform extraction.
    if not os.path.exists(center_cells_path):
        print("[Info] Center cells file not found. Running representation extraction...")
        center_cells = extract_center_cells(root=args.dataset_root)
        save_center_cells(center_cells, center_cells_path)
    else:
        print(f"[Info] Using existing center cells file: {center_cells_path}")

    # Prepare arguments for pseudotime inference.
    pseudo_args = Namespace(
        cells_input=center_cells_path,
        cells_output=os.path.join(exp_dir, "cells_with_pseudotime.pt"),
        representation_key=args.representation_key,
        dr_method=args.dr_method,
        n_components=args.n_components,
        random_state=42,  # Fixed random state for reproducibility
        cluster_method=args.cluster_method,
        n_clusters=args.n_clusters,
        start_node=args.start_node,
        num_cells=args.num_cells,
        output_dir=exp_dir
    )
    print("[Info] Running pseudotime inference...")
    run_pseudotime_analysis(pseudo_args)

    # Load the updated cells with pseudotime for plotting.
    cells_with_pt = torch.load(pseudo_args.cells_output)
    print(f"[Info] Loaded {len(cells_with_pt)} cells with pseudotime for plotting.")

    # Select the transformation function based on the user input.
    transform_mapping = {
        "none": None,
        "moving_average": moving_average,
        "normalize": normalize,
        "my_y_transform": my_y_transform
    }
    selected_transform = transform_mapping[args.transform]

    # Generate plots based on provided biomarkers and neighbor types.
    if args.biomarkers and not args.neighbor_types:
        plot_save_path = os.path.join(exp_dir, "pseudo_vs_biomarkers.svg")
        print("[Info] Generating pseudotime vs. biomarkers plot...")
        plot_pseudotime_vs_feature(cells_with_pt,
                                   x_bins=args.x_bins,
                                   biomarkers=args.biomarkers,
                                   y_transform=selected_transform,
                                   save_path=plot_save_path)
    elif args.neighbor_types and not args.biomarkers:
        plot_save_path = os.path.join(exp_dir, "pseudo_vs_neighbor.svg")
        print("[Info] Generating pseudotime vs. neighbor types plot...")
        plot_pseudotime_vs_feature(cells_with_pt,
                                   x_bins=args.x_bins,
                                   neighbor_types=args.neighbor_types,
                                   y_transform=selected_transform,
                                   save_path=plot_save_path)
    else:
        # If both are provided, generate both plots.
        biomarker_plot_path = os.path.join(exp_dir, "pseudo_vs_biomarkers.svg")
        neighbor_plot_path = os.path.join(exp_dir, "pseudo_vs_neighbor.svg")
        print("[Info] Generating pseudotime vs. biomarkers plot...")
        plot_pseudotime_vs_feature(cells_with_pt,
                                   x_bins=args.x_bins,
                                   biomarkers=args.biomarkers,
                                   y_transform=selected_transform,
                                   save_path=biomarker_plot_path)
        print("[Info] Generating pseudotime vs. neighbor types plot...")
        plot_pseudotime_vs_feature(cells_with_pt,
                                   x_bins=args.x_bins,
                                   neighbor_types=args.neighbor_types,
                                   y_transform=selected_transform,
                                   save_path=neighbor_plot_path)

    print("[Info] Pipeline completed successfully.")

if __name__ == "__main__":
    main()