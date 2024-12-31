# core/neighborhood_analysis.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spacegm.embeddings_analysis import get_composition_vector

def analyze_neighborhood_composition(subgraphs, pseudotime, cell_types, visualization_kwargs, cell_type_mapping, output_dir, plot=True):
    """
    Analyze neighborhood composition along pseudotime trajectory.

    Args:
        subgraphs (list): List of sampled subgraphs(dict).
        pseudotime (dict): Pseudotime values for subgraphs.
        cell_types (list): List of cell types to include in the analysis.
        visualization_kwargs (list): Custom visualization configurations.
        cell_type_mapping (dict): Mapping of cell type names to indices.
        output_dir (str): Directory to save results and plots.
        plot (bool): Whether to visualize neighborhood composition trends.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Cell type indices for analysis
    cell_type_indices = {name: idx for name, idx in cell_type_mapping.items() if name in cell_types}

    # Collect neighborhood composition data
    compositions = []
    pseudotime_values = []
    for subgraph_dict, pt in zip(subgraphs, pseudotime):
        composition_vec = get_composition_vector(subgraph_dict.get("subgraph",None), len(cell_type_mapping))
        compositions.append(composition_vec)
        pseudotime_values.append(pt)

    # Create a DataFrame
    df = pd.DataFrame(compositions, columns=list(cell_type_mapping.keys()))
    df["pseudotime"] = pseudotime_values

    # Aggregate by pseudotime bins
    binned = df.groupby("pseudotime").mean()

    # Save raw composition data
    binned.to_csv(os.path.join(output_dir, "neighborhood_composition.csv"))

    # Process visualization data
    visualization_data = {}
    for key in visualization_kwargs:
        if key.startswith("avg(") and key.endswith(")"):
            # Extract and average specified cell types
            avg_keys = key[4:-1].split("+")
            avg_indices = [cell_type_indices[name] for name in avg_keys if name in cell_type_indices]
            visualization_data[key] = binned.iloc[:, avg_indices].mean(axis=1)
        elif key in cell_type_indices:
            # Add individual cell type data
            visualization_data[key] = binned.iloc[:, cell_type_indices[key]]
        else:
            print(f"Warning: '{key}' not found in cell type mapping and will be ignored.")

    # Visualization
    if plot:
        plt.figure(figsize=(10, 6))
        for label, values in visualization_data.items():
            plt.plot(binned.index, values, label=label)
        plt.title("Neighborhood Composition vs Pseudotime")
        plt.xlabel("Pseudotime")
        plt.ylabel("Normalized Composition")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "neighborhood_composition_vs_pseudotime.png"))
        plt.show()
    

def compute_neighborhood_composition(
    subgraphs,
    pseudotime,
    cell_type_mapping,
    num_bins=100,
    overlap=0.2,
    use_bins=True
):
    """
    Compute neighborhood composition along pseudotime trajectory with optional overlapping bins.

    Args:
        subgraphs (list): List of sampled subgraphs (dict).
        pseudotime (dict): Pseudotime values for subgraphs.
        cell_type_mapping (dict): Mapping of cell type names to indices.
        num_bins (int): Number of bins to divide pseudotime into (if use_bins is True).
        overlap (float): Proportion of overlap between bins (0.0 to 1.0).
        use_bins (bool): Whether to use binning for pseudotime.

    Returns:
        pd.DataFrame: Aggregated neighborhood composition data across pseudotime bins.

    Structure of `aggregated_data`:
        - Index:
            - Bin center values (if `use_bins` is True).
            - Raw pseudotime values (if `use_bins` is False).
        - Columns:
            - One column per cell type, containing the average composition for that cell type in the bin.
    """
    if overlap < 0 or overlap >= 1:
        raise ValueError("Overlap must be between 0 and 1 (exclusive).")

    # Collect neighborhood composition data
    compositions = []
    pseudotime_values = []
    for subgraph_dict, pt in zip(subgraphs, pseudotime):
        composition_vec = get_composition_vector(subgraph_dict.get("subgraph", None), len(cell_type_mapping))
        compositions.append(composition_vec)
        pseudotime_values.append(pt)

    # Create a DataFrame
    df = pd.DataFrame(compositions, columns=list(cell_type_mapping.keys()))
    df["pseudotime"] = pseudotime_values

    # Calculate bin edges and apply overlap if required
    if use_bins:
        min_pt, max_pt = df["pseudotime"].min(), df["pseudotime"].max()
        bin_width = (max_pt - min_pt) / num_bins
        step_size = bin_width * (1 - overlap)  # Overlapping step size
        bin_edges = np.arange(min_pt, max_pt + step_size, step_size)

        # Aggregate data based on bins
        aggregated_data = []
        for i in range(len(bin_edges) - 1):
            bin_start, bin_end = bin_edges[i], bin_edges[i + 1]
            bin_center = (bin_start + bin_end) / 2
            bin_data = df[(df["pseudotime"] >= bin_start) & (df["pseudotime"] < bin_end)]
            if not bin_data.empty:
                bin_avg = bin_data.mean(numeric_only=True)
                bin_avg["bin_center"] = bin_center
                aggregated_data.append(bin_avg)

        aggregated_df = pd.DataFrame(aggregated_data)
        aggregated_df.set_index("bin_center", inplace=True)
    else:
        aggregated_df = df.groupby("pseudotime").mean()

    return aggregated_df
