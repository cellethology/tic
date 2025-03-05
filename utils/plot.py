# plot.py
import numpy as np
import torch

import matplotlib.pyplot as plt

from core.constant import  ALL_CELL_TYPES, GENERAL_CELL_TYPES

def plot_pseudotime_vs_feature(cells, 
                               x_bins: int = 200,
                               biomarkers: list = None,
                               neighbor_types: list = None,
                               y_transform: callable = None,
                               save_path: str = None):
    """
    Plot pseudotime vs. feature for a list of cells. The x-axis is binned pseudotime,
    and the y-axis is the average feature value in each bin. Exactly one of 'biomarkers' 
    or 'neighbor_types' must be provided.

    For biomarkers:
      - For each cell, the feature is obtained via cell.get_biomarker(bm).
    For neighbor composition:
      - For each cell, it is assumed that cell.get_feature("neighborhood_composition")
        returns a NumPy array of length len(ALL_CELL_TYPES) (order defined by ALL_CELL_TYPES).
      - If a neighbor type is one of the keys in GENERAL_CELL_TYPES, the function aggregates
        the values for all subtypes in that group by summing them.
      - Otherwise, it assumes the provided neighbor type is a raw cell type and retrieves the corresponding value.

    An optional y_transform callable (e.g., normalization or smoothing) can be applied to the binned averages.

    :param cells: List of Cell objects. Each cell must have pseudotime stored as cell.get_feature("pseudotime").
    :param x_bins: Number of bins for pseudotime (default 200).
    :param biomarkers: List of biomarker names to plot.
    :param neighbor_types: List of neighbor cell types (or general types) to plot.
    :param y_transform: Optional function to transform the binned average values.
    :param save_path: If provided, saves the plot to this path; otherwise, displays interactively.
    :raises ValueError: if neither or both biomarkers and neighbor_types are provided.
    """
    if (biomarkers is None and neighbor_types is None) or (biomarkers is not None and neighbor_types is not None):
        raise ValueError("Provide either a list of biomarkers or a list of neighbor_types (but not both).")
    
    # Extract pseudotime values for all cells.
    pseudo_times = np.array([cell.get_feature("pseudotime") for cell in cells])
    
    # Define bin edges and centers.
    bin_edges = np.linspace(pseudo_times.min(), pseudo_times.max(), x_bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    feature_means = {}  # key: feature name, value: binned average array
    
    if biomarkers is not None:
        # Process each biomarker individually.
        for bm in biomarkers:
            values = np.array([cell.get_biomarker(bm) for cell in cells], dtype=float)
            sums = np.zeros(x_bins)
            counts = np.zeros(x_bins)
            for pt, val in zip(pseudo_times, values):
                if np.isnan(val):
                    continue
                bin_idx = np.searchsorted(bin_edges, pt, side='right') - 1
                if bin_idx < 0 or bin_idx >= x_bins:
                    continue
                sums[bin_idx] += val
                counts[bin_idx] += 1
            means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts>0)
            if y_transform is not None:
                means = y_transform(means)
            feature_means[bm] = means
    else:
        # Process neighbor composition.
        # For each specified neighbor type, aggregate values.
        for ntype in neighbor_types:
            agg_values = []
            for cell in cells:
                comp = cell.get_feature("neighbor_composition")
                # If the composition is missing or of incorrect length, use nan.
                if comp is None or len(comp) != len(ALL_CELL_TYPES):
                    agg_values.append(np.nan)
                else:
                    # If ntype is a general group, aggregate over all subtypes.
                    if ntype in GENERAL_CELL_TYPES:
                        indices = [ALL_CELL_TYPES.index(subtype) for subtype in GENERAL_CELL_TYPES[ntype] if subtype in ALL_CELL_TYPES]
                        agg_val = np.sum(comp[indices])
                    else:
                        # Otherwise, treat ntype as a raw cell type.
                        try:
                            idx = ALL_CELL_TYPES.index(ntype)
                            agg_val = comp[idx]
                        except ValueError:
                            agg_val = np.nan
                    agg_values.append(agg_val)
            values = np.array(agg_values, dtype=float)
            sums = np.zeros(x_bins)
            counts = np.zeros(x_bins)
            for pt, val in zip(pseudo_times, values):
                if np.isnan(val):
                    continue
                bin_idx = np.searchsorted(bin_edges, pt, side='right') - 1
                if bin_idx < 0 or bin_idx >= x_bins:
                    continue
                sums[bin_idx] += val
                counts[bin_idx] += 1
            means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts>0)
            if y_transform is not None:
                means = y_transform(means)
            feature_means[ntype] = means

    # Create the plot.
    plt.figure(figsize=(10, 6))
    for feature_name, means in feature_means.items():
        plt.plot(bin_centers, means, label=feature_name, marker='o', markersize=3)
    
    plt.xlabel("Pseudotime")
    plt.ylabel("Feature Value")
    plt.title(f"Pseudotime vs. Feature (binned into {x_bins} intervals)")
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Annotate total number of cells.
    plt.text(0.01, 0.01, f"Total cells: {len(cells)}", transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='bottom', horizontalalignment='left',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def moving_average(y, window=5):
    # Simple moving average ignoring NaNs
    y_conv = np.convolve(np.nan_to_num(y, nan=np.nanmean(y)), np.ones(window)/window, mode='same')
    return y_conv

def normalize(y):
    # Normalize ignoring NaNs
    ymin = np.nanmin(y)
    ymax = np.nanmax(y)
    return (y - ymin) / (ymax - ymin) if ymax > ymin else y

# Combine transforms if desired
def my_y_transform(y):
    return normalize(moving_average(y, window=5))
    
if __name__ == "__main__":
    CELLS_INPUT_PATH = "/Users/zhangjiahao/Downloads/TicPlots/Expriments/num_10000/cells_with_pseudotime.pt"
    cells = torch.load(CELLS_INPUT_PATH)
    plot_pseudotime_vs_feature(cells, x_bins=100, biomarkers=["PanCK", "aSMA"],y_transform=my_y_transform,save_path="/Users/zhangjiahao/Downloads/TicPlots/Expriments/num_10000/pseudo_vs_biomarkers.svg")
    plot_pseudotime_vs_feature(cells, x_bins=100, neighbor_types=["Immune","Tumor","Stromal","Vascular"],y_transform=my_y_transform,save_path="/Users/zhangjiahao/Downloads/TicPlots/Expriments/num_10000/pseudo_vs_neighbor.svg")
    print("Done!")