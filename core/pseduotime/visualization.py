import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional
from scipy.signal import savgol_filter
from core.data.cell import CellBatch

# Transformation functions
def visualize_pseudotime_vs_biomarkers(cells: CellBatch, biomarkers_to_plot: Optional[list] = None, n_bins: int = 200, transform_fns: Optional[list] = None):
    """
    Visualize pseudotime vs biomarker expression for selected biomarkers.
    
    Args:
        cells (CellBatch): The batch of cells.
        biomarkers_to_plot (Optional[list], optional): List of biomarkers to plot. If None, plot all biomarkers.
        n_bins (int, optional): Number of bins for pseudotime (default is 200).
        transform_fns (list, optional): A list of transformation functions to apply to biomarkers (default is None).
    
    Returns:
        None
    """
    # Step 1: Get pseudotimes and biomarker expressions
    pseudotimes = cells.get_pseudotimes()
    biomarker_expressions = cells.get_biomarker_expression()

    if len(pseudotimes) == 0:
        raise ValueError("Pseudotime values are missing or empty.")

    # Step 2: Bin pseudotime values
    bin_edges = np.linspace(np.min(pseudotimes), np.max(pseudotimes), n_bins + 1)
    binned_pseudotime = np.digitize(pseudotimes, bin_edges) - 1

    # Step 3: Prepare DataFrame to hold transformed biomarker expressions
    biomarker_data = pd.DataFrame(biomarker_expressions)

    # Apply transformation functions if provided
    if transform_fns:
        for transform_fn in transform_fns:
            # Apply transformation function column-wise (to each biomarker)
            biomarker_data = biomarker_data.apply(transform_fn, axis=0)

    # Step 4: Select the biomarkers to plot (if specified)
    if biomarkers_to_plot:
        biomarker_data = biomarker_data[biomarkers_to_plot]

    # Step 5: Plot all selected biomarkers on the same plot
    plt.figure(figsize=(10, 6))

    for biomarker in biomarker_data.columns:
        # Step 5.1: Calculate the mean biomarker expression in each pseudotime bin
        mean_expression = []
        for bin_idx in range(n_bins):
            # Filter cells in the current bin
            bin_cells = biomarker_data[binned_pseudotime == bin_idx]
            mean_expression.append(bin_cells[biomarker].mean())
        
        # Step 5.2: Plot the biomarker expression against pseudotime bin
        plt.plot(bin_edges[:-1], mean_expression, label=biomarker)

    # Adding labels, title and legend
    plt.title('Biomarkers vs Pseudotime')
    plt.xlabel('Pseudotime')
    plt.ylabel('Biomarker Expression')
    plt.legend()

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

# Transformation functions
def rank_transform(values: np.ndarray) -> np.ndarray:
    """Rank transformation: Assign rank to values."""
    return np.argsort(np.argsort(values)) + 1  # Ranks start from 1

def normalize_to_01(values: np.ndarray) -> np.ndarray:
    """Normalize values to [0, 1] range."""
    value_min = np.min(values)
    value_max = np.max(values)
    value_range = value_max - value_min
    
    # Avoid division by zero if the range is zero (constant values)
    if value_range == 0:
        return np.zeros_like(values)  # Return an array of zeros if all values are constant
    
    return (values - value_min) / value_range
def smooth(values: np.ndarray, window_length: int = 5, polyorder: int = 3) -> np.ndarray:
    """Smooth values using Savitzky-Golay filter."""
    return savgol_filter(values, window_length, polyorder)

if __name__ == "__main__":
    cells = CellBatch(pkl_dir="/Users/zhangjiahao/Project/tic/results/train/2025-02-26_14-56-11/cells")
    biomarkers_to_plot = ["PanCK","aSMA"]
    transform_fns = [smooth, normalize_to_01]
    visualize_pseudotime_vs_biomarkers(cells, biomarkers_to_plot,transform_fns=transform_fns)
