# utils/plot.py
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from core.constant import ALL_BIOMARKERS, ALL_CELL_TYPES, GENERAL_CELL_TYPES

def plot_pseudotime_vs_feature(cells: list,
                               x_bins: int = 200,
                               biomarkers: list = None,
                               neighbor_types: list = None,
                               y_transform: callable = None,
                               save_path: str = None) -> plt.Figure:
    """
    Plot pseudotime versus feature values for a list of cell objects.

    The function bins the pseudotime values and computes the average feature
    value in each bin. Exactly one of 'biomarkers' or 'neighbor_types' must be provided.

    For biomarkers:
      - The feature value for each cell is obtained via cell.get_biomarker(bm).
    For neighbor composition:
      - The feature is derived from cell.get_feature("neighbor_composition"),
        and for a general neighbor type, it aggregates values over subtypes.

    Args:
        cells (list): List of cell objects. Each cell must have a "pseudotime" feature.
        x_bins (int, optional): Number of bins for pseudotime. Defaults to 200.
        biomarkers (list, optional): List of biomarker names. Defaults to None.
        neighbor_types (list, optional): List of neighbor cell types. Defaults to None.
        y_transform (callable, optional): Function to transform binned averages. Defaults to None.
        save_path (str, optional): If provided, saves the plot to the given path; otherwise, displays it.

    Returns:
        plt.Figure: The matplotlib Figure object of the plot.
    
    Raises:
        ValueError: If neither or both of biomarkers and neighbor_types are provided.
    """
    if (biomarkers is None and neighbor_types is None) or (biomarkers is not None and neighbor_types is not None):
        raise ValueError("Provide either a list of biomarkers or a list of neighbor_types (but not both).")
    
    # Extract pseudotime values from cells
    pseudo_times = np.array([cell.get_feature("pseudotime") for cell in cells])
    bin_edges = np.linspace(pseudo_times.min(), pseudo_times.max(), x_bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    feature_means = {}
    
    if biomarkers is not None:
        # Process each biomarker individually
        for bm in biomarkers:
            values = np.array([cell.get_biomarker(bm) for cell in cells], dtype=float)
            sums = np.zeros(x_bins)
            counts = np.zeros(x_bins)
            for pt, val in zip(pseudo_times, values):
                if np.isnan(val):
                    continue
                bin_idx = np.searchsorted(bin_edges, pt, side='right') - 1
                if 0 <= bin_idx < x_bins:
                    sums[bin_idx] += val
                    counts[bin_idx] += 1
            means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
            if y_transform:
                means = y_transform(means)
            feature_means[bm] = means
    else:
        # Process neighbor composition
        for ntype in neighbor_types:
            agg_values = []
            for cell in cells:
                comp = cell.get_feature("neighbor_composition")
                if comp is None or len(comp) != len(ALL_CELL_TYPES):
                    agg_values.append(np.nan)
                else:
                    if ntype in GENERAL_CELL_TYPES:
                        indices = [ALL_CELL_TYPES.index(sub) for sub in GENERAL_CELL_TYPES[ntype] if sub in ALL_CELL_TYPES]
                        agg_val = np.sum(comp[indices])
                    else:
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
                if 0 <= bin_idx < x_bins:
                    sums[bin_idx] += val
                    counts[bin_idx] += 1
            means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
            if y_transform:
                means = y_transform(means)
            feature_means[ntype] = means

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for feature_name, means in feature_means.items():
        ax.plot(bin_centers, means, label=feature_name, marker='o', markersize=3)
    
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("Feature Value")
    ax.set_title(f"Pseudotime vs. Feature (binned into {x_bins} intervals)")
    ax.legend(loc='upper right')
    ax.grid(True)
    ax.text(0.01, 0.01, f"Total cells: {len(cells)}", transform=ax.transAxes,
            fontsize=12, verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    
    return fig

def moving_average(y: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Compute a simple moving average of the input array, ignoring NaN values.

    Args:
        y (np.ndarray): Input array.
        window (int, optional): Window size for the moving average. Defaults to 5.

    Returns:
        np.ndarray: Smoothed array.
    """
    y_conv = np.convolve(np.nan_to_num(y, nan=np.nanmean(y)), np.ones(window) / window, mode='same')
    return y_conv

def normalize(y: np.ndarray) -> np.ndarray:
    """
    Normalize the input array to the range [0, 1], ignoring NaN values.

    Args:
        y (np.ndarray): Input array.

    Returns:
        np.ndarray: Normalized array.
    """
    ymin = np.nanmin(y)
    ymax = np.nanmax(y)
    return (y - ymin) / (ymax - ymin) if ymax > ymin else y

def my_y_transform(y: np.ndarray) -> np.ndarray:
    """
    Apply moving average smoothing followed by normalization to the input array.

    Args:
        y (np.ndarray): Input array.

    Returns:
        np.ndarray: Transformed array.
    """
    return normalize(moving_average(y, window=5))

#-------------------------------------------------
# Causal Infer related plot
#-------------------------------------------------
def plot_top_x_effects(results_df: pd.DataFrame, y_biomarkers: list|str, key: str = 'p_value',ascending: bool = True):
    """
    Plot the top 10 X variables ranked by their estimated causal effect for each Y biomarker.

    Args:
        results_df (pd.DataFrame): DataFrame containing the causal analysis results.
        y_biomarkers (list|str): List of Y biomarkers to plot.
    """
    if isinstance(y_biomarkers, str):
        y_biomarkers = [y_biomarkers]
    for y in y_biomarkers:
        sub_df = results_df[results_df['y_biomarker'] == y].sort_values(by=key, ascending=ascending).head(10)
        plt.figure(figsize=(10, 5))
        sns.barplot(x='x_variable', y=key, data=sub_df)
        plt.title(f'Top 10 X Variables Affecting {y}')
        plt.xticks(rotation=45)
        plt.ylabel('Estimated Causal Effect')
        plt.xlabel('X Variable')
        plt.tight_layout()
        plt.show()
    
def plot_x_effect_heatmap(results_df: pd.DataFrame, y_biomarkers: list|str, key: str = 'p_value'):
    """
    Generate a heatmap showing the effects of combinations of biomarkers and cell types on a specific Y biomarker.

    Args:
        results_df (pd.DataFrame): DataFrame containing the causal analysis results with columns for x_variable, y_biomarker, and the specified key.
        y_biomarker (str): The Y biomarker to analyze.
        key (str): The column from results_df to use for heatmap values (default 'p_value').
    """
    if isinstance(y_biomarkers, str):
        y_biomarkers = [y_biomarkers]
    for y_biomarker in y_biomarkers:
        # Filter results for the specified Y biomarker
        sub_df = results_df[results_df['y_biomarker'] == y_biomarker]

        # Create an empty dataframe to store the effects
        effect_matrix = pd.DataFrame(0.0, index=ALL_CELL_TYPES, columns=ALL_BIOMARKERS)

        # Populate the effect matrix
        for index, row in sub_df.iterrows():
            biomarker, cell_type = row['x_variable'].split('&')
            if biomarker in ALL_BIOMARKERS and cell_type in ALL_CELL_TYPES:
                effect_matrix.at[cell_type, biomarker] = row[key]

        # Plotting the heatmap
        plt.figure(figsize=(20, 10))
        sns.heatmap(effect_matrix, annot=True, fmt=".2f", cmap='viridis', linewidths=.5, cbar_kws={'label': f'Effect on {y_biomarker}'})
        plt.title(f'Effect of Biomarkers and Cell Types on {y_biomarker}, key: {key}')
        plt.xlabel('Biomarkers')
        plt.ylabel('Cell Types')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()