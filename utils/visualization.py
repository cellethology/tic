import os
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from utils.data_transform import apply_transformations
import numpy as np
from scipy.interpolate import interp1d
from typing import Any, Dict, List, Optional, Callable

#--------------------------------
# Cluster-wise biomarker averages
#--------------------------------
def plot_biomarker_bar_chart(cluster_summary, visualization_kws, output_path=None):
    """
    Plot biomarker averages for each cluster as a bar chart.

    Args:
        cluster_summary (pd.DataFrame): DataFrame containing average biomarker expression per cluster.
        visualization_kws (list): List of biomarkers or calculated metrics to visualize.
        output_path (str, optional): File path to save the bar chart.

    Returns:
        None
    """
    # Plot the bar chart for selected biomarkers
    ax = cluster_summary[visualization_kws].plot.bar(figsize=(12, 6))
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Biomarker Expression (Averaged)")
    ax.set_title("Cluster-wise Biomarker Averages")
    plt.legend(title="Biomarkers")
    plt.tight_layout()

    # Save the chart if output_path is specified
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Bar chart saved to {output_path}")
    plt.show()

def plot_trends(
    aggregated_data,
    visualization_kwargs,
    output_dir=None,
    title="Feature Trends vs Pseudotime",
    xlabel="Pseudotime",
    ylabel="Feature Value",
    show_plots=True,
    transforms=None,
):
    """
    Plot trends for aggregated data across pseudotime.

    Args:
        aggregated_data (pd.DataFrame): Aggregated feature data by pseudotime.
            - Index: Pseudotime bins or raw pseudotime values.
            - Columns: Feature values (e.g., biomarkers, cell types, etc.).
        visualization_kwargs (list): List of features to plot (e.g., individual keys or averages).
        output_dir (str, optional): Directory to save the plot.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        show_plots (bool): Whether to display the plot interactively.
        transforms (list of callable, optional): Transformations to apply to the data.
    """
    # Apply transformations if provided
    if transforms:
        aggregated_data = apply_transformations(aggregated_data, transforms)

    # Process visualization data
    visualization_data = {}
    for key in visualization_kwargs:
        if key.startswith("avg(") and key.endswith(")"):
            avg_keys = key[4:-1].split("+")
            if all(k in aggregated_data.columns for k in avg_keys):
                visualization_data[key] = aggregated_data[avg_keys].mean(axis=1)
            else:
                print(f"Warning: Some keys in '{key}' are not present in aggregated_data.")
        elif key in aggregated_data.columns:
            visualization_data[key] = aggregated_data[key]
        else:
            print(f"Warning: Key '{key}' not found in aggregated_data.")

    # Plot
    plt.figure(figsize=(10, 6))
    for label, values in visualization_data.items():
        plt.plot(aggregated_data.index, values, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"{ylabel.replace(' ', '_')}_trends_vs_pseudotime.svg")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

    if show_plots:
        plt.show()
    else:
        plt.close()

def plot_umap_vs_cell_types(umap_embeddings, cell_types, cell_type_mapping, output_dir=None, show_plots=True):
    """
    Plot UMAP embeddings colored by cell types.

    Args:
        umap_embeddings (np.ndarray): UMAP embeddings of shape (n_samples, 2).
        cell_types (list): List of cell type labels corresponding to embeddings.
        cell_type_mapping (dict): Mapping of cell type indices to names.
        output_dir (str, optional): Directory to save the plot. Defaults to None.
        show_plots (bool, optional): Whether to display the plot interactively. Defaults to True.

    Returns:
        None
    """
    # Map cell type indices to names
    cell_type_names = [cell_type_mapping.get(ct, "Unknown") for ct in cell_types]

    # Create a DataFrame for visualization
    df = pd.DataFrame(umap_embeddings, columns=["UMAP1", "UMAP2"])
    df["CellType"] = cell_type_names

    # Plot using seaborn
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x="UMAP1",
        y="UMAP2",
        hue="CellType",
        palette="tab10",
        data=df,
        alpha=0.8,
        s=30
    )
    plt.title("UMAP Embeddings vs Cell Types")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.legend(title="Cell Types", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Save or show the plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "umap_vs_cell_types.svg")
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"UMAP vs Cell Types plot saved to {plot_path}")

    if show_plots:
        plt.show()
    else:
        plt.close()
    
##########################################
# Visualize feature trends vs pseudotime #
##########################################
class Transform:
    def __init__(self, pseudotime_csv_path: str, raw_dir: str):
        self.pseudotime_csv_path = pseudotime_csv_path
        self.raw_dir = raw_dir

    def normalize(self, data: pd.Series) -> pd.Series:
        """Normalize data to the range [0, 1]."""
        return (data - data.min()) / (data.max() - data.min())

    def smooth(self, data: pd.Series, window_size: int = 5) -> pd.Series:
        """Smooth data using a rolling window average."""
        return data.rolling(window=window_size, min_periods=1).mean()

    def binning(self, merged_df: pd.DataFrame, x_key: str, num_bins: int, overlap: float) -> pd.Series:
        """Bin the data for x_key into `num_bins` bins with `overlap`."""
        min_x, max_x = merged_df[x_key].min(), merged_df[x_key].max()
        bin_width = (max_x - min_x) / num_bins
        step_size = bin_width * (1 - overlap)
        bins = np.arange(min_x, max_x + step_size, step_size)
        return pd.cut(merged_df[x_key], bins=bins, labels=False, include_lowest=True)

    def visualize_features_vs_pseudotime(
        self,
        y_keys: List[str],
        x_key: str = "pseudotime",
        output_dir: str = "./",
        x_transform: Optional[Dict[str, Any]] = None,
        y_transform: Optional[List[Callable]] = None,
        show_plots: bool = True,
    ):
        """
        Visualize features (e.g., biomarkers) vs pseudotime with x-axis binning and y-axis transformations.
        """
        # Load pseudotime CSV
        pseudotime_df = pd.read_csv(self.pseudotime_csv_path)
        pseudotime_df["region_id"] = pseudotime_df["region_id"].astype(str)
        pseudotime_df["cell_id"] = pseudotime_df["cell_id"].astype(str)

        # Initialize aggregated data
        aggregated_data = {x_key: []}
        for y_key in y_keys:
            aggregated_data[y_key] = []

        # Process each region in the pseudotime data
        for region_id in pseudotime_df["region_id"].unique():
            # Load the corresponding raw expression file
            expression_file_path = os.path.join(self.raw_dir, f"{region_id}.expression.csv")
            if not os.path.exists(expression_file_path):
                print(f"No matching data for region '{region_id}'. Skipping.")
                continue

            expression_df = pd.read_csv(expression_file_path)
            expression_df.rename(columns={"ACQUISITION_ID": "region_id", "CELL_ID": "cell_id"}, inplace=True)
            expression_df["region_id"] = expression_df["region_id"].astype(str)
            expression_df["cell_id"] = expression_df["cell_id"].astype(str)

            # Filter pseudotime data for the current region
            region_df = pseudotime_df[pseudotime_df["region_id"] == region_id]

            # Merge with pseudotime data
            merged_df = region_df.merge(expression_df, on=["region_id", "cell_id"], how="inner")

            # Apply x-axis binning by total number of bins
            if x_transform and x_transform.get("method") == "binning":
                num_bins = x_transform.get("num_bins", 100)
                overlap = x_transform.get("overlap", 0.2)
                merged_df["binned_" + x_key] = self.binning(merged_df, x_key, num_bins, overlap)
                binned_key = "binned_" + x_key
            else:
                binned_key = x_key

            # Apply y-axis transformations
            for y_key in y_keys:
                if y_key in merged_df.columns:
                    y_data = merged_df[y_key]
                    if y_transform:
                        for transform_func in y_transform:
                            y_data = transform_func(y_data)
                    merged_df[y_key] = y_data
                else:
                    print(f"Feature '{y_key}' not found in region '{region_id}'. Skipping.")

            # Aggregate data by bins
            if binned_key.startswith("binned_"):
                grouped = merged_df.groupby(binned_key).mean(numeric_only=True).reset_index()
            else:
                grouped = merged_df

            # Collect the aggregated results
            for y_key in y_keys:
                if y_key in grouped.columns:
                    aggregated_data[x_key].extend(grouped[binned_key] if binned_key.startswith("binned_") else grouped[x_key])
                    aggregated_data[y_key].extend(grouped[y_key])
                else:
                    aggregated_data[y_key].extend([np.nan] * len(grouped[x_key]))

        # Ensure all arrays in aggregated_data have the same length
        max_length = max(len(values) for values in aggregated_data.values())
        for key, values in aggregated_data.items():
            if len(values) < max_length:
                aggregated_data[key].extend([np.nan] * (max_length - len(values)))

        # Create a DataFrame
        aggregated_df = pd.DataFrame(aggregated_data)

        # Plot data
        plt.figure(figsize=(10, 6))
        for y_key in y_keys:
            if y_key in aggregated_df.columns:
                # Interpolate for smooth plotting
                x_values = aggregated_df[x_key]
                y_values = aggregated_df[y_key]

                # Drop NaN values for interpolation
                valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
                x_values = x_values[valid_indices]
                y_values = y_values[valid_indices]

                # Use linear interpolation for smooth plotting
                interpolator = interp1d(x_values, y_values, kind="linear", fill_value="extrapolate")
                x_smooth = np.linspace(x_values.min(), x_values.max(), 500)  # Increase points for smooth curve
                y_smooth = interpolator(x_smooth)

                # Plot the trend
                plt.plot(
                    x_smooth,
                    y_smooth,
                    label=y_key,
                    alpha=0.8,
                    linestyle="-",
                )

        # Configure the plot
        plt.xlabel(x_key)
        plt.ylabel("Feature Value")
        plt.title("Feature Trends vs Pseudotime")
        plt.legend()
        plt.grid(True)

        # Save and/or display the plot
        output_path = os.path.join(output_dir, "features_vs_pseudotime_continuous.svg")
        plt.savefig(output_path)
        if show_plots:
            plt.show()
        print(f"Plot saved to {output_path}")