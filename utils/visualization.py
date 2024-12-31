import os
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from utils.data_transform import apply_transformations

def plot_biomarker_vs_pseudotime(aggregated_data, output_dir=None, method=None, transforms=None, use_bins=True):
    """
    Plot biomarker expression across pseudotime with optional transformations.

    Args:
        aggregated_data (dict): Aggregated biomarker data by pseudotime.
        output_dir (str, optional): Directory to save the output PNG file.
        method (str, optional): Method for pseudotime (used in labels).
        transforms (list of callable, optional): List of transformation functions to apply.
        use_bins (bool): Whether pseudotime is binned.
    """
    # Apply transformations if provided
    if transforms:
        aggregated_data = apply_transformations(aggregated_data, transforms)

    plt.figure(figsize=(12, 6))
    for biomarker, data in aggregated_data.items():
        plt.plot(data["bin"], data["value"], label=biomarker)

    # Use a default value for method if it's None
    method_label = f"{method} Pseudotime" if method is not None else "Pseudotime"
    plt.xlabel(f"{method_label} (binned)" if use_bins else method_label)
    plt.ylabel("Biomarker Expression Level")
    plt.title("Pseudotime vs Biomarker Expression Levels")
    plt.legend()
    plt.show()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "biomarker_vs_pseudotime.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Biomarker trends saved to {plot_path}")


def plot_filtering_process(embeddings, cluster_labels, filtered_embeddings, filtered_labels, output_path=None):
    """
    Visualize embeddings before and after filtering.

    Args:
        embeddings (np.ndarray): Original embeddings.
        cluster_labels (np.ndarray): Original cluster labels.
        filtered_embeddings (np.ndarray): Filtered embeddings.
        filtered_labels (np.ndarray): Filtered cluster labels.
        output_path (str, optional): Path to save the plot.
    """
    plt.figure(figsize=(12, 6))

    # Original embeddings
    plt.subplot(1, 2, 1)
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=cluster_labels, cmap='tab10', alpha=0.5)
    plt.title("Original Embeddings")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")

    # Filtered embeddings
    plt.subplot(1, 2, 2)
    plt.scatter(filtered_embeddings[:, 0], filtered_embeddings[:, 1], c=filtered_labels, cmap='tab10', alpha=0.7)
    plt.title("Filtered Embeddings")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"Filtering visualization saved to {output_path}")
    plt.show()

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
#-------------------------------------
# Visualize neighborhood composition along pseudotime trajectory
#-------------------------------------
def visualize_neighborhood_composition(
    aggregated_data,
    visualization_kwargs,
    output_dir,
    show_plots=True,
):
    """
    Visualize neighborhood composition along pseudotime.

    Args:
        aggregated_data (pd.DataFrame): DataFrame containing aggregated neighborhood composition data.
            - Index: Binned pseudotime values or raw pseudotime values (if binning is disabled).
            - Columns: One column per cell type and additional metadata columns.
        visualization_kwargs (list): Custom visualization configurations.
            - Example: ["Vessel", "avg(Tumor+Vessel)"]
        output_dir (str): Directory to save plots.
        show_plots (bool): Whether to display plots interactively.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter cell types for visualization
    visualization_data = {}
    for key in visualization_kwargs:
        if key.startswith("avg(") and key.endswith(")"):
            # Extract and average specified cell types
            avg_keys = key[4:-1].split("+")
            if all(col in aggregated_data.columns for col in avg_keys):
                visualization_data[key] = aggregated_data[avg_keys].mean(axis=1)
            else:
                print(f"Warning: Some cell types in '{key}' are not present in the data.")
        elif key in aggregated_data.columns:
            # Add individual cell type data
            visualization_data[key] = aggregated_data[key]
        else:
            print(f"Warning: '{key}' not found in the aggregated data and will be ignored.")

    # Visualization
    plt.figure(figsize=(10, 6))
    for label, values in visualization_data.items():
        plt.plot(aggregated_data.index, values, label=label)

    plt.title("Neighborhood Composition vs Pseudotime")
    plt.xlabel("Pseudotime")
    plt.ylabel("Normalized Composition")
    plt.legend(loc="upper left")
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, "neighborhood_composition_vs_pseudotime.png")
    plt.savefig(plot_path)

    if show_plots:
        plt.show()
    else:
        plt.close()

    print(f"Plot saved to {plot_path}")

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
        plot_path = os.path.join(output_dir, f"{ylabel.replace(' ', '_')}_trends_vs_pseudotime.png")
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
        plot_path = os.path.join(output_dir, "umap_vs_cell_types.png")
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"UMAP vs Cell Types plot saved to {plot_path}")

    if show_plots:
        plt.show()
    else:
        plt.close()
