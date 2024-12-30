import os
from matplotlib import pyplot as plt
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
    raise NotImplemented
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