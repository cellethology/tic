import os
import numpy as np
import pandas as pd
from utils.data_transform import normalize
from utils.visualization import plot_biomarker_bar_chart

def compute_averages(biomarker_df, visualization_kws):
    """
    Compute averages of selected biomarkers for visualization.

    Args:
        biomarker_df (pd.DataFrame): DataFrame containing biomarker expression data.
        visualization_kws (list): List of strings specifying the biomarkers to compute,
            e.g., ["ASMA", "avg(PANCK+VIMENTIN+PODOPLANIN)"].

    Returns:
        pd.DataFrame: A DataFrame with selected biomarkers and their averages.
    """
    transformed_df = pd.DataFrame(index=biomarker_df.index)
    for kw in visualization_kws:
        # Handle averaging of multiple biomarkers
        if kw.startswith("avg(") and kw.endswith(")"):
            biomarkers = kw[4:-1].split("+")  # Extract biomarkers between "avg()"
            transformed_df[kw] = biomarker_df[biomarkers].mean(axis=1)
        else:
            # Directly select single biomarkers
            transformed_df[kw] = biomarker_df[kw]
    return transformed_df

def analyze_and_visualize_expression(subgraphs, cluster_labels, biomarkers, output_dir, visualization_kws, visualization_transform=None):
    """
    Analyze biomarker expression data and visualize trends across clusters.

    Args:
        subgraphs (list): List of sampled subgraphs.
        cluster_labels (np.ndarray): Cluster labels corresponding to each subgraph.
        biomarkers (list): List of biomarkers to analyze, e.g., ["ASMA", "PANCK"].
        

    Returns:
        None
    """
    # Collect biomarker expression data for each cluster
    biomarker_data = []
    for subgraph, label in zip(subgraphs, cluster_labels):
        node_info = subgraph.get("node_info", {})
        biomarker_expressions = node_info.get("biomarker_expression", {})
        # Append biomarker data and cluster labels
        biomarker_data.append({
            "cluster": label,
            **{biomarker: biomarker_expressions.get(biomarker, np.nan) for biomarker in biomarkers}
        })

    biomarker_df = pd.DataFrame(biomarker_data).set_index("cluster")

    # Apply transformations to biomarker data (e.g., normalize)
    transformed_biomarkers = biomarker_df.copy()
    for transform in visualization_transform:
        transformed_biomarkers = transform(transformed_biomarkers)

    # Compute averages for visualization
    visualization_df = compute_averages(transformed_biomarkers, visualization_kws)

    # Aggregate biomarker data cluster-wise
    cluster_summary = visualization_df.groupby(visualization_df.index).mean()

    # Save cluster summary as CSV
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, "cluster_biomarker_summary.csv")
    cluster_summary.to_csv(output_csv_path)
    print(f"Cluster biomarker summary saved to {output_csv_path}")

    # Visualize biomarker averages as bar chart
    bar_chart_path = os.path.join(output_dir, "biomarker_trends.png")
    plot_biomarker_bar_chart(cluster_summary, visualization_kws, output_path=bar_chart_path)
