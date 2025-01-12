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
        output_dir (str): Directory to save output files.
        visualization_kws (list): List of keys or averages for visualization, e.g., ["PANCK", "avg(ASMA+VIMENTIN)"].
        visualization_transform (list, optional): List of transformation functions to apply (e.g., normalize).

    Returns:
        None
    """
    # Collect biomarker expression data for each cluster
    biomarker_data = []
    cluster_info = []  # To save region_id and cell_id

    for subgraph, label in zip(subgraphs, cluster_labels):
        node_info = subgraph.get("node_info", {})
        biomarker_expressions = node_info.get("biomarker_expression", {})
        region_id = subgraph.get("region_id", "unknown")
        cell_id = subgraph.get("cell_id", None)
        
        # Append biomarker data and cluster labels
        biomarker_data.append({
            "cluster": label,
            **{biomarker: biomarker_expressions.get(biomarker, np.nan) for biomarker in biomarkers}
        })
        
        # Append clustering information for each cell
        cluster_info.append({
                "region_id": region_id,
                "cell_id": cell_id,
                "cluster": label
            })

    # Save clustering information with region_id and cell_id
    cluster_info_df = pd.DataFrame(cluster_info)
    cluster_info_output_path = os.path.join(output_dir, "cluster_summary.csv")
    os.makedirs(output_dir, exist_ok=True)
    cluster_info_df.to_csv(cluster_info_output_path, index=False)
    print(f"Cluster summary saved to {cluster_info_output_path}")

    # Process biomarker data
    biomarker_df = pd.DataFrame(biomarker_data).set_index("cluster")

    # Apply transformations to biomarker data (e.g., normalize)
    transformed_biomarkers = biomarker_df.copy()
    for transform in visualization_transform or []:
        transformed_biomarkers = transform(transformed_biomarkers)

    # Compute averages for visualization
    visualization_df = compute_averages(transformed_biomarkers, visualization_kws)

    # Aggregate biomarker data cluster-wise
    cluster_summary = visualization_df.groupby(visualization_df.index).mean()

    # Save cluster biomarker summary as CSV
    cluster_summary_output_path = os.path.join(output_dir, "cluster_biomarker_summary.csv")
    cluster_summary.to_csv(cluster_summary_output_path)
    print(f"Cluster biomarker summary saved to {cluster_summary_output_path}")

    # Visualize biomarker averages as a bar chart
    bar_chart_path = os.path.join(output_dir, "biomarker_trends.png")
    plot_biomarker_bar_chart(cluster_summary, visualization_kws, output_path=bar_chart_path)
    print(f"Biomarker trends bar chart saved to {bar_chart_path}")
