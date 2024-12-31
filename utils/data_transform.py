'''
This file implements tranform function/class
Author: Jiahao Zhang
File Created: 19:39 30 Dec 2024s
'''
#--------------------
# clustering embedding tranformation
#--------------------
import numpy as np
import pandas as pd


class EmbeddingTransformPipeline:
    """
    A pipeline for applying transformations to embeddings.

    Attributes:
        transformations (list): List of transformations to apply sequentially.
    """
    def __init__(self, transformations=None):
        self.transformations = transformations if transformations else []

    def add_transform(self, transform_func):
        """
        Add a transformation function to the pipeline.

        Args:
            transform_func (callable): A function that takes (embeddings, cluster_labels)
                                       and returns modified embeddings and labels.
        """
        self.transformations.append(transform_func)

    def apply(self, embeddings, cluster_labels):
        """
        Apply all transformations in sequence.

        Args:
            embeddings (np.ndarray): Input embeddings.
            cluster_labels (np.ndarray): Cluster labels corresponding to embeddings.

        Returns:
            tuple: Transformed embeddings and labels.
        """
        for transform in self.transformations:
            embeddings, cluster_labels = transform(embeddings, cluster_labels)
        return embeddings, cluster_labels

# Define the filter function as a transform
def filter_by_cluster_proximity_transform(threshold=0.8):
    def transform(embeddings, cluster_labels):
        return filter_by_cluster_proximity(embeddings, cluster_labels, threshold)
    return transform

def filter_by_cluster_proximity(embeddings, cluster_labels, threshold=0.8):
    """
    Filter data points based on their proximity to cluster centers.

    Args:
        embeddings (np.ndarray): Clustered embeddings (e.g., UMAP embeddings).
        cluster_labels (np.ndarray): Labels indicating cluster membership.
        threshold (float): Proportion of points to retain (e.g., 0.8 to keep 80% closest points).

    Returns:
        tuple: Filtered embeddings and corresponding cluster labels.
    """
    from scipy.spatial.distance import cdist

    unique_clusters = np.unique(cluster_labels)
    cluster_centers = []

    # Compute cluster centers
    for cluster in unique_clusters:
        cluster_points = embeddings[cluster_labels == cluster]
        center = cluster_points.mean(axis=0)
        cluster_centers.append(center)
    cluster_centers = np.array(cluster_centers)

    # Compute distances to cluster centers
    distances = cdist(embeddings, cluster_centers, metric='euclidean')
    closest_distances = distances[np.arange(len(cluster_labels)), cluster_labels]

    # Determine distance threshold for each cluster
    thresholds = {}
    for cluster in unique_clusters:
        cluster_distances = closest_distances[cluster_labels == cluster]
        cutoff = np.percentile(cluster_distances, threshold * 100)
        thresholds[cluster] = cutoff

    # Filter points based on threshold
    mask = np.array([
        closest_distances[i] <= thresholds[cluster_labels[i]]
        for i in range(len(cluster_labels))
    ])

    # Return filtered embeddings and labels
    filtered_embeddings = embeddings[mask]
    filtered_labels = cluster_labels[mask]
    return filtered_embeddings, filtered_labels

#------------------
# Pseudo time transformation
#------------------
# """
# Transformable Biomarker Visualization:
# This section provides a flexible framework for visualizing biomarker expression across pseudotime.
# - Allows a sequence of transformation functions (e.g., normalize -> smooth) to be applied to the biomarker data.
# - Ensures composability and flexibility by supporting user-defined transformation functions.
# - Generates plots of biomarker trends with optional output to specified directories.
# """

# def apply_transformations(aggregated_data, transformations):
#     """
#     Apply a series of transformations to the aggregated biomarker data.

#     Args:
#         aggregated_data (dict): Aggregated biomarker data by pseudotime.
#         transformations (list of callable): List of transformation functions to apply.

#     Returns:
#         dict: Transformed biomarker data by pseudotime.
#     """
#     transformed_data = aggregated_data.copy()
#     for transform in transformations:
#         if callable(transform):
#             transformed_data = transform(transformed_data)
#         else:
#             raise ValueError(f"Transformation {transform} is not callable.")
#     return transformed_data

# # Define transformation functions
# def normalize(aggregated_data):
#     normalized_data = {}
#     for biomarker, data in aggregated_data.items():
#         values = data["value"]
#         min_val = values.min()
#         max_val = values.max()
#         normalized_values = (values - min_val) / (max_val - min_val)
#         normalized_data[biomarker] = pd.DataFrame({
#             "bin": data["bin"],
#             "value": normalized_values
#         })
#     return normalized_data

# def smooth(aggregated_data, window_size=5):
#     smoothed_data = {}
#     for biomarker, data in aggregated_data.items():
#         values = data["value"].rolling(window=window_size, min_periods=1).mean()
#         smoothed_data[biomarker] = pd.DataFrame({
#             "bin": data["bin"],
#             "value": values
#         })
#     return smoothed_data

#-----------------------------------
def apply_transformations(aggregated_data, transformations):
    """
    Apply a series of transformations to the aggregated data.

    Args:
        aggregated_data (pd.DataFrame): Aggregated data with pseudotime or bins as the index.
        transformations (list of callable): List of transformation functions to apply.

    Returns:
        pd.DataFrame: Transformed aggregated data.
    """
    transformed_data = aggregated_data.copy()
    for transform in transformations:
        if callable(transform):
            transformed_data = transform(transformed_data)
        else:
            raise ValueError(f"Transformation {transform} is not callable.")
    return transformed_data

# Example transformation functions
def normalize(data):
    return (data - data.min()) / (data.max() - data.min())

def smooth(data, window_size=5):
    return data.rolling(window=window_size, min_periods=1).mean()
