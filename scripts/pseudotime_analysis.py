# Pseudo-Time Analysis Pipeline

"""
# README
This notebook implements a pipeline for analyzing pseudo-time in cellular graph datasets. It includes the following steps:

1. **Dataset Initialization**:
   - Load and preprocess cellular graph datasets.

2. **Subgraph Sampling**:
   - Sample subgraphs from the dataset based on specific parameters (e.g., cell type, region).

3. **Embedding Preparation**:
   - Generate various embeddings for sampled subgraphs:
     - Expression vectors
     - Composition vectors
     - Node embeddings
     - Graph embeddings

4. **Pseudo-Time Analysis**:
   - Perform dimensionality reduction and clustering.
   - Compute pseudo-time trajectories using selected start nodes.

5. **Visualization and Output**:
   - Visualize biomarker trends across pseudo-time.
   - Save results (e.g., pseudo-time values) to structured directories for future analysis.

# Configuration
All parameters (e.g., paths, embedding types, analysis settings) are controlled through a centralized `Config` class for flexibility and ease of use.

# Requirements
- Python libraries: `numpy`, `pandas`, `matplotlib`, `torch`
- Custom dependencies: `spacegm`, `tic`

# Usage
To execute the pipeline:
1. Modify the `Config` class to set paths and desired parameters.
2. Run the script to generate outputs in structured directories.

"""

import warnings
import os
import random
from adapters.space_gm_adapter import CustomSubgraphSampler
from core.pseudotime_analysis import aggregate_biomarker_by_pseudotime_with_overlap, perform_pseudotime_analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spacegm.utils import BIOMARKERS_UPMC, CELL_TYPE_FREQ_UPMC, CELL_TYPE_MAPPING_UPMC
import torch
import torch.nn as nn
from spacegm import CellularGraphDataset, GNN_pred

from spacegm.embeddings_analysis import (
    get_embedding,
    get_composition_vector,
    dimensionality_reduction_combo
)

from utils.data_transform import normalize
from utils.visualization import plot_biomarker_vs_pseudotime


# Suppress warnings
warnings.filterwarnings("ignore")

class Config:
    def __init__(self):
        # Paths
        self.data_root = "/root/autodl-tmp/Data/Space-Gm/Processed_Dataset/UPMC"
        self.output_dir = "/root/TIC/data/embedding_analysis/pseudotime_analysis/test"
        self.model_path = "/root/autodl-tmp/Data/Space-Gm/Processed_Dataset/UPMC/model/graph_level/GIN-primary_outcome-0/model_save_6.pt"
        self.device = 'cuda:0'

        # Dataset parameters
        self.dataset_kwargs = {
            'raw_folder_name': 'graph',
            'processed_folder_name': 'tg_graph',
            'node_features': ["cell_type", "SIZE", "biomarker_expression", "neighborhood_composition", "center_coord"],
            'edge_features': ["edge_type", "distance"],
            'cell_type_mapping': CELL_TYPE_MAPPING_UPMC,
            'cell_type_freq': CELL_TYPE_FREQ_UPMC,
            'biomarkers': BIOMARKERS_UPMC,
            'subgraph_size': 3,
            'subgraph_source': 'chunk_save',
            'subgraph_allow_distant_edge': True,
            'subgraph_radius_limit': 55 * 3 + 35,
            'biomarker_expression_process_method': "linear",
            'biomarker_expression_lower_bound': 0,
            'biomarker_expression_upper_bound': 18,
            'neighborhood_size': 10,
        }

        # Sampler parameters
        self.sampler_kwargs = {
            'total_samples': 1000,
            'cell_type': 9,
            'region_id': None,
            'batch_size': 64,
            'num_workers': 8,
            'output_csv': os.path.join(self.output_dir,'sampled_subgraphs.csv'), # save sampled subgraph infomation
            'include_node_info': True,
            'random_seed': 42,
        }

        # Pseudo-time analysis parameters
        self.embedding_keys = ["expression_vectors", "composition_vectors", "node_embeddings", "graph_embeddings", "composition_vectors+expression_vectors","node_embeddings+expression_vectors","graph_embeddings+composition_vectors"]
        self.start_nodes = [0,1]
        self.biomarkers = ["ASMA", "PANCK", "VIMENTIN", "PODOPLANIN"]
        self.show_plots = True
        self.num_bins = 100
        self.overlap = 0.2 # bin overlap
        self.use_bins = True
        self.plotting_transform = [normalize]

def initialize_dataset(root_path, dataset_kwargs):
    """Initialize the CellularGraphDataset."""
    return CellularGraphDataset(root_path, **dataset_kwargs)

def initialize_sampler(dataset, sampler_kwargs):
    """Initialize the CustomSubgraphSampler."""
    return CustomSubgraphSampler(dataset, **sampler_kwargs)

def prepare_embeddings(dataset, sampler, model_path, device, embedding_keys):
    """Prepare and add selected embeddings to subgraphs."""
    pyg_subgraphs = sampler.get_subgraph_objects()

    embeddings_dict = {}

    if "composition_vectors" in embedding_keys:
        composition_vectors = [
            get_composition_vector(data, n_cell_types=len(dataset.cell_type_mapping))
            for data in pyg_subgraphs
        ]
        sampler.add_kv_to_sampled_subgraphs(composition_vectors, key="composition_vectors")
        embeddings_dict["composition_vectors"] = composition_vectors

    if "node_embeddings" in embedding_keys or "graph_embeddings" in embedding_keys:
        model_kwargs = {
            'num_layer': dataset.subgraph_size,
            'num_node_type': len(dataset.cell_type_mapping) + 1,
            'num_feat': dataset[0].x.shape[1] - 1,
            'emb_dim': 512,
            'num_node_tasks': 0,
            'num_graph_tasks': 1,
            'node_embedding_output': 'last',
            'drop_ratio': 0.25,
            'graph_pooling': "max",
            'gnn_type': 'gin',
        }

        model = GNN_pred(**model_kwargs).to(device)
        model.load_state_dict(torch.load(model_path))

        node_embeddings, graph_embeddings, _ = get_embedding(model, pyg_subgraphs, device)

        if "node_embeddings" in embedding_keys:
            sampler.add_kv_to_sampled_subgraphs(node_embeddings, key="node_embeddings")
            embeddings_dict["node_embeddings"] = node_embeddings

        if "graph_embeddings" in embedding_keys:
            sampler.add_kv_to_sampled_subgraphs(graph_embeddings, key="graph_embeddings")
            embeddings_dict["graph_embeddings"] = graph_embeddings

    if "expression_vectors" in embedding_keys:
        def extract_expression_vector(subgraph_dict):
            node_info = subgraph_dict.get("node_info", {})
            biomarker_expressions = node_info.get("biomarker_expression", {})
            return np.array(list(biomarker_expressions.values()))

        expression_vectors = [
            extract_expression_vector(subgraph) for subgraph in sampler.get_all_sampled_subgraphs()
        ]
        sampler.add_kv_to_sampled_subgraphs(expression_vectors, key="expression_vectors")
        embeddings_dict["expression_vectors"] = expression_vectors

    # Handle concatenated embeddings
    for key in embedding_keys:
        if "+" in key:
            components = key.split("+")
            concatenated_embeddings = [
                np.concatenate([embeddings_dict[comp][i] for comp in components if comp in embeddings_dict], axis=None)
                for i in range(len(pyg_subgraphs))
            ]
            sampler.add_kv_to_sampled_subgraphs(concatenated_embeddings, key=key)

    return sampler

def save_pseudotime_to_csv(sampled_subgraphs, output_path):
    """Save region ID, cell ID, and pseudotime to a CSV file."""
    data = [
        {
            "region_id": subgraph["region_id"],
            "cell_id": subgraph["cell_id"],
            "pseudotime": subgraph.get("pseudotime", np.nan)
        }
        for subgraph in sampled_subgraphs
    ]
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Pseudotime data saved to {output_path}")


def perform_pseudo_time_analysis_pipeline(config, sampler):
    """Run pseudo-time analysis pipeline."""
    sampled_subgraph_dicts = sampler.get_all_sampled_subgraphs()

    for embedding_key in config.embedding_keys:
        embeddings = np.array([subgraph.get(embedding_key) for subgraph in sampled_subgraph_dicts])

        # Dimensionality reduction and clustering
        pca_embs, umap_embs, cluster_labels, _ = dimensionality_reduction_combo(
            embeddings, n_pca_components=10, cluster_method='kmeans', n_clusters=2, seed=42
        )

        # Attach cluster labels to subgraphs
        for i, subgraph in enumerate(sampled_subgraph_dicts):
            subgraph["cluster_label"] = cluster_labels[i]

        # Perform pseudo-time analysis for each start node
        for start_node in config.start_nodes:
            output_dir = os.path.join(config.output_dir, embedding_key, f"start_node_{start_node}")
            output_path = os.path.join(output_dir, "pseudotime.csv")

            pseudotime_results = perform_pseudotime_analysis(
                labels=cluster_labels,
                umap_embs=umap_embs,
                output_dir=output_dir,
                start=start_node,
                show_plots=config.show_plots
            )
            sampler.add_kv_to_sampled_subgraphs(pseudotime_results, key="pseudotime")

            # Save pseudotime data
            save_pseudotime_to_csv(sampled_subgraph_dicts, output_path)

            # Aggregate biomarker data
            aggregated_data = aggregate_biomarker_by_pseudotime_with_overlap(
                sampled_subgraph_dicts, config.biomarkers, num_bins=config.num_bins, overlap=config.overlap, use_bins=config.use_bins
            )

            # Plot and save biomarker trends
            plot_biomarker_vs_pseudotime(aggregated_data, output_dir,method=embedding_key,transforms=config.plotting_transform, use_bins=config.use_bins)

if __name__ == "__main__":
    # Pipeline Execution
    config = Config()
    dataset = initialize_dataset(config.data_root, config.dataset_kwargs)
    sampler = initialize_sampler(dataset, config.sampler_kwargs)
    sampler = prepare_embeddings(dataset, sampler, config.model_path, config.device, config.embedding_keys)

    perform_pseudo_time_analysis_pipeline(config, sampler)
