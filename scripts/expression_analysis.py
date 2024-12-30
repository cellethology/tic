# Expression Analysis Pipeline

"""
# README
This notebook implements a pipeline for analyzing biomarker expression in cellular graph datasets. It includes the following steps:

1. **Dataset Initialization**:
   - Load and preprocess cellular graph datasets.

2. **Subgraph Sampling**:
   - Sample subgraphs from the dataset based on specific parameters (e.g., cell type, region).

3. **Embedding Preparation**:
   - Generate various embeddings for sampled subgraphs:
     - Expression vectors
     - Composition vectors
     - Expression+Composition vectors

4. **Expression Analysis**:
   - Perform dimensionality reduction and clustering.
   - Analyze and visualize biomarker expression levels across clusters.

5. **Visualization and Output**:
   - Generate bar charts for cluster-wise biomarker averages.
   - Save results (e.g., cluster labels, biomarker averages) to structured directories for future analysis.

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adapters.space_gm_adapter import CustomSubgraphSampler
from core.expression_analysis import analyze_and_visualize_expression, plot_biomarker_bar_chart
from spacegm.utils import BIOMARKERS_UPMC, CELL_TYPE_FREQ_UPMC, CELL_TYPE_MAPPING_UPMC
from spacegm import CellularGraphDataset
from spacegm.embeddings_analysis import (
    get_embedding,
    get_composition_vector,
    dimensionality_reduction_combo
)
from utils.data_transform import normalize

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration Classes
class DatasetConfig:
    def __init__(self):
        self.data_root = "/root/autodl-tmp/Data/Space-Gm/Processed_Dataset/UPMC"
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

class SamplerConfig:
    def __init__(self):
        self.total_samples = 1000
        self.cell_type = 9
        self.region_id = None
        self.batch_size = 64
        self.num_workers = 8
        self.output_csv = "/root/tic-sci/data/embedding_analysis/expression/test/sampled_subgraphs.csv"
        self.include_node_info = True
        self.random_seed = 42

class PipelineConfig:
    def __init__(self):
        self.output_dir = "/root/tic-sci/data/embedding_analysis/expression/test"
        self.embedding_keys = ["expression_vectors", "composition_vectors"]
        self.cluster_method = "kmeans"
        self.n_clusters = 5
        self.biomarkers = ["ASMA", "PANCK", "VIMENTIN", "PODOPLANIN"]
        self.visualization_transform = []
        self.visualization_kwargs = ["PANCK", "avg(ASMA+VIMENTIN+PODOPLANIN)"]

# Initialization Functions
def initialize_dataset(config: DatasetConfig):
    """Initialize the CellularGraphDataset."""
    return CellularGraphDataset(config.data_root, **config.dataset_kwargs)

def initialize_sampler(dataset, config: SamplerConfig):
    """Initialize the CustomSubgraphSampler."""
    return CustomSubgraphSampler(dataset, **config.__dict__)

def prepare_embeddings(dataset, sampler, embedding_keys):
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

    return sampler

def perform_expression_analysis_pipeline(dataset_config, sampler_config, pipeline_config):
    """Run expression analysis pipeline."""
    # Initialize dataset and sampler
    dataset = initialize_dataset(dataset_config)
    sampler = initialize_sampler(dataset, sampler_config)

    # Prepare embeddings
    sampler = prepare_embeddings(dataset, sampler, pipeline_config.embedding_keys)
    sampled_subgraph_dicts = sampler.get_all_sampled_subgraphs()

    for embedding_key in pipeline_config.embedding_keys:
        embeddings = np.array([subgraph.get(embedding_key) for subgraph in sampled_subgraph_dicts])

        # Dimensionality reduction and clustering
        _, _, cluster_labels, _ = dimensionality_reduction_combo(
            embeddings, n_pca_components=10, cluster_method=pipeline_config.cluster_method,
            n_clusters=pipeline_config.n_clusters, seed=42
        )

        # Attach cluster labels to subgraphs
        for i, subgraph in enumerate(sampled_subgraph_dicts):
            subgraph["cluster_label"] = cluster_labels[i]

        # Analyze and visualize biomarker expression
        output_dir = os.path.join(pipeline_config.output_dir, embedding_key)
        analyze_and_visualize_expression(
            sampled_subgraph_dicts, cluster_labels, pipeline_config.biomarkers, output_dir=output_dir,
            visualization_kws=pipeline_config.visualization_kwargs,
            visualization_transform=pipeline_config.visualization_transform
        )

if __name__ == "__main__":
    # Pipeline Execution
    dataset_config = DatasetConfig()
    sampler_config = SamplerConfig()
    pipeline_config = PipelineConfig()

    perform_expression_analysis_pipeline(dataset_config, sampler_config, pipeline_config)
