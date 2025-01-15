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
The script uses Hydra for configuration management. Parameters such as paths, embedding types, analysis settings, and constants are defined in `config` files. The configuration is modular, flexible, and reusable across different datasets and experiments.

# Requirements
- Custom dependencies: `spacegm`, `tic`

# Usage
To execute the pipeline:
1. Modify the `Config` files to set paths and desired parameters.
2. Run the script to generate outputs in structured directories.

"""

import os
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
from adapters.space_gm_adapter import CustomSubgraphSampler
from core.feature_extractor import extract_composition_generalized
from core.pseudotime_analysis import aggregate_data_by_pseudotime, perform_pseudotime_analysis
from core.expression_analysis import analyze_and_visualize_expression
from spacegm import CellularGraphDataset, GNN_pred
from spacegm.embeddings_analysis import get_embedding, get_composition_vector, dimensionality_reduction_combo
from utils.constant import GENERAL_CELL_TYPES
from utils.data_transform import normalize
from utils.visualization import plot_trends, plot_umap_vs_cell_types

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Initialize Functions
def initialize_dataset(config):
    """Initialize the CellularGraphDataset."""
    return CellularGraphDataset(config.general["data_root"], **config.dataset)


def initialize_sampler(dataset, config):
    """Initialize the CustomSubgraphSampler."""
    return CustomSubgraphSampler(dataset, **config.sampler)


def prepare_embeddings(dataset, sampler, config):
    """Prepare embeddings and add them to the sampled subgraphs."""
    embedding_keys = config.pipeline["embedding_preparation"]["keys"]
    pyg_subgraphs = sampler.get_subgraph_objects()
    embeddings_dict = {}

    # Composition vectors
    if "composition_vectors" in embedding_keys:
        composition_vectors = [
            get_composition_vector(data, n_cell_types=len(dataset.cell_type_mapping))
            for data in pyg_subgraphs
        ]
        sampler.add_kv_to_sampled_subgraphs(composition_vectors, key="composition_vectors")
        embeddings_dict["composition_vectors"] = composition_vectors
    
    # Expression vectors
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

    # Node and graph embeddings
    if any(k in embedding_keys for k in ["node_embeddings", "graph_embeddings"]):
        model = GNN_pred(
            num_layer=dataset.subgraph_size,
            num_node_type=len(dataset.cell_type_mapping) + 1,
            num_feat=dataset[0].x.shape[1] - 1,
            emb_dim=512,
            num_node_tasks=0,
            num_graph_tasks=1,
            node_embedding_output="last",
            drop_ratio=0.25,
            graph_pooling="max",
            gnn_type="gin",
        ).to(config.general["device"])
        model.load_state_dict(torch.load(config.general["model_path"]))

        node_embeddings, graph_embeddings, _ = get_embedding(model, pyg_subgraphs, config.general["device"])
        if "node_embeddings" in embedding_keys:
            sampler.add_kv_to_sampled_subgraphs(node_embeddings, key="node_embeddings")
            embeddings_dict["node_embeddings"] = node_embeddings
        if "graph_embeddings" in embedding_keys:
            sampler.add_kv_to_sampled_subgraphs(graph_embeddings, key="graph_embeddings")
            embeddings_dict["graph_embeddings"] = graph_embeddings

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
    """Run the pseudo-time analysis pipeline."""
    sampled_subgraph_dicts = sampler.get_all_sampled_subgraphs()

    for embedding_key in config.pipeline["embedding_preparation"]["keys"]:
        embeddings = np.array([subgraph.get(embedding_key) for subgraph in sampled_subgraph_dicts])

        # Dimensionality reduction and clustering
        _, umap_embs, cluster_labels, _ = dimensionality_reduction_combo(
            embeddings, n_pca_components=10, cluster_method="kmeans", n_clusters=config.general['n_clusters'], seed=config.general["random_seed"]
        )

        # Plot UMAP embeddings vs Cell Types
        if len(config.sampler['cell_type']) > 1:
            umap_output_dir = os.path.join(config.general["output_dir"], embedding_key, "umap")
            plot_umap_vs_cell_types(
                umap_embeddings=umap_embs,
                cell_types=[subgraph["node_info"].get("cell_type", -1) for subgraph in sampled_subgraph_dicts],
                cell_type_mapping=config.dataset["cell_type_mapping"],
                output_dir=umap_output_dir,
                show_plots=config.pipeline["pseudo_time_analysis"]["show_plots"],
            )


        # Attach cluster labels to subgraphs
        for i, subgraph in enumerate(sampled_subgraph_dicts):
            subgraph["cluster_label"] = cluster_labels[i]

        #################################
        # Expression Analysis Submodule #
        #################################
        output_dir = os.path.join(config.general["output_dir"], embedding_key, "expression")
        analyze_and_visualize_expression(
            sampled_subgraph_dicts,
            cluster_labels,
            biomarkers=config.pipeline["expression_analysis"]["biomarkers"],
            output_dir=output_dir,
            visualization_transform=config.pipeline["expression_analysis"]["visualization_transform"],
            visualization_kws=config.pipeline["expression_analysis"]["visualization_kwargs"],
        )

        ##################################
        # Pseudo-Time Analysis Submodule #
        ##################################
        pseudotime_output_dir = os.path.join(config.general["output_dir"], embedding_key, "pseudotime")
        for start_node in config.pipeline["pseudo_time_analysis"]["start_nodes"]:
            node_output_dir = os.path.join(pseudotime_output_dir,f'start_node_{start_node}')

            pseudotime_results = perform_pseudotime_analysis(
                labels=cluster_labels,
                umap_embs=umap_embs,
                output_dir=node_output_dir,
                start=start_node,
                show_plots=config.pipeline["pseudo_time_analysis"]["show_plots"],
            )
            sampler.add_kv_to_sampled_subgraphs(pseudotime_results, key="pseudotime")

            # Save pseudotime data
            pseudotime_csv = os.path.join(node_output_dir,"pseudotime.csv")
            save_pseudotime_to_csv(sampled_subgraph_dicts, pseudotime_csv)


            ###########################################
            # Biomarker expression Analysis Submodule #
            ###########################################
            def extract_biomarkers(subgraph):
                return subgraph["node_info"].get("biomarker_expression", {})

            biomarker_data = aggregate_data_by_pseudotime(
                sampled_subgraphs=sampled_subgraph_dicts,
                pseudotime=pseudotime_results,
                feature_extractor=extract_biomarkers,
                feature_keys=config.pipeline["pseudo_time_analysis"]["feature_keys"],
                num_bins=100,
                overlap=0.2,
                use_bins=True
            )
            plot_trends(
                biomarker_data,
                visualization_kwargs = config.pipeline["pseudo_time_analysis"]["visualization_kwargs"],
                output_dir = node_output_dir,
                ylabel = 'Biomarkers',
                show_plots = True,
                transforms = [normalize]
            )

            ###############################################
            # Neighborhood Composition Analysis Submodule #
            ###############################################
            # def extract_composition(subgraph):
            #     composition_vec = get_composition_vector(subgraph.get("subgraph", None), len(dataset.cell_type_mapping))
            #     return {cell_type: composition_vec[idx] for cell_type, idx in dataset.cell_type_mapping.items()}

            neighborhood_data = aggregate_data_by_pseudotime(
                sampled_subgraphs=sampled_subgraph_dicts,
                pseudotime=pseudotime_results,
                feature_extractor=lambda subgraph: extract_composition_generalized(
                    subgraph=subgraph,
                    general_cell_types=GENERAL_CELL_TYPES,
                    cell_type_mapping=config.constants["CELL_TYPE_MAPPING_UPMC"],
                ),
                feature_keys=config.pipeline["neighborhood_analysis"]["feature_keys"],
                num_bins=config.pipeline["neighborhood_analysis"]["num_bins"],
                overlap=config.pipeline["neighborhood_analysis"]["overlap"],
                use_bins=config.pipeline["neighborhood_analysis"]["use_bins"]
            )


            plot_trends(
                neighborhood_data,
                visualization_kwargs = config.pipeline["neighborhood_analysis"]["visualization_kwargs"],
                output_dir = node_output_dir,
                ylabel = 'Neighborhood Composition',
                show_plots = True,
                transforms = [normalize]
            )

@hydra.main(config_path="../config/pseudotime", config_name="example")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Initialize Dataset and Sampler
    dataset = CellularGraphDataset(cfg.general.data_root, **cfg.dataset)
    sampler = CustomSubgraphSampler(dataset, **cfg.sampler)

    # Prepare embeddings and run the analysis pipeline
    sampler = prepare_embeddings(dataset, sampler, cfg)
    perform_pseudo_time_analysis_pipeline(cfg, sampler)

if __name__ == "__main__":
    main()