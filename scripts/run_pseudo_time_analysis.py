import os
import random
from omegaconf import OmegaConf, DictConfig
from typing import Any
import pandas as pd
import numpy as np

from spacegm.data import CellularGraphDataset
from adapters.embedding import prepare_embeddings
from adapters.sampler import CustomSubgraphSampler
from core.pseudotime_analysis import dimensionality_reduction_and_clustering, perform_pseudotime_analysis


def run_pipeline(config: DictConfig):
    """
    Run the pseudotime analysis pipeline.

    Args:
        config (DictConfig): Configuration object with pipeline settings.
    """
    print(OmegaConf.to_yaml(config))
    
    # Set global random seed
    seed = config.general.get("random_seed", 42)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Random seed set to {seed}.")

    # Step 1: Initialize Dataset
    dataset = CellularGraphDataset(config.general["data_root"], **config.dataset)
    print("Dataset initialized.")

    # Step 2: Initialize Sampler
    sampler = CustomSubgraphSampler(config.general.raw_dir, seed=seed)
    selected_cells = sampler.sample(
        total_samples=config.sampler.total_samples,
        regions=config.sampler.regions,
        cell_types=config.sampler.cell_types
    )
    print("Cells sampled.")

    # Step 3: Prepare Embeddings
    cell_embedding = prepare_embeddings(dataset, selected_cells, config.embedding)
    print("Embeddings prepared.")

    # Step 4: Analyze Each Embedding Method
    output_dir = config.general.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    for embedding_key in config.embedding.methods:
        print(f"Processing embedding: {embedding_key}")
        embedding_output_dir = os.path.join(output_dir, "embedding_analysis", embedding_key)
        os.makedirs(embedding_output_dir, exist_ok=True)
        
        # Dimensionality Reduction and Clustering
        reduced_data, umap_embs, cluster_labels, tools = dimensionality_reduction_and_clustering(
            cell_embedding.embeddings[embedding_key],
            n_pca_components=config.embedding.get("n_pca_components", 20),
            cluster_method=config.embedding.get("cluster_method", "kmeans"),
            n_clusters=config.embedding.get("n_clusters", 10),
            seed=seed  
        )
        cell_embedding.attributes["cluster_labels"] = cluster_labels.tolist()
        print(f"Dimensionality reduction and clustering completed for {embedding_key}.")

        # Pseudotime Analysis
        pseudotime_output_dir = os.path.join(embedding_output_dir, "pseudotime")
        os.makedirs(pseudotime_output_dir, exist_ok=True)

        for start_node in config.pseudotime.start_nodes:
            pseudotime_dir = os.path.join(pseudotime_output_dir, f"start_node_{start_node}")
            os.makedirs(pseudotime_dir, exist_ok=True)
            pseudotime_results = []
            pseudotime = perform_pseudotime_analysis(
                labels=cluster_labels,
                umap_embs=umap_embs,
                output_dir=pseudotime_dir,
                start=start_node,
                show_plots=config.general.show_plots,
            )
            print(f"Pseudotime analysis completed for {embedding_key} with start node {start_node}.")

            # Collect pseudotime results for saving
            for idx, (region_id, cell_id) in enumerate(cell_embedding.identifiers):
                pseudotime_results.append({
                    "region_id": region_id,
                    "cell_id": cell_id,
                    "pseudotime": pseudotime[idx]
                })
            # Save pseudotime results to CSV
            pseudotime_csv_path = os.path.join(pseudotime_dir, "pseudotime.csv")
            pseudotime_df = pd.DataFrame(pseudotime_results)
            pseudotime_df.to_csv(pseudotime_csv_path, index=False)
            print(f"Pseudotime results saved to {pseudotime_csv_path} for {embedding_key} with start node {start_node}.")


if __name__ == "__main__":
    config_path = "./config/pseudotime/new.yaml"
    config = OmegaConf.load(config_path)
    run_pipeline(config)
