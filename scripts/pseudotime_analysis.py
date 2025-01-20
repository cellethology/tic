import os
import random
import pandas as pd
import numpy as np
from omegaconf import OmegaConf, DictConfig
from typing import Any
from spacegm.data import CellularGraphDataset
from adapters.embedding import prepare_embeddings
from adapters.sampler import CustomSubgraphSampler
from core.pseudotime_analysis import CellEmbedding, dimensionality_reduction_and_clustering, perform_pseudotime_analysis


# Function to save data in CSV format (abstracted for easier migration)
class DataSaver:
    @staticmethod
    def save_cluster_labels(embedding: CellEmbedding, cluster_labels, output_dir):
        """
        Save the cluster labels with region_id and cell_id to CSV.
        Args:
            embedding: The embedding containing region_id and cell_id information.
            cluster_labels: The cluster labels assigned to each cell.
            output_dir: The directory where the CSV will be saved.
        """
        cluster_labels_df = pd.DataFrame({
            "region_id": [region_id for region_id, _ in embedding.identifiers],
            "cell_id": [cell_id for _, cell_id in embedding.identifiers],
            "cluster": cluster_labels
        })
        os.makedirs(output_dir, exist_ok=True)
        cluster_labels_csv_path = os.path.join(output_dir, "cluster_labels.csv")
        cluster_labels_df.to_csv(cluster_labels_csv_path, index=False)
        print(f"Cluster labels saved to {cluster_labels_csv_path}.")

    @staticmethod
    def save_pseudotime_results(embedding: CellEmbedding, pseudotime, output_dir):
        """
        Save pseudotime analysis results to CSV.
        Args:
            embedding: The embedding containing region_id and cell_id information.
            pseudotime: The pseudotime values.
            output_dir: The directory where the pseudotime results will be saved.
        """
        pseudotime_results = []
        for idx, (region_id, cell_id) in enumerate(embedding.identifiers):
            pseudotime_results.append({
                "region_id": region_id,
                "cell_id": cell_id,
                "pseudotime": pseudotime[idx]
            })
        os.makedirs(output_dir, exist_ok=True)
        pseudotime_csv_path = os.path.join(output_dir, "pseudotime.csv")
        pseudotime_df = pd.DataFrame(pseudotime_results)
        pseudotime_df.to_csv(pseudotime_csv_path, index=False)
        print(f"Pseudotime results saved to {pseudotime_csv_path}.")


# Function to handle the overall pipeline
def run_pipeline(config: DictConfig):
    """
    Run the pseudotime analysis pipeline.
    
    Args:
        config (DictConfig): Configuration object with pipeline settings.
    """
    # Initialize random seed for reproducibility
    seed = config.general.get("random_seed", 42)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Random seed set to {seed}.")
    
    # Initialize Dataset
    dataset = CellularGraphDataset(config.general["data_root"], **config.dataset)
    print("Dataset initialized.")
    
    # Initialize Sampler and Sample Cells
    sampler = CustomSubgraphSampler(config.general.raw_dir, seed=seed)
    selected_cells = sampler.sample(
        total_samples=config.sampler.total_samples,
        regions=config.sampler.regions,
        cell_types=config.sampler.cell_types
    )
    print("Cells sampled.")

    # Prepare Embeddings
    cell_embedding = prepare_embeddings(dataset, selected_cells, config.embedding)
    print("Embeddings prepared.")

    # Create Output Directories
    output_dir = config.general.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Process Each Embedding Method
    for embedding_key in config.embedding.methods:
        print(f"Processing embedding: {embedding_key}")
        embedding_output_dir = os.path.join(output_dir, "embedding_analysis", embedding_key)
        os.makedirs(embedding_output_dir, exist_ok=True)

        ###########################################
        # Dimensionality Reduction and Clustering #
        ###########################################

        reduced_data, umap_embs, cluster_labels, _ = dimensionality_reduction_and_clustering(
            cell_embedding.embeddings[embedding_key],
            n_pca_components=config.embedding.get("n_pca_components", 20),
            cluster_method=config.embedding.get("cluster_method", "kmeans"),
            n_clusters=config.embedding.get("n_clusters", 10),
            seed=seed
        )
        cell_embedding.attributes["cluster_labels"] = cluster_labels.tolist()
        print(f"Dimensionality reduction and clustering completed for {embedding_key}.")

        # Save Cluster Labels
        DataSaver.save_cluster_labels(cell_embedding, cluster_labels, embedding_output_dir)

        #######################
        # Pseudotime Analysis #
        #######################
        
        pseudotime_output_dir = os.path.join(embedding_output_dir, "pseudotime")
        os.makedirs(pseudotime_output_dir, exist_ok=True)

        for start_node in config.pseudotime.start_nodes:
            pseudotime_dir = os.path.join(pseudotime_output_dir, f"start_node_{start_node}")
            os.makedirs(pseudotime_dir, exist_ok=True)
            pseudotime = perform_pseudotime_analysis(
                labels=cluster_labels,
                umap_embs=umap_embs,
                output_dir=pseudotime_dir,
                start=start_node,
                show_plots=config.general.show_plots,
            )
            print(f"Pseudotime analysis completed for {embedding_key} with start node {start_node}.")

            # Save Pseudotime Results
            DataSaver.save_pseudotime_results(cell_embedding, pseudotime, pseudotime_dir)


if __name__ == "__main__":
    config_path = "./config/pseudotime/new.yaml"
    config = OmegaConf.load(config_path)
    run_pipeline(config)
