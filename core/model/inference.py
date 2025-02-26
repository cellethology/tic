import os
import torch
import numpy as np
from core.model.model import GNN_pred
from core.model.data import TumorCellGraphDataset, get_region_cell_subgraph_dataloader
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from core.data.cell import CellInfo


# Function to initialize the dataset and data loader for inference
def initialize_dataset_and_dataloader(cfg, device):
    dataset = TumorCellGraphDataset(
        dataset_root=cfg.inference.dataset_root,
        node_features=cfg.inference.node_features,
    )
    dataloader = get_region_cell_subgraph_dataloader(dataset, batch_size=1, shuffle=False)
    return dataloader

# Function to initialize the model for inference
def initialize_model(cfg, device):
    model = GNN_pred(
        num_layer=cfg.model.num_layer,
        emb_dim=cfg.model.emb_dim,
        gnn_type=cfg.model.gnn_type,
        drop_ratio=cfg.model.drop_ratio,
    ).to(device)
    return model

# Function to load the trained model from checkpoint
def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from checkpoint: {checkpoint_path}")
    return model

######################
# Inference function #
######################
def run_inference(cfg: DictConfig):
    # Set device
    device = torch.device(cfg.inference.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Dataset, DataLoader, and Model
    dataloader = initialize_dataset_and_dataloader(cfg, device)
    model = initialize_model(cfg, device)

    # Load the trained model
    model = load_checkpoint(model, cfg.inference.checkpoint_path, device)
    model.eval()  # Set model to evaluation mode

    # Prepare output directory
    os.makedirs(cfg.inference.output_dir, exist_ok=True)

    with torch.no_grad():  # Disable gradient tracking during inference
        for batch in tqdm(dataloader, desc="Inference", unit="batch"):
            # Move batch to the correct device
            batch = batch.to(device)

            # Forward pass to get graph-level embeddings and center cell predictions
            graph_embeddings, center_cell_pred = model(batch)

            # Iterate over each cell in the batch to store the embedding and other info
            for i, (region_id, cell_id) in enumerate(zip(batch.region_id, batch.cell_id)):
                # Create a CellInfo object for the current cell
                cell_info = CellInfo(
                    region_id=region_id,
                    cell_id=int(cell_id),
                    raw_dir=os.path.join(cfg.inference.dataset_root, "voronoi"),
                    embedding=graph_embeddings[i].cpu().numpy(),  # Store the embedding
                    pseudotime=None  # Placeholder for pseudotime (can be updated later if needed)
                )

                # Save the CellInfo object
                embedding_filename = os.path.join(cfg.inference.output_dir, f"{region_id}_{cell_id}_embedding.pkl")
                cell_info.save(embedding_filename)  # Save the CellInfo object

    print("Inference complete.")

@hydra.main(config_path="../../config/inference", config_name="main")
def main(cfg: DictConfig):
    run_inference(cfg)

if __name__ == "__main__":
    main()