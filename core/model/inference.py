import os
import torch
from torch_geometric.loader import DataLoader
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from core.data.dataset import MicroEDataset
from core.model.feature import biomarker_pretransform
from core.model.model import GNN_pred

# Function to initialize the dataset and data loader
def initialize_dataset_and_dataloader(cfg):
    """
    Initializes the MicroEDataset and wraps it into a PyG DataLoader.
    """
    # Initialize the dataset using the configuration parameters.
    dataset = MicroEDataset(
        root=cfg.dataset.dataset_root,
        region_ids=cfg.dataset.region_ids,           # e.g., a list of region IDs
        k=cfg.dataset.k,                             # k-hop for microenvironment extraction
        transform=None,                    # Transform function for masked learning tasks
        pre_transform=biomarker_pretransform,                          # Optionally, add a pre_transform function
        microe_neighbor_cutoff=cfg.dataset.microe_neighbor_cutoff,
        subset_cells=cfg.dataset.subset_cells,       # Whether to sample a subset of cells for large tissues
        center_cell_types=cfg.dataset.center_cell_types  # e.g., ["Tumor"] or other allowed types
    )
    
    # Create the PyG DataLoader to collate Data objects into a Batch.
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.inference.batch_size,
        shuffle=False
    )
    
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
    dataloader = initialize_dataset_and_dataloader(cfg)
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

    print("Inference complete.")

@hydra.main(config_path="../../config/inference", config_name="main")
def main(cfg: DictConfig):
    run_inference(cfg)

if __name__ == "__main__":
    main()