# train_gnn.py
import csv
import os
import random
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
import hydra
from omegaconf import DictConfig
from tic.data.dataset import MicroEDataset
from tic.model.feature import biomarker_pretransform
from tic.model.model import GNN_pred
from tic.model.transform import mask_transform
import time

def set_seed(seed:int = 1120):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random seed {seed} has been set.")

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
        transform=mask_transform,                    # Transform function for masked learning tasks
        pre_transform=biomarker_pretransform,                          # Optionally, add a pre_transform function
        microe_neighbor_cutoff=cfg.dataset.microe_neighbor_cutoff,
        subset_cells=cfg.dataset.subset_cells,       # Whether to sample a subset of cells for large tissues
        center_cell_types=cfg.dataset.center_cell_types  # e.g., ["Tumor"] or other allowed types
    )
    
    # Create the PyG DataLoader to collate Data objects into a Batch.
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.trainer.batch_size,
        shuffle=True
    )
    
    return dataloader

# Function to initialize the model
def initialize_model(cfg, device):
    model = GNN_pred(
        num_layer=cfg.model.num_layer,
        emb_dim=cfg.model.emb_dim,
        gnn_type=cfg.model.gnn_type,
        drop_ratio=cfg.model.drop_ratio,
    ).to(device)  # Move model to the device
    return model

# Function to set up the optimizer
def initialize_optimizer(model, cfg):
    optimizer = optim.Adam(model.parameters(), lr=cfg.trainer.learning_rate)
    return optimizer

# Function to initialize TensorBoard logging
def initialize_tensorboard(cfg):
    log_dir = cfg.trainer.log_dir
    writer = SummaryWriter(log_dir)
    return writer

# Function to create the checkpoint directory if it doesn't exist
def create_checkpoint_dir(cfg):
    checkpoint_dir = cfg.trainer.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

# Function to load model and optimizer states from checkpoint
def load_checkpoint(model, optimizer, checkpoint_dir, epoch, step_counter, device):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}_{step_counter}.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)  # Load checkpoint to the device
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['step_counter'], checkpoint['total_loss']
    return step_counter, 0

# Function to write region-cell pairs to CSV
def log_region_cell_pairs(region_id, cell_id, csv_file="region_cell_pairs.csv"):
    # Ensure the file exists and open it in append mode
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # If the file does not exist, write the header first
        if not file_exists:
            writer.writerow(["Region ID", "Cell ID"])  # Writing header
        
        # Write the region-cell pair
        writer.writerow([region_id, cell_id])

# Function for training the model
def train(cfg: DictConfig):
    set_seed()
    
    # 1. Initialize device, Dataset, SubgraphSampler, Model, Optimizer, and TensorBoard
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    dataloader = initialize_dataset_and_dataloader(cfg)
    model = initialize_model(cfg, device)
    optimizer = initialize_optimizer(model, cfg)
    writer = initialize_tensorboard(cfg)
    checkpoint_dir = create_checkpoint_dir(cfg)
    region_cell_logging_file = os.path.join(cfg.trainer.log_dir, "region_cell_pairs.csv")

    

    # 2. Initialize training variables
    step_counter = 0
    total_loss = 0
    start_time = time.time()  # Start time for training duration

    # 3. Load from checkpoint if resuming
    if cfg.trainer.resume_training:
        step_counter, total_loss = load_checkpoint(model, optimizer, checkpoint_dir, cfg.trainer.start_epoch, step_counter, device)

    # 4. Training Loop based on epochs
    for epoch in range(cfg.trainer.start_epoch, cfg.trainer.num_epochs):
        print(f"Epoch {epoch + 1}/{cfg.trainer.num_epochs}")
        
        # Iterate over the DataLoader for each batch
        for batch in tqdm(dataloader, desc=f"Training Epoch: {epoch + 1}", unit="batch"):
            if batch is None:
                # Skip the batch if it contains invalid data (None)
                continue

            try:
                optimizer.zero_grad()  # Zero the gradients for each batch

                # Move batch to the device
                batch = batch.to(device)

                # Forward pass through the model
                graph_embedding, center_cell_pred = model(batch)

                center_cell_pred = center_cell_pred.flatten() # [B*22]
                mask = batch.mask  # Masked indices for biomarker expression values (where the mask is applied) [B*22] 
                ground_truth = batch.y  # Unmasked biomarker values (ground truth) [B*22]

                # Compute loss
                loss = model.compute_loss(center_cell_pred, ground_truth, mask)

                # Backward pass
                loss.backward()

                # Update parameters
                optimizer.step()

                total_loss += loss.item()
                step_counter += 1

                # Log loss to TensorBoard
                writer.add_scalar('Loss/train', loss.item(), step_counter)

                for i, (region_id, cell_id) in enumerate(zip(batch.region_id, batch.cell_id)):
                    log_region_cell_pairs(region_id, int(cell_id), csv_file=region_cell_logging_file)

                # Save checkpoints every 1000 steps
                if step_counter % 1000 == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch + 1}_{step_counter}.pth")
                    torch.save({
                        'epoch': epoch + 1,
                        'step_counter': step_counter,
                        'total_loss': total_loss,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_path)

                # Check if we've reached the number of steps
                if step_counter >= cfg.trainer.num_steps:
                    break

            except Exception as e:
                print(f"Error occurred at step {step_counter}: {e}")
                # Log the error to TensorBoard
                writer.add_text('Error', f"Error at step {step_counter}: {e}", step_counter)

                # Continue training after logging the error
                continue
    
        # Print the average loss after each epoch
        print(f"Epoch {epoch + 1}/{cfg.trainer.num_epochs}, Loss: {total_loss / (step_counter + 1):.4f}")
    
    # Training complete
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time // 60} minutes and {elapsed_time % 60} seconds.")

    # Close TensorBoard writer
    writer.close()
@hydra.main(config_path="../../config/train", config_name="main")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()