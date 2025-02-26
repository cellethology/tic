import os
import random
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig
from core.model.model import GNN_pred
from core.model.data import TumorCellGraphDataset, get_region_cell_subgraph_dataloader
from core.model.transform import mask_biomarker_expression
import time

def set_seed(seed:int = 1120):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random seed {seed} has been set.")

# Function to initialize the dataset and data loader
def initialize_dataset_and_dataloader(cfg, device):
    dataset = TumorCellGraphDataset(
        dataset_root=cfg.dataset.dataset_root,
        node_features=cfg.dataset.node_features,
        transform=mask_biomarker_expression,
        cell_types=cfg.dataset.cell_types  
    )
    dataloader = get_region_cell_subgraph_dataloader(dataset, cfg.trainer.batch_size, shuffle=True)
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

# Function for training the model
def train(cfg: DictConfig):
    set_seed()
    
    # 1. Initialize device, Dataset, SubgraphSampler, Model, Optimizer, and TensorBoard
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    dataloader = initialize_dataset_and_dataloader(cfg, device)
    model = initialize_model(cfg, device)
    optimizer = initialize_optimizer(model, cfg)
    writer = initialize_tensorboard(cfg)
    checkpoint_dir = create_checkpoint_dir(cfg)

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
            try:
                optimizer.zero_grad()  # Zero the gradients for each batch

                # Move batch to the device
                batch = batch.to(device)

                # Forward pass through the model
                graph_embedding, center_cell_pred = model(batch)
                mask = batch.mask  # Masked indices for biomarker expression values (where the mask is applied)
                ground_truth = batch.y  # Unmasked biomarker values (ground truth)

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


@hydra.main(config_path="../config/train", config_name="main")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()