import os
import random
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig
from adapters.model import GNN_pred
from adapters.data import TumorCellGraphDataset, WeightedSubgraphSampler
from adapters.transform import mask_biomarker_expression
import time

def set_seed(seed:int = 1120):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random seed {seed} has been set.")

# Function to initialize the dataset and sampler
def initialize_dataset_and_sampler(cfg):
    dataset = TumorCellGraphDataset(
        dataset_root=cfg.dataset.dataset_root,
        node_features=cfg.dataset.node_features,
        transform=mask_biomarker_expression,
    )
    sampler = WeightedSubgraphSampler(dataset, k=3, batch_size=cfg.trainer.batch_size, shuffle=True, cell_types=['Tumor'])
    return dataset, sampler


# Function to initialize the model
def initialize_model(cfg):
    model = GNN_pred(
        num_layer=cfg.model.num_layer,
        emb_dim=cfg.model.emb_dim,
        gnn_type=cfg.model.gnn_type
    )
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


# Function for training the model
def train(cfg: DictConfig):
    set_seed()
    # 1. Initialize Dataset, SubgraphSampler, Model, Optimizer, and TensorBoard
    dataset, sampler = initialize_dataset_and_sampler(cfg)
    model = initialize_model(cfg)
    optimizer = initialize_optimizer(model, cfg)
    writer = initialize_tensorboard(cfg)
    checkpoint_dir = create_checkpoint_dir(cfg)

    # 2. Initialize training variables
    step_counter = 0
    total_loss = 0
    start_time = time.time()  # Start time for training duration

    # 3. Training Loop based on steps * batch_size
    while step_counter < cfg.trainer.num_steps:
        for batch in tqdm(sampler, desc=f"Training Steps: {step_counter}/{cfg.trainer.num_steps}", unit="batch"):
            optimizer.zero_grad()  # Zero the gradients for each batch

            # Forward pass through the model
            node_pred = model(batch)
            mask = batch.mask  # Masked indices for biomarker expression values (where the mask is applied)

            ground_truth = batch.y  # Unmasked biomarker values (ground truth)

            # Compute loss
            loss = model.compute_loss(node_pred, ground_truth, mask)

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
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step_counter}.pth")
                torch.save(model.state_dict(), checkpoint_path)

            # Check if we've reached the number of steps
            if step_counter >= cfg.trainer.num_steps:
                break

        # Print the average loss after each segment of steps
        print(f"Step {step_counter}/{cfg.trainer.num_steps}, Loss: {total_loss / (step_counter+1):.4f}")
    
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