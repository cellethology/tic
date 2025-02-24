import os
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig
from adapters.model import GNN_pred
from adapters.data import TumorCellGraphDataset, SubgraphSampler
from adapters.transform import mask_biomarker_expression

@hydra.main(config_path="../config/train", config_name="main")
def train(cfg: DictConfig):
    # 1. Initialize Dataset and SubgraphSampler
    dataset = TumorCellGraphDataset(
        dataset_root=cfg.dataset.dataset_root,
        node_features=cfg.dataset.node_features,
        transform=mask_biomarker_expression,
    )

    # Create SubgraphSampler instance
    sampler = SubgraphSampler(dataset, k=3, batch_size=cfg.trainer.batch_size, shuffle=True, cell_types=['Tumor'])

    # 2. Initialize Model
    model = GNN_pred(num_layer=cfg.model.num_layer,
                     emb_dim=cfg.model.emb_dim,
                     gnn_type=cfg.model.gnn_type)

    # 3. Set up the Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.trainer.learning_rate)

    # 4. Setup TensorBoard
    log_dir = cfg.trainer.log_dir
    writer = SummaryWriter(log_dir)

    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = cfg.trainer.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 5. Training Loop based on cases (steps * batch_size)
    step_counter = 0
    total_loss = 0
    while step_counter < cfg.trainer.num_steps:
        for batch in tqdm(sampler, desc=f"Training Steps: {step_counter}/{cfg.trainer.num_steps}", unit="batch"):
            optimizer.zero_grad()  # Zero the gradients for each batch

            # Forward pass through the model
            node_pred = model(batch)
            # Masked indices for biomarker expression values (where the mask is applied)
            mask = batch.mask

            # Extract ground truth biomarker expression values
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

            # Save checkpoints every 10000 steps
            if step_counter % 1000 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step_counter}.pth")
                torch.save(model.state_dict(), checkpoint_path)

            if step_counter >= cfg.trainer.num_steps:
                break

        # Print the average loss after each segment of steps
        print(f"Step {step_counter}/{cfg.trainer.num_steps}, Loss: {total_loss / (step_counter+1):.4f}")

    writer.close()

if __name__ == "__main__":
    train()