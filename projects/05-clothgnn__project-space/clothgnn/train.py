"""
Simple training script for ClothGNN.
Usage: python train.py --data_path <path> --epochs 100 --batch_size 1
"""
import argparse
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch_geometric.data import Data
import h5py
from tqdm import tqdm

from models.clothgnn import ClothGNNModel
from utils.losses import position_loss, total_loss, compute_rest_lengths


def load_trajectory_data(data_path):
    """Load HDF5 trajectory dataset."""
    with h5py.File(data_path, 'r') as f:
        trajectory = torch.tensor(f['trajectory'][:], dtype=torch.float32)
        edge_index = torch.tensor(f['edge_index'][:], dtype=torch.long)
        rest_positions = torch.tensor(f['rest_positions'][:], dtype=torch.float32)
    return trajectory, edge_index, rest_positions


def create_node_features(positions, velocities=None, dim=16):
    """Create node features from positions and velocities."""
    if velocities is None:
        velocities = torch.zeros_like(positions)
    # Concatenate position (3) + velocity (3) + padding (10) = 16
    features = torch.cat([positions, velocities], dim=1)
    padding = torch.zeros(len(positions), dim - 6, device=positions.device)
    return torch.cat([features, padding], dim=1)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    # Load data
    trajectory, edge_index, rest_positions = load_trajectory_data(args.data_path)
    rest_lengths = compute_rest_lengths(rest_positions, edge_index)

    num_frames = len(trajectory)
    num_nodes = len(rest_positions)

    print(f"Dataset: {num_frames} frames, {num_nodes} vertices")

    # Move to device
    trajectory = trajectory.to(device)
    edge_index = edge_index.to(device)
    rest_lengths = rest_lengths.to(device)

    # Initialize model
    model = ClothGNNModel(node_feat_dim=16, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        # Reset hidden state for each sequence
        h = model.init_hidden(num_nodes, device)

        # Iterate through frames
        for t in range(num_frames - 1):
            current_pos = trajectory[t]
            next_pos = trajectory[t + 1]
            target_disp = next_pos - current_pos

            # Create features
            if t > 0:
                velocity = current_pos - trajectory[t - 1]
            else:
                velocity = torch.zeros_like(current_pos)

            node_features = create_node_features(current_pos, velocity)
            data = Data(x=node_features, edge_index=edge_index, pos=current_pos)

            # Forward pass
            pred_disp, h = model(data, h)

            # Compute loss
            pred_pos = current_pos + pred_disp
            loss, _ = total_loss(
                pred_disp, target_disp, pred_pos,
                edge_index, rest_lengths,
                lambda_p=1.0, lambda_e=args.lambda_e
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Detach hidden state for truncated BPTT
            h = h.detach()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / (num_frames - 1)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = f"checkpoint_epoch{epoch+1}.pt"
            model.save_checkpoint(
                checkpoint_path,
                optimizer=optimizer,
                epoch=epoch+1,
                loss=avg_loss
            )
            print(f"  Saved checkpoint: {checkpoint_path}")

    # Save final model
    model.save_checkpoint("model_final.pt", optimizer=optimizer, epoch=args.epochs, loss=avg_loss)
    print(f"Training complete. Final loss: {avg_loss:.6f}")
    print(f"Saved final model: model_final.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ClothGNN model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to HDF5 dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--lambda_e', type=float, default=0.1, help='Edge length loss weight')
    parser.add_argument('--save_every', type=int, default=50, help='Save checkpoint every N epochs')

    args = parser.parse_args()
    train(args)
