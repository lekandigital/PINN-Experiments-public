"""
High-level inference wrapper for ClothGNN.
Usage: python inference.py --model checkpoint.pt --input mesh.h5 --output predictions.h5
"""
import argparse
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import h5py
import numpy as np
from torch_geometric.data import Data

from models.clothgnn import ClothGNNModel


class ClothGNNInference:
    """Inference wrapper with preprocessing and batching."""

    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model, checkpoint = ClothGNNModel.load_checkpoint(model_path, self.device)
        self.model.eval()
        self.hidden_state = None
        print(f"Loaded model from {model_path}")
        print(f"  Hidden dim: {checkpoint['config']['hidden_dim']}")
        if 'epoch' in checkpoint:
            print(f"  Trained for: {checkpoint['epoch']} epochs")
        if 'loss' in checkpoint:
            print(f"  Final loss: {checkpoint['loss']:.6f}")

    def reset_hidden(self, num_nodes):
        """Reset hidden state for new sequence."""
        self.hidden_state = self.model.init_hidden(num_nodes, self.device)

    def predict_step(self, positions, edge_index, velocity=None):
        """Predict one frame displacement."""
        num_nodes = len(positions)

        if self.hidden_state is None:
            self.reset_hidden(num_nodes)

        # Create features
        if velocity is None:
            velocity = torch.zeros_like(positions)

        features = torch.cat([positions, velocity], dim=1)
        padding = torch.zeros(num_nodes, 10, device=self.device)
        node_features = torch.cat([features, padding], dim=1)

        data = Data(
            x=node_features,
            edge_index=edge_index.to(self.device),
            pos=positions.to(self.device)
        )

        with torch.no_grad():
            pred_disp, self.hidden_state = self.model(data, self.hidden_state)

        return pred_disp.cpu()

    def rollout(self, initial_positions, edge_index, num_steps):
        """Multi-step prediction rollout."""
        self.reset_hidden(len(initial_positions))

        positions = initial_positions.clone().to(self.device)
        edge_index = edge_index.to(self.device)
        trajectory = [positions.cpu().clone()]

        print(f"Running rollout for {num_steps} steps...")
        for step in range(num_steps):
            velocity = trajectory[-1] - trajectory[-2] if len(trajectory) > 1 else torch.zeros_like(positions.cpu())
            displacement = self.predict_step(positions, edge_index, velocity.to(self.device))
            positions = positions + displacement.to(self.device)
            trajectory.append(positions.cpu().clone())

            if (step + 1) % 10 == 0:
                print(f"  Step {step + 1}/{num_steps}")

        return torch.stack(trajectory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Model checkpoint path')
    parser.add_argument('--input', required=True, help='Input mesh HDF5')
    parser.add_argument('--output', required=True, help='Output predictions HDF5')
    parser.add_argument('--steps', type=int, default=60, help='Prediction steps')
    args = parser.parse_args()

    # Load input
    print(f"Loading input from {args.input}")
    with h5py.File(args.input, 'r') as f:
        positions = torch.tensor(f['rest_positions'][:], dtype=torch.float32)
        edge_index = torch.tensor(f['edge_index'][:], dtype=torch.long)

    print(f"  Mesh: {len(positions)} vertices, {edge_index.shape[1]} edges")

    # Run inference
    inferencer = ClothGNNInference(args.model)
    trajectory = inferencer.rollout(positions, edge_index, args.steps)

    # Save output
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('trajectory', data=trajectory.numpy())
        f.create_dataset('edge_index', data=edge_index.numpy())

    print(f"Saved {args.steps + 1} frames to {args.output}")


if __name__ == '__main__':
    main()
