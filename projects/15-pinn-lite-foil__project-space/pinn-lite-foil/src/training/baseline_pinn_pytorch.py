"""
Baseline Physics-Informed Neural Network (PINN) for 2D Airfoil Flow - PyTorch Version
Implements the teacher model with 8 layers x 128 units.

Architecture:
    Input: [x, y, AoA] -> 8 x 128 (tanh) -> Output: [u, v, p]

Loss:
    L_total = lambda_data * L_data + lambda_pde * L_pde

Usage:
    python baseline_pinn_pytorch.py --data data/processed/training_data.h5 --epochs 10000 --output models/teacher/
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class NavierStokesPINN(nn.Module):
    """
    Physics-Informed Neural Network for 2D incompressible Navier-Stokes equations.

    Enforces:
        - Continuity: du/dx + dv/dy = 0
        - X-Momentum: u*du/dx + v*du/dy + (1/rho)*dp/dx - nu*(d2u/dx2 + d2u/dy2) = 0
        - Y-Momentum: u*dv/dx + v*dv/dy + (1/rho)*dp/dy - nu*(d2v/dx2 + d2v/dy2) = 0
    """

    def __init__(
        self,
        hidden_layers: int = 8,
        hidden_units: int = 128,
        activation: str = 'tanh',
        nu: float = 1e-3,
        rho: float = 1.0
    ):
        super().__init__()

        self.nu = nu
        self.rho = rho
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units

        # Build network
        layers = []
        in_features = 3  # x, y, aoa

        for i in range(hidden_layers):
            layers.append(nn.Linear(in_features, hidden_units))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'silu':
                layers.append(nn.SiLU())
            in_features = hidden_units

        self.hidden = nn.Sequential(*layers)

        # Output heads
        self.output_u = nn.Linear(hidden_units, 1)
        self.output_v = nn.Linear(hidden_units, 1)
        self.output_p = nn.Linear(hidden_units, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: [batch, 3] -> [batch, 3]"""
        h = self.hidden(x)
        u = self.output_u(h)
        v = self.output_v(h)
        p = self.output_p(h)
        return torch.cat([u, v, p], dim=-1)

    def compute_pde_residuals(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        aoa: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Navier-Stokes PDE residuals using automatic differentiation."""
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)

        inputs = torch.stack([x, y, aoa], dim=-1)
        outputs = self(inputs)

        u = outputs[:, 0]
        v = outputs[:, 1]
        p = outputs[:, 2]

        # First derivatives
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
        p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]

        # Second derivatives
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]

        # Continuity: du/dx + dv/dy = 0
        continuity = u_x + v_y

        # X-Momentum
        momentum_x = (
            u * u_x + v * u_y
            + (1.0 / self.rho) * p_x
            - self.nu * (u_xx + u_yy)
        )

        # Y-Momentum
        momentum_y = (
            u * v_x + v * v_y
            + (1.0 / self.rho) * p_y
            - self.nu * (v_xx + v_yy)
        )

        return continuity, momentum_x, momentum_y

    def get_config(self) -> Dict:
        return {
            'hidden_layers': self.hidden_layers,
            'hidden_units': self.hidden_units,
            'nu': self.nu,
            'rho': self.rho
        }


def load_dataset(
    filepath: str,
    normalize: bool = True,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Load dataset from HDF5 file (flat format)."""
    with h5py.File(filepath, 'r') as f:
        x = f['x'][:]
        y = f['y'][:]
        aoa = f['aoa'][:]
        u = f['u'][:]
        v = f['v'][:]
        p = f['p'][:]

    # Stack inputs and outputs
    inputs = np.stack([x, y, aoa], axis=-1).astype(np.float32)
    targets = np.stack([u, v, p], axis=-1).astype(np.float32)

    stats = {}

    if normalize:
        # Normalize inputs
        input_mean = inputs.mean(axis=0)
        input_std = inputs.std(axis=0) + 1e-8
        inputs = (inputs - input_mean) / input_std

        # Normalize outputs
        target_mean = targets.mean(axis=0)
        target_std = targets.std(axis=0) + 1e-8
        targets = (targets - target_mean) / target_std

        stats = {
            'input_mean': input_mean.tolist(),
            'input_std': input_std.tolist(),
            'target_mean': target_mean.tolist(),
            'target_std': target_std.tolist()
        }

    inputs = torch.tensor(inputs, dtype=torch.float32, device=device)
    targets = torch.tensor(targets, dtype=torch.float32, device=device)

    return inputs, targets, stats


def train_pinn(
    data_path: str,
    output_dir: str,
    epochs: int = 10000,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    hidden_layers: int = 8,
    hidden_units: int = 128,
    nu: float = 1e-3,
    lambda_data: float = 1.0,
    lambda_pde: float = 10.0,
    checkpoint_interval: int = 1000,
    log_interval: int = 100,
    device: str = 'cuda',
    quick_test: bool = False
) -> Dict:
    """Train baseline PINN model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if quick_test:
        epochs = min(100, epochs)
        checkpoint_interval = 50

    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print(f"\nLoading dataset from {data_path}...")
    inputs, targets, stats = load_dataset(data_path, normalize=True, device=device)

    # Split train/val (90/10)
    n_samples = len(inputs)
    n_train = int(0.9 * n_samples)

    indices = torch.randperm(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_inputs = inputs[train_idx]
    train_targets = targets[train_idx]
    val_inputs = inputs[val_idx]
    val_targets = targets[val_idx]

    print(f"  Train samples: {len(train_inputs):,}")
    print(f"  Val samples: {len(val_inputs):,}")

    # Save normalization stats
    with open(output_dir / 'normalization_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # Create data loaders
    train_dataset = TensorDataset(train_inputs, train_targets)
    val_dataset = TensorDataset(val_inputs, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    print(f"\nCreating PINN model ({hidden_layers} layers x {hidden_units} units)...")
    model = NavierStokesPINN(
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        nu=nu
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Optimizer with learning rate schedule
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # Training loop
    history = {
        'loss': [], 'data_loss': [], 'pde_loss': [],
        'val_loss': [], 'val_data_loss': []
    }

    print("\n" + "="*60)
    print("PINN Training Started")
    print("="*60)

    start_time = time.time()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_data_loss = 0.0
        epoch_pde_loss = 0.0
        n_batches = 0

        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()

            # Data loss
            predictions = model(batch_inputs)
            data_loss = torch.mean((predictions - batch_targets) ** 2)

            # PDE loss (on subset for efficiency)
            pde_batch_size = min(1024, len(batch_inputs))
            pde_idx = torch.randint(0, len(batch_inputs), (pde_batch_size,))

            x = batch_inputs[pde_idx, 0]
            y = batch_inputs[pde_idx, 1]
            aoa = batch_inputs[pde_idx, 2]

            cont, mom_x, mom_y = model.compute_pde_residuals(x, y, aoa)
            pde_loss = (
                torch.mean(cont ** 2) +
                torch.mean(mom_x ** 2) +
                torch.mean(mom_y ** 2)
            )

            # Total loss
            total_loss = lambda_data * data_loss + lambda_pde * pde_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_data_loss += data_loss.item()
            epoch_pde_loss += pde_loss.item()
            n_batches += 1

        # Average losses
        epoch_loss /= n_batches
        epoch_data_loss /= n_batches
        epoch_pde_loss /= n_batches

        history['loss'].append(epoch_loss)
        history['data_loss'].append(epoch_data_loss)
        history['pde_loss'].append(epoch_pde_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_data_loss = 0.0

        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                predictions = model(batch_inputs)
                loss = torch.mean((predictions - batch_targets) ** 2)
                val_loss += loss.item()
                val_data_loss += loss.item()

        val_loss /= len(val_loader)
        val_data_loss /= len(val_loader)

        history['val_loss'].append(val_loss)
        history['val_data_loss'].append(val_data_loss)

        # Learning rate scheduling
        if (epoch + 1) % 100 == 0:
            scheduler.step()

        # Logging
        if (epoch + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:5d} | "
                  f"Loss: {epoch_loss:.6f} | "
                  f"Data: {epoch_data_loss:.6f} | "
                  f"PDE: {epoch_pde_loss:.6f} | "
                  f"Val: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {elapsed:.1f}s")

        # Checkpointing
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1:05d}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / 'best_model.pt')

    elapsed = time.time() - start_time
    print("="*60)
    print(f"Training Complete! Total time: {elapsed:.1f}s")
    print("="*60 + "\n")

    # Save final model
    torch.save(model.state_dict(), output_dir / 'baseline_pinn.pt')

    # Save model config
    with open(output_dir / 'model_config.json', 'w') as f:
        json.dump(model.get_config(), f, indent=2)

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nModel saved to {output_dir}")
    print(f"  Final train loss: {history['loss'][-1]:.6f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.6f}")
    print(f"  Best val loss: {best_val_loss:.6f}")

    return history


def main():
    parser = argparse.ArgumentParser(description='Train baseline PINN (PyTorch)')
    parser.add_argument('--data', type=str, required=True, help='Path to HDF5 dataset')
    parser.add_argument('--output', type=str, default='models/teacher/', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10000, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_layers', type=int, default=8, help='Hidden layers')
    parser.add_argument('--hidden_units', type=int, default=128, help='Units per layer')
    parser.add_argument('--nu', type=float, default=1e-3, help='Kinematic viscosity')
    parser.add_argument('--lambda_data', type=float, default=1.0, help='Data loss weight')
    parser.add_argument('--lambda_pde', type=float, default=10.0, help='PDE loss weight')
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='Checkpoint frequency')
    parser.add_argument('--log_interval', type=int, default=100, help='Log frequency')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--quick_test', action='store_true', help='Quick test mode')

    args = parser.parse_args()

    train_pinn(
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_layers=args.hidden_layers,
        hidden_units=args.hidden_units,
        nu=args.nu,
        lambda_data=args.lambda_data,
        lambda_pde=args.lambda_pde,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        device=args.device,
        quick_test=args.quick_test
    )


if __name__ == '__main__':
    main()
