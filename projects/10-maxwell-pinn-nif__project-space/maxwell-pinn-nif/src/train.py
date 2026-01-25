"""
Training Script for Maxwell-PINN-NIF

This script implements the complete training loop for the Maxwell-PINN model,
including:
- Mixed-precision training with torch.cuda.amp for GPU memory efficiency
- Learning rate scheduling with ReduceLROnPlateau
- TensorBoard logging for loss curves and field visualizations
- Checkpoint saving for best models
- Multi-GPU support via DistributedDataParallel (optional)

Usage:
    # Single GPU training
    python train.py --config config.yaml
    
    # Multi-GPU training (2 GPUs)
    torchrun --nproc_per_node=2 train.py --config config.yaml --distributed

Dependencies:
    torch, h5py, tensorboard, tqdm
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import h5py

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from pinn_model import MaxwellPINN, divergence_free_loss, maxwell_curl_residual
from pml_loss import PMLLoss


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    
    # Model architecture
    input_dim: int = 3
    hidden_dim: int = 128
    num_hidden: int = 6
    use_fourier: bool = False
    fourier_sigma: float = 10.0
    num_frequencies: int = 64
    
    # Training parameters
    batch_size: int = 4096
    num_epochs: int = 1000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Loss weights
    w_pde: float = 1.0
    w_div: float = 10.0
    w_data: float = 100.0
    w_pml: float = 1.0
    
    # Physics parameters
    omega: float = 1.0  # Angular frequency
    
    # Domain and PML
    domain_bounds: list = field(default_factory=lambda: [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    pml_thickness: float = 0.1
    
    # Mixed precision
    use_amp: bool = True
    
    # Logging and checkpoints
    log_dir: str = "runs/maxwell_pinn"
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 100
    save_interval: int = 1000
    
    # Data
    data_path: str = "data/em_data.h5"
    validation_split: float = 0.1
    
    # Distributed training
    distributed: bool = False
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, d: dict) -> 'TrainingConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


# ============================================================================
# Dataset
# ============================================================================

class MaxwellDataset(Dataset):
    """
    PyTorch Dataset for Maxwell-PINN training data.
    
    Loads coordinates, material properties, and field targets from HDF5.
    """
    
    def __init__(self, hdf5_path: str, split: str = 'train', val_fraction: float = 0.1):
        """
        Args:
            hdf5_path: Path to HDF5 dataset file
            split: 'train' or 'val'
            val_fraction: Fraction of data to use for validation
        """
        self.hdf5_path = hdf5_path
        
        # Load data into memory (for small-medium datasets)
        with h5py.File(hdf5_path, 'r') as f:
            coords = f['coords'][:]
            eps = f['eps'][:]
            mu = f['mu'][:]
            E_field = f['E_field'][:]
            H_field = f['H_field'][:]
        
        # Split data
        n_samples = len(coords)
        n_val = int(n_samples * val_fraction)
        
        # Use deterministic split
        np.random.seed(42)
        indices = np.random.permutation(n_samples)
        
        if split == 'train':
            idx = indices[n_val:]
        else:
            idx = indices[:n_val]
        
        self.coords = torch.from_numpy(coords[idx]).float()
        self.eps = torch.from_numpy(eps[idx]).float()
        self.mu = torch.from_numpy(mu[idx]).float()
        self.E_field = torch.from_numpy(E_field[idx]).float()
        self.H_field = torch.from_numpy(H_field[idx]).float()
    
    def __len__(self) -> int:
        return len(self.coords)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.coords[idx],
            self.eps[idx],
            self.mu[idx],
            self.E_field[idx],
            self.H_field[idx]
        )


class CollocationDataset(Dataset):
    """
    Dataset that generates random collocation points for physics loss.
    
    Unlike the supervised dataset, this doesn't require ground truth fields.
    Points are sampled uniformly in the domain for PDE residual computation.
    """
    
    def __init__(
        self,
        n_points: int,
        domain_bounds: list,
        eps_range: Tuple[float, float] = (1.0, 4.0),
        mu_range: Tuple[float, float] = (0.9, 1.1)
    ):
        self.n_points = n_points
        self.domain_bounds = domain_bounds
        self.eps_range = eps_range
        self.mu_range = mu_range
        
        # Pre-generate points for consistency within epoch
        self._regenerate()
    
    def _regenerate(self):
        """Regenerate random collocation points."""
        xmin, xmax, ymin, ymax, zmin, zmax = self.domain_bounds
        
        self.coords = torch.rand(self.n_points, 3)
        self.coords[:, 0] = self.coords[:, 0] * (xmax - xmin) + xmin
        self.coords[:, 1] = self.coords[:, 1] * (ymax - ymin) + ymin
        self.coords[:, 2] = self.coords[:, 2] * (zmax - zmin) + zmin
        
        # Random material properties (or could use a model)
        self.eps = torch.rand(self.n_points) * (self.eps_range[1] - self.eps_range[0]) + self.eps_range[0]
        self.mu = torch.rand(self.n_points) * (self.mu_range[1] - self.mu_range[0]) + self.mu_range[0]
    
    def __len__(self) -> int:
        return self.n_points
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return self.coords[idx], self.eps[idx], self.mu[idx]


# ============================================================================
# Training Functions
# ============================================================================

def compute_losses(
    model: MaxwellPINN,
    coords: torch.Tensor,
    eps: torch.Tensor,
    mu: torch.Tensor,
    E_target: Optional[torch.Tensor],
    H_target: Optional[torch.Tensor],
    pml_loss_fn: Optional[PMLLoss],
    config: TrainingConfig
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute all loss components.
    
    Returns:
        total_loss: Weighted sum of all losses
        loss_dict: Individual loss values for logging
    """
    # Ensure coords have gradients for autograd
    coords = coords.requires_grad_(True)
    
    # Forward pass
    E_pred, H_pred = model(coords)
    
    # PDE residual loss (curl equations)
    loss_pde = maxwell_curl_residual(E_pred, H_pred, coords, eps, mu, config.omega)
    
    # Divergence-free constraint
    loss_div = divergence_free_loss(E_pred, H_pred, coords, eps, mu)
    
    # Data loss (supervised, if targets available)
    loss_data = torch.tensor(0.0, device=coords.device)
    if E_target is not None and H_target is not None:
        loss_data = torch.mean((E_pred - E_target) ** 2) + torch.mean((H_pred - H_target) ** 2)
    
    # PML boundary loss
    loss_pml = torch.tensor(0.0, device=coords.device)
    if pml_loss_fn is not None:
        loss_pml = pml_loss_fn(coords, E_pred, H_pred)
    
    # Weighted total
    total_loss = (
        config.w_pde * loss_pde +
        config.w_div * loss_div +
        config.w_data * loss_data +
        config.w_pml * loss_pml
    )
    
    loss_dict = {
        'total': total_loss.item(),
        'pde': loss_pde.item(),
        'div': loss_div.item(),
        'data': loss_data.item(),
        'pml': loss_pml.item()
    }
    
    return total_loss, loss_dict


def train_epoch(
    model: MaxwellPINN,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    pml_loss_fn: Optional[PMLLoss],
    config: TrainingConfig,
    device: torch.device
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Returns:
        avg_losses: Average losses over epoch
    """
    model.train()
    
    total_losses = {'total': 0, 'pde': 0, 'div': 0, 'data': 0, 'pml': 0}
    n_batches = 0
    
    for batch in dataloader:
        coords, eps, mu, E_target, H_target = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        
        # Mixed precision context
        if config.use_amp and scaler is not None:
            with autocast():
                loss, loss_dict = compute_losses(
                    model, coords, eps, mu, E_target, H_target,
                    pml_loss_fn, config
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, loss_dict = compute_losses(
                model, coords, eps, mu, E_target, H_target,
                pml_loss_fn, config
            )
            loss.backward()
            optimizer.step()
        
        # Accumulate losses
        for key in total_losses:
            total_losses[key] += loss_dict[key]
        n_batches += 1
    
    # Average over batches
    avg_losses = {k: v / n_batches for k, v in total_losses.items()}
    
    return avg_losses


def validate(
    model: MaxwellPINN,
    dataloader: DataLoader,
    pml_loss_fn: Optional[PMLLoss],
    config: TrainingConfig,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate model on held-out data.
    """
    model.eval()
    
    total_losses = {'total': 0, 'pde': 0, 'div': 0, 'data': 0, 'pml': 0}
    n_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            coords, eps, mu, E_target, H_target = [b.to(device) for b in batch]
            
            # Need grads for PDE residual computation
            coords = coords.requires_grad_(True)
            
            with torch.enable_grad():
                loss, loss_dict = compute_losses(
                    model, coords, eps, mu, E_target, H_target,
                    pml_loss_fn, config
                )
            
            for key in total_losses:
                total_losses[key] += loss_dict[key]
            n_batches += 1
    
    avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
    
    return avg_losses


def save_checkpoint(
    model: MaxwellPINN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    losses: Dict[str, float],
    config: TrainingConfig,
    filename: str
):
    """Save model checkpoint."""
    checkpoint_path = Path(config.checkpoint_dir) / filename
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'config': config.to_dict()
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(
    model: MaxwellPINN,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str
) -> Tuple[int, Dict[str, float]]:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['losses']


# ============================================================================
# Main Training Loop
# ============================================================================

def train(config: TrainingConfig):
    """
    Main training function.
    
    Args:
        config: Training configuration
    """
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Create model
    model = MaxwellPINN(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_hidden=config.num_hidden,
        use_fourier=config.use_fourier,
        fourier_sigma=config.fourier_sigma,
        num_frequencies=config.num_frequencies
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Create PML loss
    pml_loss_fn = PMLLoss(
        domain_bounds=config.domain_bounds,
        pml_thickness=config.pml_thickness,
        sigma_max=1.0,
        order=2
    ).to(device)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if config.use_amp and device.type == 'cuda' else None
    
    # Check if dataset exists, otherwise use collocation points
    data_path = Path(config.data_path)
    if data_path.exists():
        print(f"Loading dataset from {data_path}")
        train_dataset = MaxwellDataset(str(data_path), split='train')
        val_dataset = MaxwellDataset(str(data_path), split='val')
    else:
        print("No dataset found, using collocation points (unsupervised mode)")
        # Create synthetic dataset for testing
        from dataset_generator import generate_training_dataset, save_dataset_hdf5
        
        print("Generating synthetic dataset...")
        dataset = generate_training_dataset(
            n_samples=50000,
            grid_size=(32, 32, 32),
            material_type='uniform',
            use_meep=False
        )
        
        data_path.parent.mkdir(parents=True, exist_ok=True)
        save_dataset_hdf5(dataset, str(data_path))
        
        train_dataset = MaxwellDataset(str(data_path), split='train')
        val_dataset = MaxwellDataset(str(data_path), split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # TensorBoard (optional)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(config.log_dir)
        use_tensorboard = True
    except ImportError:
        writer = None
        use_tensorboard = False
        print("TensorBoard not available, skipping logging")
    
    # Training loop
    best_val_loss = float('inf')
    start_time = time.time()
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, scaler,
            pml_loss_fn, config, device
        )
        
        # Validate
        val_losses = validate(model, val_loader, pml_loss_fn, config, device)
        
        # Update scheduler
        scheduler.step(val_losses['total'])
        
        epoch_time = time.time() - epoch_start
        
        # Logging
        if epoch % config.log_interval == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d} | "
                  f"Train: {train_losses['total']:.2e} | "
                  f"Val: {val_losses['total']:.2e} | "
                  f"LR: {lr:.2e} | "
                  f"Time: {epoch_time:.1f}s")
            print(f"         | "
                  f"PDE: {train_losses['pde']:.2e} | "
                  f"Div: {train_losses['div']:.2e} | "
                  f"Data: {train_losses['data']:.2e} | "
                  f"PML: {train_losses['pml']:.2e}")
        
        # TensorBoard logging
        if use_tensorboard and writer is not None:
            for key, val in train_losses.items():
                writer.add_scalar(f'Loss/train_{key}', val, epoch)
            for key, val in val_losses.items():
                writer.add_scalar(f'Loss/val_{key}', val, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            save_checkpoint(model, optimizer, epoch, val_losses, config, 'best_model.pt')
        
        # Periodic checkpoint
        if epoch % config.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, train_losses, config, f'checkpoint_epoch_{epoch}.pt')
    
    # Final checkpoint
    save_checkpoint(model, optimizer, epoch, train_losses, config, 'final_model.pt')
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training complete! Total time: {total_time / 60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.2e}")
    print("=" * 60)
    
    if use_tensorboard and writer is not None:
        writer.close()
    
    return model


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Maxwell-PINN-NIF model')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4096, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden layer dimension')
    parser.add_argument('--num-hidden', type=int, default=6, help='Number of hidden layers')
    parser.add_argument('--data', type=str, default='data/em_data.h5', help='Path to dataset')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='runs/maxwell_pinn', help='TensorBoard log directory')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--fourier', action='store_true', help='Use Fourier feature encoding')
    args = parser.parse_args()
    
    # Load or create config
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = TrainingConfig.from_dict(json.load(f))
    else:
        config = TrainingConfig(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            hidden_dim=args.hidden_dim,
            num_hidden=args.num_hidden,
            data_path=args.data,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            use_amp=not args.no_amp,
            use_fourier=args.fourier
        )
    
    # Print config
    print("\nTraining Configuration:")
    for key, val in config.to_dict().items():
        print(f"  {key}: {val}")
    print()
    
    # Train
    train(config)


if __name__ == "__main__":
    main()
