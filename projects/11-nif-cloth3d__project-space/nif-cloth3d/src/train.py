"""
NIF-Cloth3D-Interactive: Training Script with Curriculum Learning

This script trains the SIREN-based cloth simulation model with:
- Physics-informed losses (stretch, bend, momentum)
- Curriculum learning (force magnitude ramping)
- Mixed-precision training (FP16)
- Gradient accumulation for low-VRAM setups
- Checkpointing and logging

Usage:
    python train.py --config configs/l40s.yaml
    python train.py --config configs/rtx4090.yaml --data data/test_cloth.pt
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from model import SineMLP, create_model
from losses import PhysicsLoss, BendLossVectorized


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class ClothDataset(Dataset):
    """
    Dataset for cloth simulation training.
    
    Each sample contains:
    - Vertex positions (N, 3)
    - Time step (scalar)
    - External forces (N, 3) or (3,) broadcast
    - Material ID (scalar)
    - Target displacement (N, 3) - optional for supervised training
    """
    
    def __init__(
        self,
        vertices: torch.Tensor,
        edges: torch.Tensor,
        n_samples: int = 1000,
        force_scale: float = 1.0,
        device: str = 'cpu'
    ):
        self.vertices = vertices.to(device)
        self.edges = edges.to(device)
        self.n_samples = n_samples
        self.force_scale = force_scale
        self.device = device
        self.n_vertices = vertices.shape[0]
        
        # Precompute rest lengths
        i_idx = edges[:, 0]
        j_idx = edges[:, 1]
        edge_vectors = vertices[i_idx] - vertices[j_idx]
        self.rest_lengths = torch.norm(edge_vectors, dim=1).to(device)
        
        # Build Laplacian for bend loss
        bend_loss_fn = BendLossVectorized()
        self.laplacian = bend_loss_fn.build_laplacian(
            self.n_vertices, edges, torch.device(device)
        )
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Random time in [0, 1]
        t = torch.rand(1, device=self.device)
        
        # Random external force scaled by curriculum
        force = torch.randn(3, device=self.device) * self.force_scale
        
        # Random material ID (0-4)
        material_id = torch.randint(0, 5, (1,), device=self.device).float()
        
        return {
            'vertices': self.vertices,
            'time': t,
            'force': force,
            'material_id': material_id,
            'edges': self.edges,
            'rest_lengths': self.rest_lengths,
            'laplacian': self.laplacian
        }
    
    def set_force_scale(self, scale: float):
        """Update force scale for curriculum learning."""
        self.force_scale = scale


def generate_cloth_mesh(
    resolution: int = 50,
    size: float = 2.0,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a simple plane mesh for testing/training.
    
    Args:
        resolution: Grid resolution (resolution x resolution vertices)
        size: Physical size of cloth
        device: Target device
    
    Returns:
        vertices: (N, 3) vertex positions
        edges: (E, 2) edge indices
    """
    import numpy as np
    
    # Create grid of vertices
    x = np.linspace(-size/2, size/2, resolution)
    y = np.linspace(-size/2, size/2, resolution)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)
    
    vertices = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
    vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
    
    # Create edges (horizontal and vertical)
    edges = []
    for i in range(resolution):
        for j in range(resolution - 1):
            idx = i * resolution + j
            edges.append([idx, idx + 1])  # horizontal
    
    for i in range(resolution - 1):
        for j in range(resolution):
            idx = i * resolution + j
            edges.append([idx, idx + resolution])  # vertical
    
    edges = torch.tensor(edges, dtype=torch.long, device=device)
    
    return vertices, edges


# -----------------------------------------------------------------------------
# Training Functions
# -----------------------------------------------------------------------------

def prepare_input(
    vertices: torch.Tensor,
    time: torch.Tensor,
    force: torch.Tensor,
    material_id: torch.Tensor
) -> torch.Tensor:
    """
    Prepare input tensor for the model.
    
    Combines vertex positions, time, force, and material ID into 8D input.
    """
    N = vertices.shape[0]
    
    # Expand scalars to match batch size
    t_expanded = time.expand(N, 1)
    force_expanded = force.unsqueeze(0).expand(N, 3)
    mat_expanded = material_id.expand(N, 1)
    
    # Concatenate: [x, y, z, t, fx, fy, fz, material_id]
    inp = torch.cat([vertices, t_expanded, force_expanded, mat_expanded], dim=1)
    
    return inp


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: PhysicsLoss,
    scaler: GradScaler,
    config: Dict,
    epoch: int,
    device: torch.device
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Returns:
        Dictionary of average loss values
    """
    model.train()
    
    grad_accum = config['training'].get('grad_accumulation', 1)
    use_amp = config['training'].get('mixed_precision', True)
    
    epoch_losses = {'total': 0.0, 'stretch': 0.0, 'bend': 0.0, 'momentum': 0.0}
    n_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        vertices = batch['vertices'].squeeze(0).to(device)
        t = batch['time'].squeeze(0).to(device)
        force = batch['force'].squeeze(0).to(device)
        material_id = batch['material_id'].squeeze(0).to(device)
        edges = batch['edges'].squeeze(0).to(device)
        rest_lengths = batch['rest_lengths'].squeeze(0).to(device)
        laplacian = batch['laplacian'].squeeze(0).to(device)
        
        # Prepare input
        inp = prepare_input(vertices, t, force, material_id)
        
        # Forward pass with mixed precision
        with autocast(enabled=use_amp):
            # Model predicts displacement
            pred_disp = model(inp)
            
            # Predicted positions
            pred_pos = vertices + pred_disp
            
            # External forces per vertex
            N = vertices.shape[0]
            external_forces = force.unsqueeze(0).expand(N, 3)
            
            # Compute physics losses
            losses = loss_fn(
                pred_pos, vertices, edges, laplacian, external_forces, 
                sdf_values=None, rest_lengths=rest_lengths
            )
            
            loss = losses['total'] / grad_accum
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Accumulate losses
        for key in epoch_losses:
            if key in losses:
                epoch_losses[key] += losses[key].item()
        n_batches += 1
    
    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= max(n_batches, 1)
    
    return epoch_losses


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: PhysicsLoss,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate model on held-out data.
    """
    model.eval()
    
    val_losses = {'total': 0.0, 'stretch': 0.0, 'bend': 0.0}
    n_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            vertices = batch['vertices'].squeeze(0).to(device)
            t = batch['time'].squeeze(0).to(device)
            force = batch['force'].squeeze(0).to(device)
            material_id = batch['material_id'].squeeze(0).to(device)
            edges = batch['edges'].squeeze(0).to(device)
            rest_lengths = batch['rest_lengths'].squeeze(0).to(device)
            laplacian = batch['laplacian'].squeeze(0).to(device)
            
            inp = prepare_input(vertices, t, force, material_id)
            pred_disp = model(inp)
            pred_pos = vertices + pred_disp
            
            N = vertices.shape[0]
            external_forces = force.unsqueeze(0).expand(N, 3)
            
            losses = loss_fn(
                pred_pos, vertices, edges, laplacian, external_forces,
                rest_lengths=rest_lengths
            )
            
            for key in val_losses:
                if key in losses:
                    val_losses[key] += losses[key].item()
            n_batches += 1
    
    for key in val_losses:
        val_losses[key] /= max(n_batches, 1)
    
    return val_losses


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    loss: float,
    config: Dict,
    path: str
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'loss': loss,
        'config': config
    }
    torch.save(checkpoint, path)
    
    # Also save as latest
    latest_path = Path(path).parent / 'latest.pt'
    torch.save(checkpoint, latest_path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None
) -> int:
    """Load checkpoint and return starting epoch."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scaler is not None:
        scaler.load_state_dict(checkpoint['scaler'])
    
    return checkpoint['epoch']


# -----------------------------------------------------------------------------
# Curriculum Learning
# -----------------------------------------------------------------------------

def get_force_scale(epoch: int, config: Dict) -> float:
    """
    Compute force scale based on curriculum schedule.
    
    Linear ramp from force_start to force_end over curriculum_epochs.
    """
    curriculum = config.get('curriculum', {})
    force_start = curriculum.get('force_start', 0.0)
    force_end = curriculum.get('force_end', 1.0)
    curriculum_epochs = curriculum.get('epochs', config['training']['epochs'] // 2)
    
    if epoch >= curriculum_epochs:
        return force_end
    
    # Linear interpolation
    progress = epoch / curriculum_epochs
    return force_start + (force_end - force_start) * progress


# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------

def train(config: Dict, args: argparse.Namespace):
    """Main training function."""
    
    # Setup logging
    log_dir = Path(args.output) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'train_{datetime.now():%Y%m%d_%H%M%S}.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create or load dataset
    if args.data and Path(args.data).exists():
        logger.info(f"Loading data from {args.data}")
        data = torch.load(args.data)
        vertices = data['vertices']
        edges = data['edges']
    else:
        logger.info("Generating synthetic cloth mesh")
        resolution = config.get('data', {}).get('resolution', 50)
        vertices, edges = generate_cloth_mesh(resolution=resolution, device=device)
        
        # Save for future use
        data_path = Path(args.output) / 'data' / 'generated_cloth.pt'
        data_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'vertices': vertices, 'edges': edges}, data_path)
        logger.info(f"Saved generated data to {data_path}")
    
    logger.info(f"Cloth mesh: {vertices.shape[0]} vertices, {edges.shape[0]} edges")
    
    # Create dataset
    n_samples = config['training'].get('samples_per_epoch', 1000)
    dataset = ClothDataset(
        vertices=vertices,
        edges=edges,
        n_samples=n_samples,
        force_scale=0.0,  # Start with zero force
        device=str(device)
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Create model
    model_config = config.get('model', {})
    model = create_model(model_config).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    
    # Optimizer
    lr = config['training'].get('learning_rate', 5e-4)
    if isinstance(lr, str):
        lr = float(lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loss function
    physics_config = config.get('physics', {})
    loss_fn = PhysicsLoss(
        stretch_weight=physics_config.get('stretch_weight', 1.0),
        bend_weight=physics_config.get('bend_weight', 0.5),
        momentum_weight=physics_config.get('momentum_weight', 0.3),
        damping_weight=physics_config.get('damping_weight', 0.01)
    )
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=config['training'].get('mixed_precision', True))
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        start_epoch = load_checkpoint(args.resume, model, optimizer, scaler) + 1
    
    # Training loop
    epochs = config['training'].get('epochs', 1000)
    checkpoint_interval = config['training'].get('checkpoint_interval', 50)
    checkpoint_dir = Path(args.output) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    
    logger.info(f"Starting training for {epochs} epochs")
    logger.info(f"Config: {config}")
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        
        # Update curriculum (force magnitude)
        force_scale = get_force_scale(epoch, config)
        dataset.set_force_scale(force_scale)
        
        # Train
        train_losses = train_epoch(
            model, dataloader, optimizer, loss_fn, scaler, config, epoch, device
        )
        
        epoch_time = time.time() - epoch_start
        
        # Logging
        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Loss: {train_losses['total']:.6f} | "
                f"Stretch: {train_losses['stretch']:.6f} | "
                f"Bend: {train_losses['bend']:.6f} | "
                f"Force: {force_scale:.3f} | "
                f"Time: {epoch_time:.2f}s"
            )
        
        # Checkpointing
        if (epoch + 1) % checkpoint_interval == 0 or epoch == epochs - 1:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
            save_checkpoint(model, optimizer, scaler, epoch, train_losses['total'], config, str(checkpoint_path))
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if train_losses['total'] < best_loss:
            best_loss = train_losses['total']
            best_path = checkpoint_dir / 'best.pt'
            save_checkpoint(model, optimizer, scaler, epoch, train_losses['total'], config, str(best_path))
    
    logger.info(f"Training complete. Best loss: {best_loss:.6f}")
    
    # Export TorchScript model for inference
    model.eval()
    traced = torch.jit.trace(model, torch.randn(100, 8, device=device))
    traced_path = checkpoint_dir / 'model_traced.pt'
    traced.save(str(traced_path))
    logger.info(f"Exported TorchScript model: {traced_path}")
    
    return model


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Train NIF-Cloth3D-Interactive')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--data', type=str, default=None, help='Path to cloth data (.pt file)')
    parser.add_argument('--output', type=str, default='checkpoints', help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--wandb-project', type=str, default=None, help='Weights & Biases project name')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Train
    train(config, args)


if __name__ == '__main__':
    main()
