"""
PEGNN-Deform: Training Script

Training loop with:
- Position MSE loss
- Velocity MSE loss  
- Energy conservation loss
- Mixed-precision (AMP) training with GradScaler
- Checkpoint scheduling optimized for 48GB GPU
- wandb integration for experiment tracking

Author: PEGNN-Deform Team
"""

import os
import sys
import argparse
import random
import math
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import torch.cuda.amp
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from pegdeform_model import PEGNNDeform, count_parameters


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MeshDataset(Dataset):
    """
    Dataset for deformable mesh sequences.
    
    Each sample contains:
    - pos: Current vertex positions [N, 3]
    - vel: Current velocities [N, 3]
    - pos_next: Ground truth next positions [N, 3]
    - vel_next: Ground truth next velocities [N, 3]
    - edge_index: Mesh edges [2, E]
    - edge_attr: Spring parameters [E, 2] - (stiffness, rest_length)
    """
    
    def __init__(self, data_dir: str, split: str = 'train'):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Find all data files
        self.files = sorted(self.data_dir.glob(f"{split}_*.pt"))
        
        if len(self.files) == 0:
            print(f"Warning: No data files found in {data_dir} for split '{split}'")
            print("Generating synthetic data...")
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self, num_samples: int = 100):
        """Generate synthetic training data if none exists."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a simple deformable mesh (grid)
        grid_size = 20
        N = grid_size * grid_size
        
        # Create grid positions
        x = torch.linspace(0, 1, grid_size)
        y = torch.linspace(0, 1, grid_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        pos_rest = torch.stack([xx.flatten(), yy.flatten(), torch.zeros(N)], dim=1)
        
        # Build edges (grid connectivity)
        edge_list = []
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                # Right neighbor
                if j < grid_size - 1:
                    edge_list.append([idx, idx + 1])
                    edge_list.append([idx + 1, idx])
                # Bottom neighbor
                if i < grid_size - 1:
                    edge_list.append([idx, idx + grid_size])
                    edge_list.append([idx + grid_size, idx])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Spring parameters
        stiffness = 10.0
        rest_lengths = torch.norm(
            pos_rest[edge_index[0]] - pos_rest[edge_index[1]], dim=1
        )
        edge_attr = torch.stack([
            torch.full_like(rest_lengths, stiffness),
            rest_lengths
        ], dim=1)
        
        # Generate trajectory using simple spring physics
        dt = 0.01
        pos = pos_rest.clone()
        vel = torch.zeros_like(pos)
        
        # Add initial perturbation
        pos[:, 2] += 0.1 * torch.sin(pos[:, 0] * math.pi) * torch.sin(pos[:, 1] * math.pi)
        
        for sample_idx in range(num_samples):
            # Save current state
            pos_current = pos.clone()
            vel_current = vel.clone()
            
            # Compute forces (spring + gravity)
            forces = torch.zeros_like(pos)
            src, dst = edge_index
            diff = pos[src] - pos[dst]
            dist = torch.norm(diff, dim=1, keepdim=True)
            direction = diff / (dist + 1e-8)
            
            k = edge_attr[:, 0:1]
            L0 = edge_attr[:, 1:2]
            spring_force = -k * (dist - L0) * direction
            
            # Scatter add forces
            forces.scatter_add_(0, dst.unsqueeze(1).expand(-1, 3), spring_force)
            
            # Add gravity
            forces[:, 2] -= 0.5
            
            # Add damping
            forces -= 0.1 * vel
            
            # Update velocity and position
            vel_next = vel + dt * forces
            pos_next = pos + dt * vel_next
            
            # Fix top edge (boundary condition)
            top_row = torch.arange(grid_size) * grid_size + (grid_size - 1)
            pos_next[top_row] = pos_rest[top_row]
            vel_next[top_row] = 0
            
            # Save sample
            data = Data(
                pos=pos_current.float(),
                vel=vel_current.float(),
                pos_next=pos_next.float(),
                vel_next=vel_next.float(),
                edge_index=edge_index,
                edge_attr=edge_attr.float()
            )
            torch.save(data, self.data_dir / f"{self.split}_{sample_idx:04d}.pt")
            
            # Update state for next sample
            pos = pos_next.clone()
            vel = vel_next.clone()
        
        self.files = sorted(self.data_dir.glob(f"{self.split}_*.pt"))
        print(f"Generated {len(self.files)} synthetic samples")
    
    def len(self) -> int:
        return len(self.files)
    
    def get(self, idx: int) -> Data:
        return torch.load(self.files[idx])


def compute_energy(
    pos: torch.Tensor,
    vel: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    mass: float = 1.0
) -> torch.Tensor:
    """
    Compute total mechanical energy (kinetic + spring potential).
    
    E = 0.5 * m * v² + 0.5 * k * (d - L0)²
    """
    # Kinetic energy
    kinetic = 0.5 * mass * (vel ** 2).sum()
    
    # Spring potential energy
    src, dst = edge_index
    diff = pos[src] - pos[dst]
    dist = torch.norm(diff, dim=1)
    
    k = edge_attr[:, 0]
    L0 = edge_attr[:, 1]
    
    # Divide by 2 to avoid double counting (each edge appears twice)
    spring_potential = 0.25 * (k * (dist - L0) ** 2).sum()
    
    return kinetic + spring_potential


def train_step(
    model: nn.Module,
    batch: Data,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    lambda_vel: float = 0.1,
    lambda_energy: float = 1.0
) -> Dict[str, float]:
    """
    Single training step with mixed precision.
    
    Returns dictionary of loss components for logging.
    """
    model.train()
    
    # Move data to device
    pos = batch.pos.to(device)
    vel = batch.vel.to(device)
    pos_gt = batch.pos_next.to(device)
    vel_gt = batch.vel_next.to(device)
    edge_index = batch.edge_index.to(device)
    edge_attr = batch.edge_attr.to(device)
    
    optimizer.zero_grad()
    
    # Forward pass with mixed precision
    with torch.cuda.amp.autocast():
        pos_pred, vel_pred, _ = model(pos, vel, edge_index, edge_attr)
        
        # Position loss (MSE)
        loss_pos = F.mse_loss(pos_pred, pos_gt)
        
        # Velocity loss (MSE)
        loss_vel = F.mse_loss(vel_pred, vel_gt)
        
        # Energy conservation loss
        energy_pred = compute_energy(pos_pred, vel_pred, edge_index, edge_attr)
        energy_gt = compute_energy(pos_gt, vel_gt, edge_index, edge_attr)
        loss_energy = (energy_pred - energy_gt).pow(2) / (energy_gt.pow(2) + 1e-8)
        
        # Total loss
        loss = loss_pos + lambda_vel * loss_vel + lambda_energy * loss_energy
    
    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    return {
        'loss': loss.item(),
        'loss_pos': loss_pos.item(),
        'loss_vel': loss_vel.item(),
        'loss_energy': loss_energy.item()
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    lambda_vel: float = 0.1,
    lambda_energy: float = 1.0
) -> Dict[str, float]:
    """Validate model on held-out data."""
    model.eval()
    
    total_loss = 0.0
    total_pos_loss = 0.0
    total_vel_loss = 0.0
    total_energy_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        pos = batch.pos.to(device)
        vel = batch.vel.to(device)
        pos_gt = batch.pos_next.to(device)
        vel_gt = batch.vel_next.to(device)
        edge_index = batch.edge_index.to(device)
        edge_attr = batch.edge_attr.to(device)
        
        with torch.cuda.amp.autocast():
            pos_pred, vel_pred, _ = model(pos, vel, edge_index, edge_attr)
            
            loss_pos = F.mse_loss(pos_pred, pos_gt)
            loss_vel = F.mse_loss(vel_pred, vel_gt)
            
            energy_pred = compute_energy(pos_pred, vel_pred, edge_index, edge_attr)
            energy_gt = compute_energy(pos_gt, vel_gt, edge_index, edge_attr)
            loss_energy = (energy_pred - energy_gt).pow(2) / (energy_gt.pow(2) + 1e-8)
            
            loss = loss_pos + lambda_vel * loss_vel + lambda_energy * loss_energy
        
        total_loss += loss.item()
        total_pos_loss += loss_pos.item()
        total_vel_loss += loss_vel.item()
        total_energy_loss += loss_energy.item()
        num_batches += 1
    
    return {
        'val_loss': total_loss / num_batches,
        'val_loss_pos': total_pos_loss / num_batches,
        'val_loss_vel': total_vel_loss / num_batches,
        'val_loss_energy': total_energy_loss / num_batches
    }


def train(args):
    """Main training function."""
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create data directories
    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = MeshDataset(data_dir / 'synthetic', split='train')
    val_dataset = MeshDataset(data_dir / 'synthetic', split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    model = PEGNNDeform(
        hidden_size=args.hidden_size,
        num_mp_layers=args.num_mp_layers,
        dt=args.dt
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.01
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Wandb logging (optional)
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project="pegnn-deform",
                config={
                    **vars(args),
                    'model_params': count_parameters(model),
                    'device': str(device),
                    'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU',
                    'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if device.type == 'cuda' else 0,
                },
                name=f"pegnn_{args.hidden_size}h_{args.num_mp_layers}mp_seed{args.seed}",
                tags=['training', f'hidden_{args.hidden_size}', f'mp_{args.num_mp_layers}']
            )
            # Log model architecture
            wandb.watch(model, log='all', log_freq=100)
            # Log model summary
            wandb.config.update({
                'model_architecture': str(model),
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset)
            })
        except ImportError:
            print("wandb not installed, skipping logging")
            args.use_wandb = False
    
    # Training loop
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        epoch_losses = {'loss': 0, 'loss_pos': 0, 'loss_vel': 0, 'loss_energy': 0}
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            losses = train_step(
                model, batch, optimizer, scaler, device,
                lambda_vel=args.lambda_vel,
                lambda_energy=args.lambda_energy
            )
            
            for k, v in losses.items():
                epoch_losses[k] += v
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{losses['loss']:.4f}",
                'pos': f"{losses['loss_pos']:.4f}"
            })
        
        # Average epoch losses
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
        
        # Validation
        val_metrics = validate(
            model, val_loader, device,
            lambda_vel=args.lambda_vel,
            lambda_energy=args.lambda_energy
        )
        
        # Learning rate step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Logging
        print(f"Epoch {epoch+1}: train_loss={epoch_losses['loss']:.4f}, "
              f"val_loss={val_metrics['val_loss']:.4f}, lr={current_lr:.2e}")
        
        if args.use_wandb:
            log_dict = {
                'epoch': epoch + 1,
                **{f'train_{k}': v for k, v in epoch_losses.items()},
                **val_metrics,
                'learning_rate': current_lr
            }
            # Add GPU metrics if available
            if device.type == 'cuda':
                log_dict['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
                log_dict['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
                log_dict['gpu_memory_peak_gb'] = torch.cuda.max_memory_allocated() / 1e9
            wandb.log(log_dict)
        
        # Checkpointing
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state': scaler.state_dict(),
                'val_loss': best_val_loss,
                'args': vars(args)
            }, checkpoint_dir / 'best_model.pth')
            print(f"  → Saved best model (val_loss={best_val_loss:.4f})")
        
        # Regular checkpoint every N epochs
        if (epoch + 1) % args.checkpoint_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state': scaler.state_dict(),
                'val_loss': val_metrics['val_loss'],
                'args': vars(args)
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')
        
        # Clear GPU cache periodically
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    
    if args.use_wandb:
        wandb.finish()
    
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Train PEGNN-Deform model')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    
    # Model
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Hidden dimension for GRU')
    parser.add_argument('--num-mp-layers', type=int, default=3,
                        help='Number of message passing layers')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Time step for integration')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    
    # Loss weights
    parser.add_argument('--lambda-vel', type=float, default=0.1,
                        help='Velocity loss weight')
    parser.add_argument('--lambda-energy', type=float, default=1.0,
                        help='Energy conservation loss weight')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Checkpointing
    parser.add_argument('--checkpoint-freq', type=int, default=10,
                        help='Checkpoint frequency (epochs)')
    
    # Logging
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    
    args = parser.parse_args()
    
    train(args)


if __name__ == '__main__':
    main()
