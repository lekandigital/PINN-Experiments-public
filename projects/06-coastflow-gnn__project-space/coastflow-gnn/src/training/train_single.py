"""
Single-GPU Training Script for CoastFlow-GNN

Features:
- Mixed precision training (torch.cuda.amp)
- Learning rate scheduling (ReduceLROnPlateau)
- Gradient clipping to prevent exploding gradients
- Checkpoint saving (best model + periodic)
- Training curve logging
- Physics-informed loss combination
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.coastflow_gnn import CoastFlowGNN
from src.models.physics_losses import physics_informed_loss, PhysicsInformedLoss
from src.data.dataset import SyntheticCoastalDataset, create_data_loaders


class Trainer:
    """
    Trainer class for CoastFlow-GNN.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = None,
        output_dir: str = "./outputs",
        use_amp: bool = True,
        grad_clip_norm: float = 1.0,
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.grad_clip_norm = grad_clip_norm
        
        # Loss function
        self.loss_weights = loss_weights or {
            'lambda_data': 1.0,
            'lambda_cont': 1.0,
            'lambda_mom': 0.1,
            'lambda_turb': 0.01,
            'lambda_bc': 1.0,
        }
        
        # Mixed precision
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
            'physics_residuals': {
                'continuity': [],
                'momentum': [],
                'turbulence': [],
            },
            'epoch_time': [],
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        loss_components = {'data': 0, 'continuity': 0, 'momentum': 0, 'turbulence': 0, 'boundary': 0}
        num_batches = 0
        
        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    predictions = self.model(data.x, data.edge_index, data.batch)
                    losses = physics_informed_loss(data, predictions, **self.loss_weights)
                    loss = losses['total']
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(data.x, data.edge_index, data.batch)
                losses = physics_informed_loss(data, predictions, **self.loss_weights)
                loss = losses['total']
                
                loss.backward()
                
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                
                self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            for key in loss_components:
                if key in losses:
                    loss_components[key] += losses[key].item()
            num_batches += 1
        
        # Average losses
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        return {'total': avg_loss, **loss_components}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        
        total_loss = 0
        loss_components = {'data': 0, 'continuity': 0, 'momentum': 0, 'turbulence': 0, 'boundary': 0}
        num_batches = 0
        
        for data in self.val_loader:
            data = data.to(self.device)
            
            if self.use_amp:
                with autocast():
                    predictions = self.model(data.x, data.edge_index, data.batch)
                    losses = physics_informed_loss(data, predictions, **self.loss_weights)
            else:
                predictions = self.model(data.x, data.edge_index, data.batch)
                losses = physics_informed_loss(data, predictions, **self.loss_weights)
            
            total_loss += losses['total'].item()
            for key in loss_components:
                if key in losses:
                    loss_components[key] += losses[key].item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        return {'total': avg_loss, **loss_components}
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'loss_weights': self.loss_weights,
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  → Saved best model (val_loss={self.best_val_loss:.6f})")
        
        # Periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            periodic_path = self.output_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, periodic_path)
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def train(self, num_epochs: int, verbose: bool = True):
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            verbose: Print progress
        """
        print(f"\n{'='*60}")
        print(f"Training CoastFlow-GNN")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_losses = self.train_epoch(epoch)
            
            # Validate
            val_losses = self.validate()
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total'])
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_losses['total'])
            self.history['lr'].append(current_lr)
            self.history['physics_residuals']['continuity'].append(val_losses['continuity'])
            self.history['physics_residuals']['momentum'].append(val_losses['momentum'])
            self.history['physics_residuals']['turbulence'].append(val_losses['turbulence'])
            self.history['epoch_time'].append(time.time() - epoch_start)
            
            # Check for best model
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
                self.best_epoch = epoch
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Print progress
            if verbose:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train: {train_losses['total']:.6f} | "
                      f"Val: {val_losses['total']:.6f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Time: {self.history['epoch_time'][-1]:.1f}s"
                      f"{' *' if is_best else ''}")
        
        # Save final history
        self.save_history()
        
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best val loss: {self.best_val_loss:.6f} at epoch {self.best_epoch+1}")
        print(f"{'='*60}\n")


def train_model(
    num_samples: int = 140,
    num_nodes: int = 200,
    hidden_channels: int = 64,
    num_epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    output_dir: str = "./outputs",
    seed: int = 42,
    use_amp: bool = True,
) -> Trainer:
    """
    Train CoastFlow-GNN model.
    
    Convenience function for training with default settings.
    
    Returns:
        Trained Trainer object
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset
    print("Creating synthetic dataset...")
    dataset = SyntheticCoastalDataset(
        num_samples=num_samples,
        num_nodes=num_nodes,
        seed=seed,
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset, batch_size=batch_size
    )
    
    print(f"Dataset: {len(dataset)} samples")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
    
    # Create model
    model = CoastFlowGNN(
        in_channels=6,
        hidden_channels=hidden_channels,
        out_channels=4,
    )
    print(f"\nModel: {model.count_parameters():,} parameters")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        use_amp=use_amp,
    )
    
    # Train
    trainer.train(num_epochs=num_epochs)
    
    return trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CoastFlow-GNN")
    
    # Data
    parser.add_argument('--num-samples', type=int, default=140,
                        help='Number of training samples')
    parser.add_argument('--num-nodes', type=int, default=200,
                        help='Number of nodes per mesh')
    
    # Model
    parser.add_argument('--hidden', type=int, default=64,
                        help='Hidden channels in GNN')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision training')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    trainer = train_model(
        num_samples=args.num_samples,
        num_nodes=args.num_nodes,
        hidden_channels=args.hidden,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        seed=args.seed,
        use_amp=not args.no_amp,
    )
