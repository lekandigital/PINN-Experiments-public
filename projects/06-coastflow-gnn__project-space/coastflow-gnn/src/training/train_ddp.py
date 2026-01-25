"""
Multi-GPU DDP Training Script for CoastFlow-GNN

Supports distributed data parallel training across multiple GPUs
using PyTorch's DistributedDataParallel (DDP).

Usage:
    # Single node, multi-GPU
    torchrun --nproc_per_node=4 train_ddp.py --epochs 50
    
    # Multi-node
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr=<addr> train_ddp.py
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.coastflow_gnn import CoastFlowGNN
from src.models.physics_losses import physics_informed_loss
from src.data.dataset import SyntheticCoastalDataset


def setup_ddp(rank: int, world_size: int):
    """Initialize DDP process group."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up DDP process group."""
    dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


class DDPTrainer:
    """
    Distributed Data Parallel Trainer for CoastFlow-GNN.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset,
        rank: int,
        world_size: int,
        batch_size: int = 4,
        learning_rate: float = 1e-3,
        output_dir: str = "./outputs",
        use_amp: bool = True,
        grad_clip_norm: float = 1.0,
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')
        self.is_main = rank == 0
        
        # Move model to device and wrap with DDP
        model = model.to(self.device)
        self.model = DDP(model, device_ids=[rank])
        
        # Create distributed samplers
        self.train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        self.val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            sampler=self.train_sampler, num_workers=2
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            sampler=self.val_sampler, num_workers=2
        )
        
        # Optimizer (scale LR by world size for large batch training)
        scaled_lr = learning_rate * world_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=scaled_lr)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Mixed precision
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None
        self.grad_clip_norm = grad_clip_norm
        
        # Output
        self.output_dir = Path(output_dir)
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # History (only track on main process)
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        self.train_sampler.set_epoch(epoch)  # Important for shuffling
        
        total_loss = 0
        num_batches = 0
        
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    predictions = self.model(data.x, data.edge_index, data.batch)
                    losses = physics_informed_loss(data, predictions)
                    loss = losses['total']
                
                self.scaler.scale(loss).backward()
                
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(data.x, data.edge_index, data.batch)
                losses = physics_informed_loss(data, predictions)
                loss = losses['total']
                
                loss.backward()
                
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Aggregate loss across all processes
        avg_loss = torch.tensor(total_loss / num_batches, device=self.device)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss.item() / self.world_size
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Run validation."""
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        for data in self.val_loader:
            data = data.to(self.device)
            
            if self.use_amp:
                with autocast():
                    predictions = self.model(data.x, data.edge_index, data.batch)
                    losses = physics_informed_loss(data, predictions)
            else:
                predictions = self.model(data.x, data.edge_index, data.batch)
                losses = physics_informed_loss(data, predictions)
            
            total_loss += losses['total'].item()
            num_batches += 1
        
        # Aggregate loss
        avg_loss = torch.tensor(total_loss / max(num_batches, 1), device=self.device)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss.item() / self.world_size
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save checkpoint (only on main process)."""
        if not self.is_main:
            return
        
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),  # Save unwrapped model
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }
        
        torch.save(checkpoint, self.output_dir / 'latest_checkpoint.pth')
        
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best_model.pth')
    
    def train(self, num_epochs: int):
        """Full training loop."""
        if self.is_main:
            print(f"\nDDP Training on {self.world_size} GPUs")
            print(f"Epochs: {num_epochs}")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Update history
            if self.is_main:
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
                
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                      f"Time: {time.time()-epoch_start:.1f}s")
            
            # Checkpoint
            self.save_checkpoint(epoch, val_loss)
            
            # Synchronize
            dist.barrier()
        
        if self.is_main:
            print(f"\nTraining complete! Best val loss: {self.best_val_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description="DDP Training for CoastFlow-GNN")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output-dir', type=str, default='./outputs_ddp')
    args = parser.parse_args()
    
    # Get DDP info from environment (set by torchrun)
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Initialize DDP
    setup_ddp(rank, world_size)
    
    try:
        # Create dataset (same on all processes)
        dataset = SyntheticCoastalDataset(
            num_samples=140,
            num_nodes=200,
            seed=42,
        )
        
        # Split dataset
        n = len(dataset)
        n_train = int(n * 0.714)
        n_val = int(n * 0.143)
        
        torch.manual_seed(42)
        indices = torch.randperm(n)
        train_dataset = dataset[indices[:n_train]]
        val_dataset = dataset[indices[n_train:n_train + n_val]]
        
        # Create model
        model = CoastFlowGNN(
            in_channels=6,
            hidden_channels=args.hidden,
            out_channels=4,
        )
        
        # Create trainer
        trainer = DDPTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            rank=rank,
            world_size=world_size,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            output_dir=args.output_dir,
        )
        
        # Train
        trainer.train(args.epochs)
        
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
