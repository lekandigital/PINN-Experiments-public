"""
Trainer: Training Loop for HGNN-NIF-Cloth

Implements:
- Curriculum learning with 3 stages
- Mixed precision training (torch.cuda.amp)
- Gradient clipping
- Learning rate scheduling (warmup + cosine)
- Checkpoint saving
- TensorBoard logging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import time
from tqdm import tqdm
import json

from .losses import PhysicsLoss, CurriculumWeightScheduler


class Trainer:
    """
    Training manager for HGNN-NIF-Cloth model.
    
    Features:
    - Mixed precision training for memory efficiency
    - Gradient accumulation for large effective batch sizes
    - Curriculum learning with staged difficulty
    - Comprehensive logging and checkpointing
    
    Args:
        model: HGNN_NIF_ClothModel instance
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        loss_fn: PhysicsLoss instance
        optimizer: PyTorch optimizer
        device: Device to train on
        output_dir: Directory for checkpoints and logs
        use_amp: Whether to use automatic mixed precision
        gradient_clip: Max gradient norm (None to disable)
        log_interval: Steps between logging
        save_interval: Epochs between checkpoint saves
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[PhysicsLoss] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cuda',
        output_dir: str = 'outputs',
        use_amp: bool = True,
        gradient_clip: float = 1.0,
        log_interval: int = 10,
        save_interval: int = 10
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.gradient_clip = gradient_clip
        self.log_interval = log_interval
        self.save_interval = save_interval
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Loss function
        self.loss_fn = loss_fn or PhysicsLoss()
        
        # Optimizer
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Learning rate scheduler
        self.scheduler = None
        
        # Logging
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Curriculum scheduler
        self.curriculum = None
        
        # Rest lengths (computed from first batch)
        self.rest_lengths = None
        
    def setup_scheduler(
        self,
        num_epochs: int,
        warmup_epochs: int = 5,
        min_lr: float = 1e-6
    ):
        """
        Setup learning rate scheduler with warmup.
        
        Uses linear warmup followed by cosine annealing.
        """
        warmup_steps = warmup_epochs * len(self.train_loader)
        total_steps = num_epochs * len(self.train_loader)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
                
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        
    def setup_curriculum(self, num_epochs: int):
        """Setup curriculum learning scheduler."""
        self.curriculum = CurriculumWeightScheduler(self.loss_fn, num_epochs)
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Run one training epoch.
        
        Returns:
            Dictionary of average metrics for the epoch
        """
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'spring': 0.0,
            'sdf': 0.0,
        }
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            fine_pos = batch['fine_pos'].to(self.device)
            fine_edges = batch['fine_edges'].to(self.device)
            coarse_pos = batch['coarse_pos'].to(self.device)
            coarse_edges = batch['coarse_edges'].to(self.device)
            query_points = batch['query_points'].to(self.device)
            query_sdf = batch['query_sdf'].to(self.device)
            
            # Compute rest lengths from first batch
            if self.rest_lengths is None:
                self.rest_lengths = self._compute_rest_lengths(fine_pos[0], fine_edges)
                
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                output = self.model(
                    (fine_pos, fine_edges),
                    (coarse_pos, coarse_edges),
                    query_points
                )
                
                # Compute loss
                loss, loss_dict = self.loss_fn(
                    pred_pos=fine_pos,  # For now, use input as "predicted" (model doesn't predict positions yet)
                    pred_sdf=output['sdf'],
                    gt_sdf=query_sdf,
                    edge_index=fine_edges,
                    rest_lengths=self.rest_lengths
                )
                
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.gradient_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
                
            # Update metrics
            epoch_losses['total'] += loss.item()
            for key in ['spring', 'sdf']:
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key].item()
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # TensorBoard logging
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
            
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """
        Run validation loop.
        
        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}
            
        self.model.eval()
        
        val_losses = {
            'total': 0.0,
            'spring': 0.0,
            'sdf': 0.0,
        }
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                fine_pos = batch['fine_pos'].to(self.device)
                fine_edges = batch['fine_edges'].to(self.device)
                coarse_pos = batch['coarse_pos'].to(self.device)
                coarse_edges = batch['coarse_edges'].to(self.device)
                query_points = batch['query_points'].to(self.device)
                query_sdf = batch['query_sdf'].to(self.device)
                
                with autocast(enabled=self.use_amp):
                    output = self.model(
                        (fine_pos, fine_edges),
                        (coarse_pos, coarse_edges),
                        query_points
                    )
                    
                    loss, loss_dict = self.loss_fn(
                        pred_pos=fine_pos,
                        pred_sdf=output['sdf'],
                        gt_sdf=query_sdf,
                        edge_index=fine_edges,
                        rest_lengths=self.rest_lengths
                    )
                    
                val_losses['total'] += loss.item()
                for key in ['spring', 'sdf']:
                    if key in loss_dict:
                        val_losses[key] += loss_dict[key].item()
                num_batches += 1
                
        # Average
        for key in val_losses:
            val_losses[key] /= max(num_batches, 1)
            
        return val_losses
    
    def train(
        self,
        num_epochs: int,
        warmup_epochs: int = 5,
        use_curriculum: bool = True
    ) -> Dict[str, Any]:
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            warmup_epochs: Number of warmup epochs
            use_curriculum: Whether to use curriculum learning
            
        Returns:
            Training history dictionary
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"  Device: {self.device}")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"  Output dir: {self.output_dir}")
        
        # Setup schedulers
        self.setup_scheduler(num_epochs, warmup_epochs)
        if use_curriculum:
            self.setup_curriculum(num_epochs)
            
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
        }
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Update curriculum
            if self.curriculum is not None:
                weights = self.curriculum.step(epoch)
                print(f"\n[Epoch {epoch}] Curriculum stage: {weights['stage']}")
                
            # Training
            train_metrics = self.train_epoch()
            history['train_loss'].append(train_metrics['total'])
            history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Validation
            val_metrics = self.validate()
            if val_metrics:
                history['val_loss'].append(val_metrics['total'])
                
            # Logging
            self.writer.add_scalar('epoch/train_loss', train_metrics['total'], epoch)
            if val_metrics:
                self.writer.add_scalar('epoch/val_loss', val_metrics['total'], epoch)
                
            print(f"Epoch {epoch}: train_loss={train_metrics['total']:.4f}", end='')
            if val_metrics:
                print(f", val_loss={val_metrics['total']:.4f}", end='')
            print()
            
            # Checkpointing
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch{epoch+1}.pth")
                
            # Save best model
            if val_metrics and val_metrics['total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total']
                self.save_checkpoint("best_model.pth")
                print(f"  → New best model saved!")
                
        # Final save
        self.save_checkpoint("final_model.pth")
        
        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed/60:.1f} minutes")
        
        # Save history
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
            
        self.writer.close()
        
        return history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'rest_lengths': self.rest_lengths,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        torch.save(checkpoint, self.output_dir / filename)
        
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.rest_lengths = checkpoint.get('rest_lengths')
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        print(f"Loaded checkpoint from epoch {self.epoch}")
        
    def _compute_rest_lengths(
        self,
        positions: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Compute rest lengths from initial positions."""
        src, tgt = edge_index[0], edge_index[1]
        edge_vecs = positions[src] - positions[tgt]
        return torch.norm(edge_vecs, dim=-1)


def create_trainer(
    model: nn.Module,
    data_path: str,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    output_dir: str = 'outputs',
    **kwargs
) -> Trainer:
    """
    Convenience function to create a Trainer with standard settings.
    
    Args:
        model: Model to train
        data_path: Path to HDF5 data file
        batch_size: Training batch size
        learning_rate: Initial learning rate
        output_dir: Output directory
        **kwargs: Additional Trainer arguments
        
    Returns:
        Configured Trainer instance
    """
    from ..data.dataset import create_dataloader
    
    train_loader = create_dataloader(data_path, 'train', batch_size)
    val_loader = create_dataloader(data_path, 'val', batch_size)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = PhysicsLoss()
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        output_dir=output_dir,
        **kwargs
    )
    
    return trainer
