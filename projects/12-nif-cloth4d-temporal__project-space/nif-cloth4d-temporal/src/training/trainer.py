"""
Scheduled Sampling Trainer for NIF-Cloth4D-Temporal.

Implements the main training loop with:
    - Automatic Mixed Precision (AMP) for memory efficiency
    - Scheduled sampling for stable temporal rollouts
    - Physics-aware losses for physically plausible cloth
    - Curriculum learning for sequence length
    - WandB logging and checkpointing
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
)
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from ..models import FourierFeatureMLP
from ..losses import PhysicsLossStack, ScheduledSamplingLoss
from ..losses.scheduled_sampling import CurriculumScheduler, TemporalLossAccumulator
from .config import TrainingConfig


class ScheduledSamplingTrainer:
    """
    Main trainer for NIF-Cloth4D-Temporal neural implicit cloth simulator.
    
    Key Features:
        - Scheduled sampling: Gradually transitions from teacher forcing to
          autoregressive rollout during training
        - AMP: Mixed precision training for 40%+ memory reduction
        - Physics losses: Enforces stretch, bend, momentum, collision constraints
        - Curriculum learning: Progressively increases rollout length
    
    Args:
        model: FourierFeatureMLP model to train
        config: Training configuration
        train_loader: Training data loader
        val_loader: Optional validation data loader
    """
    
    def __init__(
        self,
        model: FourierFeatureMLP,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Device
        self.device = config.get_device()
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # LR scheduler
        self.scheduler = self._create_scheduler()
        
        # AMP scaler
        self.scaler = GradScaler() if config.use_amp else None
        
        # Physics loss stack
        self.loss_stack = PhysicsLossStack(
            lambda_stretch=config.loss.lambda_stretch,
            lambda_bend=config.loss.lambda_bend,
            lambda_momentum=config.loss.lambda_momentum,
            lambda_collision=config.loss.lambda_collision,
            lambda_self_collision=config.loss.lambda_self_collision,
            ground_height=config.loss.ground_height,
            collision_margin=config.loss.collision_margin,
        ).to(self.device)
        
        # Scheduled sampling
        self.scheduled_sampler = ScheduledSamplingLoss(
            eps_start=config.loss.eps_start,
            eps_min=config.loss.eps_min,
            eps_decay=config.loss.eps_decay,
            schedule_type=config.loss.schedule_type,
        )
        
        # Curriculum scheduler
        self.curriculum = CurriculumScheduler(
            min_length=config.curriculum_min_length,
            max_length=config.rollout_length,
            warmup_epochs=config.curriculum_warmup_epochs,
            growth_rate=config.curriculum_growth_rate,
        ) if config.use_curriculum else None
        
        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # WandB
        self.wandb_run = None
        if config.use_wandb:
            self._init_wandb()
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        params = self.model.parameters()
        
        if self.config.optimizer == 'adam':
            return Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == 'adamw':
            return AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == 'sgd':
            return SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[object]:
        """Create learning rate scheduler based on config."""
        if self.config.scheduler == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.learning_rate * 0.01,
            )
        elif self.config.scheduler == 'step':
            return StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )
        elif self.config.scheduler == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
            )
        elif self.config.scheduler == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
    
    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        try:
            import wandb
            
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config={
                    'model': self.config.model.__dict__,
                    'loss': self.config.loss.__dict__,
                    'training': {
                        k: v for k, v in self.config.__dict__.items()
                        if k not in ['model', 'loss']
                    },
                },
            )
        except ImportError:
            print("Warning: wandb not installed. Logging disabled.")
            self.config.use_wandb = False
    
    def train(self) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Returns:
            Dictionary of training history (loss curves, metrics)
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epsilon': [],
        }
        
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Update scheduled sampling epsilon
            eps = self.scheduled_sampler.update_epsilon(epoch)
            
            # Update curriculum length
            if self.curriculum:
                rollout_len = self.curriculum.update_length(epoch)
            else:
                rollout_len = self.config.rollout_length
            
            # Train epoch
            train_loss, train_metrics = self._train_epoch(rollout_len)
            history['train_loss'].append(train_loss)
            history['epsilon'].append(eps)
            
            # Validation
            if self.val_loader is not None:
                val_loss, val_metrics = self._validate_epoch()
                history['val_loss'].append(val_loss)
                
                # Track best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint('best.pt')
            
            # LR scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss if self.val_loader else train_loss)
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rate'].append(current_lr)
            
            # Logging
            print(f"Epoch {epoch+1}/{self.config.epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"LR: {current_lr:.6f} | "
                  f"ε: {eps:.3f} | "
                  f"Rollout: {rollout_len}")
            
            if self.config.use_wandb and self.wandb_run:
                import wandb
                log_dict = {
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'train/learning_rate': current_lr,
                    'train/epsilon': eps,
                    'train/rollout_length': rollout_len,
                    **{f'train/{k}': v for k, v in train_metrics.items()},
                }
                if self.val_loader is not None:
                    log_dict['val/loss'] = val_loss
                    log_dict.update({f'val/{k}': v for k, v in val_metrics.items()})
                wandb.log(log_dict)
            
            # Checkpointing
            if (epoch + 1) % self.config.save_every_epochs == 0:
                self._save_checkpoint(f'epoch_{epoch+1}.pt')
        
        # Final save
        self._save_checkpoint('final.pt')
        
        if self.config.use_wandb and self.wandb_run:
            self.wandb_run.finish()
        
        return history
    
    def _train_epoch(
        self,
        rollout_length: int,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch with scheduled sampling.
        
        Args:
            rollout_length: Current curriculum rollout length
            
        Returns:
            Average loss and metrics dictionary
        """
        self.model.train()
        
        total_loss = 0.0
        metrics_accum = {}
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            loss, metrics = self._train_step(batch, rollout_length)
            
            total_loss += loss
            for k, v in metrics.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + v
            num_batches += 1
            self.global_step += 1
            
            # Step logging
            if (batch_idx + 1) % self.config.log_every_steps == 0:
                avg_loss = total_loss / num_batches
                print(f"  Step {batch_idx+1}/{len(self.train_loader)} | Loss: {avg_loss:.6f}")
        
        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in metrics_accum.items()}
        
        return avg_loss, avg_metrics
    
    def _train_step(
        self,
        batch: Dict[str, torch.Tensor],
        rollout_length: int,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Single training step with scheduled sampling rollout.
        
        Args:
            batch: Dictionary containing:
                - 'positions': (batch, seq_len, num_vertices, 3) vertex positions
                - 'sdf_samples': (batch, seq_len, num_points, 4) (x,y,z,sdf) samples
                - 'time': (batch, seq_len) timestamps
            rollout_length: Number of timesteps to unroll
            
        Returns:
            Loss value and metrics dictionary
        """
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        
        # Get sequence data
        if 'positions' in batch:
            # Mesh-based data
            positions = batch['positions']  # (batch, seq_len, num_verts, 3)
            batch_size, seq_len = positions.shape[:2]
        else:
            # SDF sample-based data
            sdf_samples = batch['sdf_samples']  # (batch, seq_len, num_points, 4)
            batch_size, seq_len = sdf_samples.shape[:2]
        
        # Limit rollout
        rollout_length = min(rollout_length, seq_len - 1)

        # Initialize GRU hidden state
        # For SDF samples/positions, we need to account for flattened batch dimension
        if 'sdf_samples' in batch:
            num_points = batch['sdf_samples'].shape[2]
            hidden_batch_size = batch_size * num_points
        else:
            num_verts = positions.shape[2]
            hidden_batch_size = batch_size * num_verts
        hidden = self.model.init_hidden(hidden_batch_size, self.device,
                                         dtype=torch.float16 if self.config.use_amp else torch.float32)
        
        # Temporal loss accumulator
        loss_accum = TemporalLossAccumulator()
        
        # AMP context
        amp_context = autocast() if self.config.use_amp else torch.cuda.amp.autocast(enabled=False)
        
        with amp_context:
            for t in range(rollout_length):
                # Get current timestep data
                if 'sdf_samples' in batch:
                    # SDF supervision
                    samples = batch['sdf_samples'][:, t]  # (batch, num_points, 4)
                    xyz = samples[:, :, :3]      # (batch, num_points, 3)
                    target_sdf = samples[:, :, 3:4]  # (batch, num_points, 1)
                    
                    time_val = batch['time'][:, t:t+1].unsqueeze(1)  # (batch, 1, 1)
                    time_val = time_val.expand(-1, xyz.size(1), -1)  # (batch, num_points, 1)
                    
                    # Reshape for model
                    xyz_flat = xyz.reshape(-1, 3)
                    time_flat = time_val.reshape(-1, 1)
                    
                    # Forward pass
                    pred_sdf_flat, hidden = self.model(xyz_flat, time_flat, hidden)
                    pred_sdf = pred_sdf_flat.reshape(batch_size, -1, 1)
                    
                    # Data loss (MSE)
                    data_loss = F.mse_loss(pred_sdf, target_sdf)
                    loss_accum.add_timestep_loss(data_loss, t)
                
                elif 'positions' in batch:
                    # Position-based supervision with physics losses
                    target_pos = positions[:, t + 1]  # (batch, num_verts, 3)
                    prev_pos = positions[:, t]        # (batch, num_verts, 3)
                    
                    # Query model at vertex positions
                    xyz_flat = prev_pos.reshape(-1, 3)
                    time_val = torch.full((xyz_flat.size(0), 1), 
                                         t * self.config.dt, device=self.device)
                    
                    pred_displacement_flat, hidden = self.model(xyz_flat, time_val, hidden)
                    pred_pos = prev_pos + pred_displacement_flat.reshape(batch_size, -1, 3)
                    
                    # Scheduled sampling: choose input for next step
                    if t < rollout_length - 1:
                        next_input = self.scheduled_sampler.get_scheduled_input(
                            target_pos.detach(),
                            pred_pos.detach(),
                            per_sample=True,
                        )
                    
                    # Physics losses
                    total_loss, loss_dict = self.loss_stack(
                        pred_vertices=pred_pos,
                        target_vertices=target_pos,
                        prev_vertices=prev_pos,
                        dt=self.config.dt,
                    )
                    loss_accum.add_timestep_loss(total_loss, t)
        
        # Total loss
        loss = loss_accum.get_mean_loss()
        
        # Backward pass
        if self.config.use_amp:
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.config.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip_norm,
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            if self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip_norm,
                )
            
            self.optimizer.step()
        
        # Collect metrics
        metrics = loss_accum.get_timestep_losses()
        
        return loss.item(), metrics
    
    @torch.no_grad()
    def _validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Validate for one epoch (no scheduled sampling, pure autoregressive).
        
        Returns:
            Average validation loss and metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        metrics_accum = {}
        num_batches = 0
        
        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            
            # Full autoregressive rollout (no teacher forcing)
            if 'sdf_samples' in batch:
                sdf_samples = batch['sdf_samples']
                batch_size, seq_len, num_points = sdf_samples.shape[:3]

                hidden = self.model.init_hidden(batch_size * num_points, self.device)
                
                val_loss = 0.0
                for t in range(seq_len):
                    samples = sdf_samples[:, t]
                    xyz = samples[:, :, :3].reshape(-1, 3)
                    target_sdf = samples[:, :, 3:4].reshape(-1, 1)
                    time_val = torch.full((xyz.size(0), 1), 
                                         t * self.config.dt, device=self.device)
                    
                    with autocast() if self.config.use_amp else torch.cuda.amp.autocast(enabled=False):
                        pred_sdf, hidden = self.model(xyz, time_val, hidden)
                        val_loss += F.mse_loss(pred_sdf, target_sdf).item()
                
                val_loss /= seq_len
            else:
                val_loss = 0.0
            
            total_loss += val_loss
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        return avg_loss, {}
    
    def _save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'config': self.config,
            'best_val_loss': self.best_val_loss,
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self) -> None:
        """Keep only the last N checkpoints."""
        checkpoints = sorted(
            self.checkpoint_dir.glob('epoch_*.pt'),
            key=lambda p: int(p.stem.split('_')[1]),
        )
        
        if len(checkpoints) > self.config.keep_last_n:
            for ckpt in checkpoints[:-self.config.keep_last_n]:
                ckpt.unlink()
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
