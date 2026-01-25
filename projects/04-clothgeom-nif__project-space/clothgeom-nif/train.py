#!/usr/bin/env python3
"""
Training Script for ClothGeom-NIF

Trains the Neural Implicit Field decoder on synthetic cloth data
using combined SDF, Eikonal, and variance losses.

Usage:
    python train.py --data data/generated/cloth_dataset.h5 --epochs 100
    python train.py --config configs/train_config.yaml --data data/generated/cloth_dataset.h5
    python train.py --epochs 10 --batch_size 4 --quick  # Quick test run
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import NIFDecoder, create_nif_decoder
from data import ClothSDFDataset, create_dataloader


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class SDFLoss(nn.Module):
    """
    Combined loss function for SDF training.
    
    L = λ_sdf * L_sdf + λ_eik * L_eikonal + λ_var * L_variance
    
    where:
    - L_sdf: MSE between predicted and ground truth SDF
    - L_eikonal: ||∇SDF| - 1|² (gradient magnitude constraint)
    - L_variance: Regularization on variance prediction
    """
    
    def __init__(
        self,
        lambda_sdf: float = 1.0,
        lambda_eikonal: float = 0.1,
        lambda_variance: float = 0.01,
        clamp_sdf: float = 0.0,
        surface_weight: float = 1.0
    ):
        super().__init__()
        self.lambda_sdf = lambda_sdf
        self.lambda_eikonal = lambda_eikonal
        self.lambda_variance = lambda_variance
        self.clamp_sdf = clamp_sdf
        self.surface_weight = surface_weight
    
    def forward(
        self,
        pred_sdf: torch.Tensor,
        gt_sdf: torch.Tensor,
        pred_var: Optional[torch.Tensor] = None,
        gradient: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            pred_sdf: Predicted SDF values [B, 1]
            gt_sdf: Ground truth SDF values [B, 1]
            pred_var: Predicted variance [B, 1] (optional)
            gradient: SDF gradient [B, 3] (optional, for Eikonal loss)
        
        Returns:
            dict with 'total' loss and individual components
        """
        losses = {}
        
        # Clamp SDF values if specified
        if self.clamp_sdf > 0:
            pred_sdf_clamped = torch.clamp(pred_sdf, -self.clamp_sdf, self.clamp_sdf)
            gt_sdf_clamped = torch.clamp(gt_sdf, -self.clamp_sdf, self.clamp_sdf)
        else:
            pred_sdf_clamped = pred_sdf
            gt_sdf_clamped = gt_sdf
        
        # Surface weighting (higher weight for points near surface)
        if self.surface_weight != 1.0:
            surface_weights = 1.0 + (self.surface_weight - 1.0) * torch.exp(
                -10 * torch.abs(gt_sdf_clamped)
            )
        else:
            surface_weights = torch.ones_like(gt_sdf)
        
        # SDF loss (weighted MSE)
        sdf_loss = (surface_weights * (pred_sdf_clamped - gt_sdf_clamped)**2).mean()
        losses['sdf'] = self.lambda_sdf * sdf_loss
        
        # Eikonal loss (gradient magnitude should be 1)
        if gradient is not None and self.lambda_eikonal > 0:
            gradient_norm = torch.linalg.norm(gradient, dim=-1, keepdim=True)
            eikonal_loss = ((gradient_norm - 1.0)**2).mean()
            losses['eikonal'] = self.lambda_eikonal * eikonal_loss
        else:
            losses['eikonal'] = torch.tensor(0.0, device=pred_sdf.device)
        
        # Variance regularization
        if pred_var is not None and self.lambda_variance > 0:
            # Encourage low variance in general, but allow higher variance
            # in uncertain regions (this is a simple regularizer)
            var_loss = pred_var.mean()
            losses['variance'] = self.lambda_variance * var_loss
        else:
            losses['variance'] = torch.tensor(0.0, device=pred_sdf.device)
        
        # Total loss
        losses['total'] = losses['sdf'] + losses['eikonal'] + losses['variance']
        
        return losses


class Trainer:
    """
    Training manager for ClothGeom-NIF.
    
    Handles:
    - Model initialization and checkpointing
    - Training loop with validation
    - Mixed precision training (FP16)
    - Learning rate scheduling
    - Early stopping
    - Logging (console + TensorBoard)
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        data_path: str,
        output_dir: str = 'outputs'
    ):
        self.config = config
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        if self.device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Set random seed
        self._set_seed(config.get('seed', 42))
        
        # Initialize components
        self._setup_model()
        self._setup_data()
        self._setup_training()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if self.config.get('deterministic', False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _setup_model(self):
        """Initialize model."""
        model_cfg = self.config.get('model', {})
        
        self.model = create_nif_decoder(model_cfg)
        self.model = self.model.to(self.device)
        
        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {num_params:,}")
    
    def _setup_data(self):
        """Setup data loaders."""
        data_cfg = self.config.get('data', {})
        train_cfg = self.config.get('training', {})
        
        # Load full dataset
        full_dataset = ClothSDFDataset(
            self.data_path,
            num_points_per_sample=train_cfg.get('num_points', 8192),
            surface_ratio=data_cfg.get('surface_ratio', 0.5),
            surface_std=data_cfg.get('surface_std', 0.05),
            augment=data_cfg.get('augment', True)
        )
        
        # Split into train/val
        train_ratio = data_cfg.get('train_ratio', 0.8)
        n_train = int(len(full_dataset) * train_ratio)
        n_val = len(full_dataset) - n_train
        
        train_dataset, val_dataset = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(self.config.get('seed', 42))
        )
        
        # Disable augmentation for validation
        val_dataset.dataset = ClothSDFDataset(
            self.data_path,
            num_points_per_sample=train_cfg.get('num_points', 8192),
            surface_ratio=data_cfg.get('surface_ratio', 0.5),
            augment=False
        )
        
        # Create data loaders
        batch_size = train_cfg.get('batch_size', 8)
        num_workers = data_cfg.get('num_workers', 4)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        logger.info(f"Batch size: {batch_size}, Points per sample: {train_cfg.get('num_points', 8192)}")
    
    def _setup_training(self):
        """Setup optimizer, scheduler, loss, and other training components."""
        train_cfg = self.config.get('training', {})
        loss_cfg = self.config.get('loss', {})
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=train_cfg.get('learning_rate', 1e-4),
            weight_decay=train_cfg.get('weight_decay', 1e-6)
        )
        
        # Learning rate scheduler
        scheduler_type = train_cfg.get('lr_scheduler', 'cosine')
        epochs = train_cfg.get('epochs', 100)
        
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=train_cfg.get('lr_min', 1e-6)
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=epochs // 3,
                gamma=0.1
            )
        elif scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
        else:
            self.scheduler = None
        
        # Loss function
        self.criterion = SDFLoss(
            lambda_sdf=loss_cfg.get('lambda_sdf', 1.0),
            lambda_eikonal=loss_cfg.get('lambda_eikonal', 0.1),
            lambda_variance=loss_cfg.get('lambda_variance', 0.01),
            clamp_sdf=loss_cfg.get('clamp_sdf', 0.0),
            surface_weight=loss_cfg.get('surface_weight', 1.0)
        )
        
        # Mixed precision scaler
        self.use_amp = train_cfg.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient clipping
        self.grad_clip = train_cfg.get('grad_clip', 0.0)
        
        # TensorBoard
        if self.config.get('logging', {}).get('tensorboard', True):
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = self.output_dir / self.config.get('logging', {}).get('log_dir', 'logs')
                self.writer = SummaryWriter(log_dir)
            except ImportError:
                logger.warning("TensorBoard not available")
                self.writer = None
        else:
            self.writer = None
    
    def _flatten_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Flatten batch from [B, N, D] to [B*N, D]."""
        B, N = batch['coords'].shape[:2]
        
        coords = batch['coords'].reshape(B * N, -1)
        sdf = batch['sdf'].reshape(B * N, -1)
        
        # Expand latent to match each point
        latent = batch['latent'].unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        
        return {
            'coords': coords.to(self.device),
            'latent': latent.to(self.device),
            'sdf': sdf.to(self.device)
        }
    
    def _train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        epoch_losses = {'total': 0, 'sdf': 0, 'eikonal': 0, 'variance': 0}
        num_batches = 0
        
        train_cfg = self.config.get('training', {})
        log_every = self.config.get('logging', {}).get('log_every', 10)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Flatten and move to device
            flat_batch = self._flatten_batch(batch)
            coords = flat_batch['coords']
            latent = flat_batch['latent']
            gt_sdf = flat_batch['sdf']
            
            # Enable gradient computation for Eikonal loss
            coords.requires_grad_(True)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                pred_sdf, pred_var = self.model(coords, latent, return_variance=True)
                
                # Compute gradient for Eikonal loss
                if self.criterion.lambda_eikonal > 0:
                    gradient = torch.autograd.grad(
                        outputs=pred_sdf,
                        inputs=coords,
                        grad_outputs=torch.ones_like(pred_sdf),
                        create_graph=True,
                        retain_graph=True
                    )[0]
                else:
                    gradient = None
                
                # Compute loss
                losses = self.criterion(pred_sdf, gt_sdf, pred_var, gradient)
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(losses['total']).backward()
                
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total'].backward()
                
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            num_batches += 1
            
            # Log progress
            if (batch_idx + 1) % log_every == 0:
                avg_loss = epoch_losses['total'] / num_batches
                logger.info(
                    f"  Batch {batch_idx+1}/{len(self.train_loader)} | "
                    f"Loss: {losses['total'].item():.4f} | "
                    f"Avg: {avg_loss:.4f}"
                )
        
        # Average losses
        return {k: v / num_batches for k, v in epoch_losses.items()}
    
    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        val_losses = {'total': 0, 'sdf': 0, 'eikonal': 0, 'variance': 0}
        num_batches = 0
        
        for batch in self.val_loader:
            flat_batch = self._flatten_batch(batch)
            coords = flat_batch['coords']
            latent = flat_batch['latent']
            gt_sdf = flat_batch['sdf']
            
            # Forward pass (no gradient needed for validation)
            with autocast(enabled=self.use_amp):
                pred_sdf, pred_var = self.model(coords, latent, return_variance=True)
                losses = self.criterion(pred_sdf, gt_sdf, pred_var, None)
            
            for key in val_losses:
                val_losses[key] += losses[key].item()
            num_batches += 1
        
        return {k: v / num_batches for k, v in val_losses.items()}
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        train_cfg = self.config.get('training', {})
        checkpoint_dir = self.output_dir / self.config.get('output', {}).get('checkpoint_dir', 'checkpoints')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'model_config': self.model.get_config()
        }
        
        # Save latest
        latest_path = checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save epoch checkpoint
        if (self.current_epoch + 1) % train_cfg.get('checkpoint_every', 10) == 0:
            epoch_path = checkpoint_dir / f'epoch_{self.current_epoch+1:04d}.pt'
            torch.save(checkpoint, epoch_path)
            logger.info(f"Saved checkpoint: {epoch_path}")
            
            # Clean up old checkpoints
            keep_n = train_cfg.get('keep_last_n', 3)
            checkpoints = sorted(checkpoint_dir.glob('epoch_*.pt'))
            for ckpt in checkpoints[:-keep_n]:
                ckpt.unlink()
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint and resume training."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Resumed from epoch {self.current_epoch}")
    
    def train(self, epochs: Optional[int] = None):
        """Run full training loop."""
        if epochs is None:
            epochs = self.config.get('training', {}).get('epochs', 100)
        
        train_cfg = self.config.get('training', {})
        
        logger.info("=" * 60)
        logger.info("Starting training")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Training
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            train_losses = self._train_epoch()
            
            # Validation
            val_losses = self._validate()
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total'])
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start
            logger.info(
                f"  Train Loss: {train_losses['total']:.4f} "
                f"(sdf={train_losses['sdf']:.4f}, eik={train_losses['eikonal']:.4f})"
            )
            logger.info(
                f"  Val Loss: {val_losses['total']:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # TensorBoard logging
            if self.writer:
                for key, value in train_losses.items():
                    self.writer.add_scalar(f'train/{key}', value, epoch)
                for key, value in val_losses.items():
                    self.writer.add_scalar(f'val/{key}', value, epoch)
                self.writer.add_scalar('lr', current_lr, epoch)
            
            # Track losses
            self.train_losses.append(train_losses['total'])
            self.val_losses.append(val_losses['total'])
            
            # Check for best model
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
                logger.info(f"  ★ New best validation loss: {self.best_val_loss:.4f}")
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self._save_checkpoint(is_best)
            
            # Early stopping
            if train_cfg.get('early_stopping', True):
                patience = train_cfg.get('patience', 20)
                if self.patience_counter >= patience:
                    logger.info(f"\n⚠️ Early stopping triggered (patience={patience})")
                    break
        
        # Training complete
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info(f"  Total time: {total_time/60:.1f} minutes")
        logger.info(f"  Best val loss: {self.best_val_loss:.4f}")
        logger.info("=" * 60)
        
        if self.writer:
            self.writer.close()
        
        return self.best_val_loss


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train ClothGeom-NIF decoder',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/train_config.yaml',
        help='Path to config file'
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='data/generated/cloth_dataset.h5',
        help='Path to training data'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs',
        help='Output directory'
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    
    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test run with minimal settings'
    )
    
    parser.add_argument(
        '--resolution',
        type=int,
        default=None,
        help='SDF resolution for quick test'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        logger.warning(f"Config not found: {args.config}, using defaults")
        config = {}
    
    # Override with command line args
    if args.epochs:
        config.setdefault('training', {})['epochs'] = args.epochs
    if args.batch_size:
        config.setdefault('training', {})['batch_size'] = args.batch_size
    if args.lr:
        config.setdefault('training', {})['learning_rate'] = args.lr
    
    # Quick test mode
    if args.quick:
        logger.info("Quick test mode enabled")
        config.setdefault('training', {})['epochs'] = 5
        config.setdefault('training', {})['batch_size'] = 2
        config.setdefault('training', {})['num_points'] = 1000
        config.setdefault('data', {})['num_workers'] = 0
    
    # Check data exists
    if not os.path.exists(args.data):
        logger.error(f"Data not found: {args.data}")
        logger.info("Generate data first with: python generate_dataset.py")
        sys.exit(1)
    
    # Create trainer
    trainer = Trainer(
        config=config,
        data_path=args.data,
        output_dir=args.output
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(args.epochs)
    
    print("\n✅ Training complete!")
    print(f"   Best model saved to: {args.output}/checkpoints/best.pt")
    print(f"\n📊 Next steps:")
    print(f"   - Extract meshes: python extract_meshes.py")
    print(f"   - Run demo: python demo.py")


if __name__ == '__main__':
    main()
