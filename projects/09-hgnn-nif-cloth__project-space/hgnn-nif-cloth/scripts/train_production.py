#!/usr/bin/env python
"""
Production Training Script for HGNN-NIF-Cloth

Optimized for RTX 3090 (24GB VRAM) with:
- Large model configuration support
- Gradient accumulation for effective large batch sizes
- Mixed precision training (AMP)
- Comprehensive logging and checkpointing
- Resume from checkpoint capability

Usage:
    # Full production training
    python scripts/train_production.py --config configs/rtx3090_large.yaml --data data/production_train.h5

    # Resume from checkpoint
    python scripts/train_production.py --config configs/rtx3090_large.yaml --data data/production_train.h5 --resume outputs/production/checkpoint_epoch50.pth

    # Quick test
    python scripts/train_production.py --config configs/rtx3090_large.yaml --test_mode
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

from src.models.hybrid_model import HGNN_NIF_ClothModel, load_model_from_config, load_config
from src.data.dataset import H5ClothDataset, create_dataloader
from src.data.synthetic_data import generate_synthetic_cloth_data
from src.training.losses import PhysicsLoss, CurriculumWeightScheduler


class ProductionTrainer:
    """
    Production-grade trainer with gradient accumulation and comprehensive monitoring.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        loss_fn: PhysicsLoss,
        config: dict,
        output_dir: str,
        device: torch.device
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device

        # Training config
        train_cfg = config['training']
        self.epochs = train_cfg['epochs']
        self.lr = train_cfg['lr']
        self.weight_decay = train_cfg['weight_decay']
        self.warmup_epochs = train_cfg.get('warmup_epochs', 5)
        self.use_amp = train_cfg.get('use_amp', True) and device.type == 'cuda'
        self.gradient_clip = train_cfg.get('gradient_clip', 1.0)
        self.accumulation_steps = train_cfg.get('accumulation_steps', 1)
        self.save_interval = train_cfg.get('save_interval', 10)
        self.log_interval = train_cfg.get('log_interval', 10)
        self.use_curriculum = train_cfg.get('use_curriculum', True)

        # Setup output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=self.use_amp)

        # Learning rate scheduler
        self.scheduler = None
        self._setup_scheduler()

        # Curriculum learning
        self.curriculum = None
        if self.use_curriculum:
            self.curriculum = CurriculumWeightScheduler(self.loss_fn, self.epochs)

        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / 'logs')

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.rest_lengths = None

        # Log model info
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nModel parameters: {num_params:,} ({trainable_params:,} trainable)")

    def _setup_scheduler(self):
        """Setup learning rate scheduler with warmup + cosine annealing."""
        warmup_steps = self.warmup_epochs * len(self.train_loader)
        total_steps = self.epochs * len(self.train_loader)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            else:
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

    def _compute_rest_lengths(self, positions: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute rest edge lengths from initial positions."""
        src, tgt = edge_index[0], edge_index[1]
        edge_vecs = positions[src] - positions[tgt]
        return torch.norm(edge_vecs, dim=-1)

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch with gradient accumulation."""
        self.model.train()

        epoch_losses = {'total': 0.0, 'spring': 0.0, 'sdf': 0.0}
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            fine_pos = batch['fine_pos'].to(self.device)
            fine_edges = batch['fine_edges'].to(self.device)
            coarse_pos = batch['coarse_pos'].to(self.device)
            coarse_edges = batch['coarse_edges'].to(self.device)
            query_points = batch['query_points'].to(self.device)
            query_sdf = batch['query_sdf'].to(self.device)

            # Compute rest lengths on first batch
            if self.rest_lengths is None:
                self.rest_lengths = self._compute_rest_lengths(fine_pos[0], fine_edges)

            # Forward pass with mixed precision
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

                # Scale loss for gradient accumulation
                loss = loss / self.accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            # Update weights every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                if self.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

            # Track metrics
            epoch_losses['total'] += loss.item() * self.accumulation_steps
            for key in ['spring', 'sdf']:
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key].item()
            num_batches += 1
            self.global_step += 1

            # Logging
            if batch_idx % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{loss.item() * self.accumulation_steps:.4f}",
                    'lr': f"{current_lr:.2e}"
                })

                self.writer.add_scalar('train/loss', loss.item() * self.accumulation_steps, self.global_step)
                self.writer.add_scalar('train/lr', current_lr, self.global_step)

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)

        return epoch_losses

    def validate(self) -> Dict[str, float]:
        """Run validation loop."""
        if self.val_loader is None:
            return {}

        self.model.eval()

        val_losses = {'total': 0.0, 'spring': 0.0, 'sdf': 0.0}
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

    def train(self) -> Dict[str, Any]:
        """Full training loop."""
        print("\n" + "=" * 60)
        print("Starting Production Training")
        print("=" * 60)
        print(f"  Device: {self.device}")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"  Gradient accumulation: {self.accumulation_steps}x")
        print(f"  Effective batch size: {self.config['training']['batch_size'] * self.accumulation_steps}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Output: {self.output_dir}")

        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }

        start_time = time.time()

        for epoch in range(self.epoch, self.epochs):
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

            # Print progress
            msg = f"Epoch {epoch}: train_loss={train_metrics['total']:.4f}"
            if val_metrics:
                msg += f", val_loss={val_metrics['total']:.4f}"
            print(msg)

            # Checkpointing
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch{epoch+1}.pth")

            # Save best model
            if val_metrics and val_metrics['total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total']
                self.save_checkpoint("best_model.pth")
                print(f"  -> New best model saved! (val_loss={self.best_val_loss:.4f})")

        # Final save
        self.save_checkpoint("final_model.pth")

        elapsed = time.time() - start_time
        print(f"\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"  Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"  Checkpoints: {self.output_dir}")
        print(f"  TensorBoard: tensorboard --logdir {self.output_dir / 'logs'}")

        # Save history
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        self.writer.close()

        return history

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'rest_lengths': self.rest_lengths,
            'config': self.config
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, self.output_dir / filename)

    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.epoch = checkpoint['epoch'] + 1  # Start from next epoch
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.rest_lengths = checkpoint.get('rest_lengths')

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Resumed from checkpoint: epoch {checkpoint['epoch']}")


def main():
    parser = argparse.ArgumentParser(
        description='Production training for HGNN-NIF-Cloth',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--config', type=str, default='configs/rtx3090_large.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--data', type=str, default='data/production_train.h5',
                        help='Path to training data (HDF5)')

    # Optional arguments
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: outputs/production_YYYYMMDD)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    # Test mode
    parser.add_argument('--test_mode', action='store_true',
                        help='Quick test with 3 epochs and synthetic data')

    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    # Test mode overrides
    if args.test_mode:
        config['training']['epochs'] = 3
        config['training']['batch_size'] = 2
        config['training']['accumulation_steps'] = 1
        config['training']['save_interval'] = 1
        args.data = 'data/test_data.h5'
        print("=" * 60)
        print("TEST MODE: Running quick validation with 3 epochs")
        print("=" * 60)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'outputs/production_{timestamp}'

    # Check/generate data
    data_path = Path(args.data)
    if not data_path.exists():
        if args.test_mode:
            print(f"\nGenerating test data...")
            data_path.parent.mkdir(parents=True, exist_ok=True)
            data_cfg = config.get('data', {})
            generate_synthetic_cloth_data(
                num_samples=50,
                fine_resolution=data_cfg.get('fine_resolution', 20),
                coarse_stride=data_cfg.get('coarse_stride', 2),
                grid_res=data_cfg.get('sdf_resolution', 32),
                output_path=str(data_path)
            )
        else:
            print(f"ERROR: Data file not found: {data_path}")
            print("Generate data first with:")
            print(f"  python scripts/generate_large_dataset.py --config {args.config}")
            sys.exit(1)

    print(f"\nLoading data from: {data_path}")

    # Create data loaders
    train_cfg = config['training']
    data_cfg = config.get('data', {})

    train_loader = create_dataloader(
        str(data_path), 'train',
        batch_size=train_cfg['batch_size'],
        num_workers=train_cfg.get('num_workers', 4),
        num_query_points=data_cfg.get('query_points', 1000)
    )
    val_loader = create_dataloader(
        str(data_path), 'val',
        batch_size=train_cfg['batch_size'],
        num_workers=train_cfg.get('num_workers', 4),
        num_query_points=data_cfg.get('query_points', 1000)
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model from config
    print("\nCreating model...")
    model = load_model_from_config(args.config, device)

    # Create loss function
    loss_weights = config.get('loss_weights', {})
    loss_fn = PhysicsLoss(
        lambda_spring=loss_weights.get('spring', 0.1),
        lambda_sdf=loss_weights.get('sdf', 1.0)
    )

    # Create trainer
    trainer = ProductionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=config,
        output_dir=args.output_dir,
        device=device
    )

    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    history = trainer.train()

    return history


if __name__ == '__main__':
    main()
