#!/usr/bin/env python3
"""
train.py - Training Pipeline for Geom-INR-Motion

This script provides a complete training pipeline for the Geom-INR model,
including:

1. Data loading and preprocessing
2. Model initialization
3. Training loop with geometry losses
4. Learning rate scheduling
5. Checkpointing and logging
6. Evaluation during training

Training Configuration:
    - Optimizer: Adam (lr=1e-4)
    - Scheduler: StepLR (step_size=50, gamma=0.5)
    - Batch Size: 1024 (time-joint pairs)
    - Epochs: 100
    - Loss Weights:
        - MPJPE (base): 1.0
        - Curvature: 0.001
        - Torsion: 0.0001

Usage:
    # Train with synthetic data
    python train.py --synthetic --epochs 50
    
    # Train with AMASS data
    python train.py --data-dir /path/to/AMASS --epochs 100
    
    # Resume training
    python train.py --resume checkpoints/model_epoch50.pt

Dependencies:
    pip install torch numpy scipy tqdm tensorboard
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm

# Local imports
from model import GeomINR, create_model
from geometry_losses import GeometryLoss, compute_curvature_torsion
from data_pipeline import (
    MotionDataset, 
    create_synthetic_dataset,
    motion_collate_fn
)


# ============================================================================
# Training Configuration
# ============================================================================

class TrainingConfig:
    """Configuration for training."""
    
    def __init__(
        self,
        # Data
        data_dir: Optional[str] = None,
        use_synthetic: bool = True,
        num_synthetic_sequences: int = 20,
        num_frames: int = 100,
        num_joints: int = 24,
        
        # Model
        model_variant: str = "base",
        num_actors: int = 100,
        
        # Training
        epochs: int = 100,
        batch_size: int = 1024,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        
        # Scheduler
        scheduler_type: str = "step",  # "step" or "cosine"
        step_size: int = 50,
        gamma: float = 0.5,
        
        # Loss weights
        mpjpe_weight: float = 1.0,
        curvature_weight: float = 0.001,
        torsion_weight: float = 0.0001,
        jerk_weight: float = 0.0,
        bone_length_weight: float = 0.0,
        
        # Geometry loss computation
        geometry_loss_interval: int = 10,  # Compute every N batches
        trajectory_length: int = 50,  # Time steps for geometry loss
        
        # Checkpointing
        checkpoint_dir: str = "checkpoints",
        save_interval: int = 10,
        
        # Logging
        log_dir: str = "logs",
        log_interval: int = 10,
        use_tensorboard: bool = True,
        
        # Device
        device: str = "auto",
        
        # Reproducibility
        seed: int = 42
    ):
        self.data_dir = data_dir
        self.use_synthetic = use_synthetic
        self.num_synthetic_sequences = num_synthetic_sequences
        self.num_frames = num_frames
        self.num_joints = num_joints
        
        self.model_variant = model_variant
        self.num_actors = num_actors
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.scheduler_type = scheduler_type
        self.step_size = step_size
        self.gamma = gamma
        
        self.mpjpe_weight = mpjpe_weight
        self.curvature_weight = curvature_weight
        self.torsion_weight = torsion_weight
        self.jerk_weight = jerk_weight
        self.bone_length_weight = bone_length_weight
        
        self.geometry_loss_interval = geometry_loss_interval
        self.trajectory_length = trajectory_length
        
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.use_tensorboard = use_tensorboard
        
        self.device = device
        self.seed = seed
    
    def to_dict(self) -> Dict:
        return vars(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'TrainingConfig':
        return cls(**d)
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingConfig':
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))


# ============================================================================
# Trainer Class
# ============================================================================

class Trainer:
    """
    Trainer for Geom-INR-Motion model.
    
    Handles training loop, loss computation, checkpointing, and logging.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        
        # Set device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Set seed
        self._set_seed(config.seed)
        
        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.geometry_loss = None
        self.train_loader = None
        self.val_loader = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'curvature_loss': [],
            'torsion_loss': [],
            'mpjpe': []
        }
        
        # TensorBoard
        self.writer = None
        if config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.writer = SummaryWriter(os.path.join(config.log_dir, f"run_{timestamp}"))
            except ImportError:
                print("TensorBoard not available, skipping...")
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def setup_data(self):
        """Setup data loaders."""
        config = self.config
        
        if config.use_synthetic:
            print("Creating synthetic dataset...")
            sequences = create_synthetic_dataset(
                num_sequences=config.num_synthetic_sequences,
                num_frames=config.num_frames,
                num_joints=config.num_joints
            )
        else:
            print(f"Loading data from {config.data_dir}...")
            from data_pipeline import load_amass_dataset
            sequences = load_amass_dataset(config.data_dir)
            if not sequences:
                print("Warning: No sequences loaded, falling back to synthetic")
                sequences = create_synthetic_dataset()
        
        # Split into train/val
        split_idx = int(len(sequences) * 0.9)
        train_sequences = sequences[:split_idx]
        val_sequences = sequences[split_idx:] if split_idx < len(sequences) else sequences[:2]
        
        print(f"Train sequences: {len(train_sequences)}, Val sequences: {len(val_sequences)}")
        
        # Create datasets
        train_dataset = MotionDataset(
            train_sequences,
            samples_per_sequence=100,
            normalize=True
        )
        val_dataset = MotionDataset(
            val_sequences,
            samples_per_sequence=50,
            normalize=True
        )
        
        # Create loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=motion_collate_fn,
            num_workers=0,  # Set to 0 for compatibility
            pin_memory=self.device.type == "cuda"
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=motion_collate_fn,
            num_workers=0,
            pin_memory=self.device.type == "cuda"
        )
        
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
    
    def setup_model(self):
        """Initialize model, optimizer, and scheduler."""
        config = self.config
        
        # Create model
        self.model = create_model(
            variant=config.model_variant,
            num_joints=config.num_joints,
            num_actors=config.num_actors,
            max_time=float(config.num_frames)
        ).to(self.device)
        
        print(f"Model parameters: {self.model.get_num_params():,}")
        
        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        if config.scheduler_type == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=config.step_size,
                gamma=config.gamma
            )
        else:  # cosine
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.epochs,
                eta_min=config.learning_rate * 0.01
            )
        
        # Geometry loss
        self.geometry_loss = GeometryLoss(
            curvature_weight=config.curvature_weight,
            torsion_weight=config.torsion_weight,
            jerk_weight=config.jerk_weight,
            bone_length_weight=config.bone_length_weight
        )
    
    def compute_mpjpe(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Mean Per-Joint Position Error.
        
        Args:
            pred: (B, 3) predicted positions
            target: (B, 3) target positions
        
        Returns:
            MPJPE in same units as positions
        """
        return torch.norm(pred - target, dim=-1).mean()
    
    def compute_trajectory_loss(
        self,
        model: nn.Module,
        actor_id: int,
        start_time: float,
        num_steps: int
    ) -> Dict[str, torch.Tensor]:
        """
        Compute geometry losses on predicted trajectory.
        
        Args:
            model: GeomINR model
            actor_id: Actor ID for prediction
            start_time: Starting time
            num_steps: Number of time steps
        
        Returns:
            Dictionary of geometry losses
        """
        times = torch.linspace(
            start_time,
            start_time + num_steps - 1,
            num_steps,
            device=self.device
        )
        
        with torch.enable_grad():
            trajectory = model.predict_trajectory(actor_id, times, self.device)
        
        return self.geometry_loss(trajectory)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary of average metrics
        """
        self.model.train()
        config = self.config
        
        metrics = {
            'loss': 0.0,
            'mpjpe': 0.0,
            'curvature': 0.0,
            'torsion': 0.0
        }
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (actor_ids, joint_idxs, times, targets) in enumerate(pbar):
            # Move to device
            actor_ids = actor_ids.to(self.device)
            joint_idxs = joint_idxs.to(self.device)
            times = times.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(actor_ids, joint_idxs, times, apply_graph=False)
            
            # Base loss (MSE / MPJPE)
            mpjpe_loss = F.mse_loss(predictions, targets)
            total_loss = config.mpjpe_weight * mpjpe_loss
            
            # Geometry losses (computed periodically on trajectories)
            geo_losses = {'curvature': 0.0, 'torsion': 0.0}
            if batch_idx % config.geometry_loss_interval == 0:
                # Sample a random actor and time for trajectory
                sample_actor = actor_ids[0].item() % config.num_actors
                sample_start = max(0, times.min().item())
                
                try:
                    geo_dict = self.compute_trajectory_loss(
                        self.model,
                        sample_actor,
                        sample_start,
                        min(config.trajectory_length, config.num_frames - int(sample_start))
                    )
                    total_loss = total_loss + geo_dict['total']
                    geo_losses['curvature'] = geo_dict.get('curvature', torch.tensor(0.0)).item()
                    geo_losses['torsion'] = geo_dict.get('torsion', torch.tensor(0.0)).item()
                except Exception as e:
                    # Skip geometry loss on error
                    pass
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            metrics['loss'] += total_loss.item()
            metrics['mpjpe'] += mpjpe_loss.item()
            metrics['curvature'] += geo_losses['curvature']
            metrics['torsion'] += geo_losses['torsion']
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'mpjpe': f"{mpjpe_loss.item():.4f}"
            })
            
            # Logging
            if self.writer and batch_idx % config.log_interval == 0:
                self.writer.add_scalar('Train/Loss', total_loss.item(), self.global_step)
                self.writer.add_scalar('Train/MPJPE', mpjpe_loss.item(), self.global_step)
                self.writer.add_scalar('Train/LR', self.scheduler.get_last_lr()[0], self.global_step)
            
            self.global_step += 1
        
        # Average metrics
        for key in metrics:
            metrics[key] /= max(num_batches, 1)
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        metrics = {
            'loss': 0.0,
            'mpjpe': 0.0
        }
        num_batches = 0
        
        for actor_ids, joint_idxs, times, targets in self.val_loader:
            actor_ids = actor_ids.to(self.device)
            joint_idxs = joint_idxs.to(self.device)
            times = times.to(self.device)
            targets = targets.to(self.device)
            
            predictions = self.model(actor_ids, joint_idxs, times, apply_graph=False)
            
            loss = F.mse_loss(predictions, targets)
            mpjpe = self.compute_mpjpe(predictions, targets)
            
            metrics['loss'] += loss.item()
            metrics['mpjpe'] += mpjpe.item()
            num_batches += 1
        
        for key in metrics:
            metrics[key] /= max(num_batches, 1)
        
        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history,
            'config': self.config.to_dict()
        }
        
        # Save regular checkpoint
        path = os.path.join(self.config.checkpoint_dir, f"model_epoch{epoch}.pt")
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "model_best.pt")
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.history = checkpoint.get('history', self.history)
        
        print(f"Resumed from epoch {self.current_epoch}")
    
    def train(self, resume_path: Optional[str] = None):
        """
        Run full training loop.
        
        Args:
            resume_path: Path to checkpoint to resume from
        """
        # Setup
        self.setup_data()
        self.setup_model()
        
        # Resume if specified
        if resume_path:
            self.load_checkpoint(resume_path)
        
        # Save config
        config_path = os.path.join(self.config.checkpoint_dir, "config.json")
        self.config.save(config_path)
        
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Curvature weight: {self.config.curvature_weight}")
        print(f"Torsion weight: {self.config.torsion_weight}")
        print("=" * 60 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch + 1, self.config.epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            lr = self.scheduler.get_last_lr()[0]
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, MPJPE: {train_metrics['mpjpe']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, MPJPE: {val_metrics['mpjpe']:.4f}")
            print(f"  Curvature: {train_metrics['curvature']:.6f}, Torsion: {train_metrics['torsion']:.6f}")
            print(f"  LR: {lr:.6f}")
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['learning_rate'].append(lr)
            self.history['curvature_loss'].append(train_metrics['curvature'])
            self.history['torsion_loss'].append(train_metrics['torsion'])
            self.history['mpjpe'].append(val_metrics['mpjpe'])
            
            # TensorBoard
            if self.writer:
                self.writer.add_scalar('Epoch/TrainLoss', train_metrics['loss'], epoch)
                self.writer.add_scalar('Epoch/ValLoss', val_metrics['loss'], epoch)
                self.writer.add_scalar('Epoch/MPJPE', val_metrics['mpjpe'], epoch)
                self.writer.add_scalar('Epoch/Curvature', train_metrics['curvature'], epoch)
                self.writer.add_scalar('Epoch/Torsion', train_metrics['torsion'], epoch)
            
            # Checkpointing
            is_best = val_metrics['loss'] < self.best_loss
            if is_best:
                self.best_loss = val_metrics['loss']
            
            if epoch % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        # Training complete
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Total time: {elapsed / 60:.1f} minutes")
        print(f"Best validation loss: {self.best_loss:.4f}")
        print(f"Final MPJPE: {self.history['mpjpe'][-1]:.4f}")
        print("=" * 60)
        
        # Save final model
        self.save_checkpoint(self.config.epochs, is_best=False)
        
        # Save training history
        history_path = os.path.join(self.config.checkpoint_dir, "history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        if self.writer:
            self.writer.close()
        
        return self.history


# ============================================================================
# Main
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Geom-INR-Motion model")
    
    # Data
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to AMASS/H3.6M data directory")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for testing")
    parser.add_argument("--num-sequences", type=int, default=20,
                        help="Number of synthetic sequences")
    
    # Model
    parser.add_argument("--model", type=str, default="base",
                        choices=["small", "base", "large"],
                        help="Model variant")
    parser.add_argument("--num-joints", type=int, default=24,
                        help="Number of skeleton joints")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    
    # Loss weights
    parser.add_argument("--curvature-weight", type=float, default=0.001,
                        help="Curvature loss weight")
    parser.add_argument("--torsion-weight", type=float, default=0.0001,
                        help="Torsion loss weight")
    
    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Disable TensorBoard logging")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create config
    config = TrainingConfig(
        data_dir=args.data_dir,
        use_synthetic=args.synthetic or args.data_dir is None,
        num_synthetic_sequences=args.num_sequences,
        num_joints=args.num_joints,
        model_variant=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        curvature_weight=args.curvature_weight,
        torsion_weight=args.torsion_weight,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        use_tensorboard=not args.no_tensorboard
    )
    
    # Create trainer and run
    trainer = Trainer(config)
    history = trainer.train(resume_path=args.resume)
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
