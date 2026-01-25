"""
NIF-Cloth4D: Training Script

This script trains the NIF-Cloth4D neural implicit cloth model using
synthetic or simulation-derived SDF data.

Usage:
    python train_nif_cloth4d.py --config config_test.yaml --data_dir /tmp/cloth_test_data
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Local imports
from nif_cloth4d import FourierFeatureSIREN, NIFCloth4DLoss, create_model
from synthetic_data import load_sdf_from_hdf5, sample_points_from_sdf


class SDFDataset(Dataset):
    """
    PyTorch Dataset for SDF training data.
    
    Loads SDF volumes from HDF5 files and samples training points.
    """
    
    def __init__(
        self,
        data_dir: str,
        samples_per_frame: int = 10000,
        near_surface_ratio: float = 0.7,
        surface_band: float = 0.1,
        frame_indices: Optional[list] = None
    ):
        """
        Args:
            data_dir: Directory containing SDF HDF5 files
            samples_per_frame: Number of points to sample per frame
            near_surface_ratio: Fraction of samples near surface
            surface_band: SDF threshold for near-surface sampling
            frame_indices: Optional list of frame indices to use
        """
        self.data_dir = Path(data_dir) / 'sdf'
        self.samples_per_frame = samples_per_frame
        self.near_surface_ratio = near_surface_ratio
        self.surface_band = surface_band
        
        # Find all HDF5 files
        self.files = sorted(self.data_dir.glob('*.h5'))
        if frame_indices is not None:
            self.files = [self.files[i] for i in frame_indices if i < len(self.files)]
        
        if len(self.files) == 0:
            raise ValueError(f"No HDF5 files found in {self.data_dir}")
        
        print(f"SDFDataset: Found {len(self.files)} frames")
        
        # Pre-load all SDF data (for small datasets)
        self.sdf_data = []
        for f in self.files:
            sdf, coords, attrs = load_sdf_from_hdf5(str(f))
            self.sdf_data.append({
                'sdf': sdf,
                'coords': coords,
                'time': attrs['time']
            })
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get training samples for a frame.
        
        Returns:
            coords: (N, 4) tensor of (x, y, z, t) coordinates
            sdf_values: (N, 1) tensor of SDF values
        """
        data = self.sdf_data[idx]
        
        # Sample points from this frame
        points, sdf_vals = sample_points_from_sdf(
            sdf=data['sdf'],
            coords=data['coords'],
            time=data['time'],
            n_samples=self.samples_per_frame,
            near_surface_ratio=self.near_surface_ratio,
            surface_band=self.surface_band
        )
        
        return torch.from_numpy(points), torch.from_numpy(sdf_vals)


def collate_batch(batch):
    """Custom collate function to combine samples from multiple frames."""
    coords_list, sdf_list = zip(*batch)
    coords = torch.cat(coords_list, dim=0)
    sdf = torch.cat(sdf_list, dim=0)
    return coords, sdf


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    config: dict,
    filepath: str
):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }, filepath)


def load_checkpoint(filepath: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


class Trainer:
    """
    Training manager for NIF-Cloth4D.
    """
    
    def __init__(self, config: dict, data_dir: str, output_dir: str):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
            data_dir: Path to training data
            output_dir: Path for checkpoints and logs
        """
        self.config = config
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        self.device = torch.device(
            config.get('hardware', {}).get('device', 'cuda')
            if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Create model
        self.model = create_model(config['model']).to(self.device)
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {num_params:,}")
        
        # Create dataset and dataloader
        self._setup_data()
        
        # Optimizer
        train_cfg = config['training']
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=train_cfg['learning_rate'],
            weight_decay=train_cfg.get('weight_decay', 0.0)
        )
        
        # Scheduler
        scheduler_params = train_cfg.get('scheduler_params', {})
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=scheduler_params.get('T_max', train_cfg['epochs']),
            eta_min=scheduler_params.get('eta_min', 1e-6)
        )
        
        # Loss function
        loss_weights = train_cfg.get('loss_weights', {})
        self.loss_fn = NIFCloth4DLoss(
            sdf_weight=loss_weights.get('sdf', 1.0),
            eikonal_weight=loss_weights.get('eikonal', 0.01),
            stretch_weight=loss_weights.get('stretch', 0.0),
            bend_weight=loss_weights.get('bend', 0.0)
        )
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.start_time = None
    
    def _setup_data(self):
        """Setup datasets and dataloaders."""
        data_cfg = self.config['data']
        train_cfg = self.config['training']
        
        # Create full dataset
        full_dataset = SDFDataset(
            data_dir=self.data_dir,
            samples_per_frame=data_cfg.get('samples_per_frame', 10000),
            near_surface_ratio=data_cfg.get('near_surface_ratio', 0.7),
            surface_band=data_cfg.get('surface_band', 0.1)
        )
        
        # Split into train/val
        n_frames = len(full_dataset)
        val_split = data_cfg.get('val_split', 0.1)
        n_val = max(1, int(n_frames * val_split))
        n_train = n_frames - n_val
        
        # Random split
        indices = np.random.permutation(n_frames)
        train_indices = indices[:n_train].tolist()
        val_indices = indices[n_train:].tolist()
        
        # Create train dataset
        self.train_dataset = SDFDataset(
            data_dir=self.data_dir,
            samples_per_frame=data_cfg.get('samples_per_frame', 10000),
            near_surface_ratio=data_cfg.get('near_surface_ratio', 0.7),
            surface_band=data_cfg.get('surface_band', 0.1),
            frame_indices=train_indices
        )
        
        # Create val dataset
        self.val_dataset = SDFDataset(
            data_dir=self.data_dir,
            samples_per_frame=data_cfg.get('samples_per_frame', 10000) // 2,
            near_surface_ratio=data_cfg.get('near_surface_ratio', 0.7),
            surface_band=data_cfg.get('surface_band', 0.1),
            frame_indices=val_indices
        )
        
        hw_cfg = self.config.get('hardware', {})
        batch_size = train_cfg['batch_size']
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=max(1, len(self.train_dataset)),  # Load all frames per epoch
            shuffle=True,
            num_workers=hw_cfg.get('num_workers', 0),
            pin_memory=hw_cfg.get('pin_memory', False),
            collate_fn=collate_batch
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=max(1, len(self.val_dataset)),
            shuffle=False,
            num_workers=hw_cfg.get('num_workers', 0),
            pin_memory=hw_cfg.get('pin_memory', False),
            collate_fn=collate_batch
        )
        
        print(f"Train frames: {len(self.train_dataset)}, Val frames: {len(self.val_dataset)}")
    
    def train_epoch(self) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        batch_size = self.config['training']['batch_size']
        log_interval = self.config['training'].get('log_interval', 10)
        
        for batch_idx, (coords, sdf_gt) in enumerate(self.train_loader):
            # Move to device
            coords = coords.to(self.device)
            sdf_gt = sdf_gt.to(self.device)
            
            # Process in smaller batches if needed
            n_samples = coords.shape[0]
            n_batches = (n_samples + batch_size - 1) // batch_size
            
            epoch_batch_loss = 0.0
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                batch_coords = coords[start_idx:end_idx]
                batch_sdf_gt = sdf_gt[start_idx:end_idx]
                
                # Forward pass
                self.optimizer.zero_grad()
                sdf_pred = self.model(batch_coords)
                
                # Compute loss
                loss, loss_dict = self.loss_fn(sdf_pred, batch_sdf_gt)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_batch_loss += loss.item() * (end_idx - start_idx)
                total_samples += (end_idx - start_idx)
            
            total_loss += epoch_batch_loss
            
            # Logging
            if (batch_idx + 1) % log_interval == 0 or batch_idx == 0:
                avg_loss = epoch_batch_loss / n_samples
                elapsed = time.time() - self.start_time if self.start_time else 0
                print(f"  Batch {batch_idx+1}/{len(self.train_loader)}, "
                      f"Loss: {avg_loss:.6f}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e}, "
                      f"Time: {elapsed:.1f}s")
        
        return total_loss / total_samples if total_samples > 0 else 0.0
    
    @torch.no_grad()
    def validate(self) -> float:
        """Run validation. Returns average loss."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        for coords, sdf_gt in self.val_loader:
            coords = coords.to(self.device)
            sdf_gt = sdf_gt.to(self.device)
            
            sdf_pred = self.model(coords)
            loss, _ = self.loss_fn(sdf_pred, sdf_gt)
            
            total_loss += loss.item() * coords.shape[0]
            total_samples += coords.shape[0]
        
        return total_loss / total_samples if total_samples > 0 else 0.0
    
    def train(self):
        """Main training loop."""
        train_cfg = self.config['training']
        epochs = train_cfg['epochs']
        val_interval = train_cfg.get('val_interval', 5)
        checkpoint_interval = train_cfg.get('checkpoint_interval', 10)
        
        print("\n" + "="*60)
        print("Starting NIF-Cloth4D Training")
        print("="*60)
        print(f"Epochs: {epochs}")
        print(f"Batch size: {train_cfg['batch_size']}")
        print(f"Learning rate: {train_cfg['learning_rate']}")
        print(f"Output: {self.output_dir}")
        print("="*60 + "\n")
        
        self.start_time = time.time()
        
        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            epoch_start = time.time()
            
            print(f"\nEpoch {self.current_epoch}/{epochs}")
            print("-" * 40)
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Update scheduler
            self.scheduler.step()
            
            # Validate
            if (epoch + 1) % val_interval == 0 or epoch == epochs - 1:
                val_loss = self.validate()
                self.val_losses.append(val_loss)
                
                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    save_checkpoint(
                        self.model, self.optimizer, self.current_epoch, val_loss, self.config,
                        str(self.output_dir / 'model_best.pt')
                    )
                    print(f"  ✓ New best model saved (val_loss: {val_loss:.6f})")
            else:
                val_loss = None
            
            # Save checkpoint
            if (epoch + 1) % checkpoint_interval == 0 or epoch == epochs - 1:
                save_checkpoint(
                    self.model, self.optimizer, self.current_epoch, train_loss, self.config,
                    str(self.output_dir / f'model_epoch_{self.current_epoch}.pt')
                )
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            val_str = f", Val: {val_loss:.6f}" if val_loss is not None else ""
            print(f"  Train Loss: {train_loss:.6f}{val_str}")
            print(f"  Epoch time: {epoch_time:.1f}s")
            
            # Memory usage
            if self.device.type == 'cuda':
                mem_used = torch.cuda.max_memory_allocated() / 1e9
                print(f"  GPU memory: {mem_used:.2f} GB")
        
        # Final summary
        total_time = time.time() - self.start_time
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Final train loss: {self.train_losses[-1]:.6f}")
        print(f"Best val loss: {self.best_loss:.6f}")
        print(f"Checkpoints saved to: {self.output_dir}")
        print("="*60)
        
        # Save training history
        np.savez(
            self.output_dir / 'training_history.npz',
            train_losses=np.array(self.train_losses),
            val_losses=np.array(self.val_losses)
        )
        
        return self.model


def main():
    parser = argparse.ArgumentParser(description='Train NIF-Cloth4D')
    parser.add_argument('--config', type=str, default='config_test.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='/tmp/cloth_test_data',
                        help='Path to training data directory')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        # Use default config
        print(f"Config file not found: {args.config}, using defaults")
        config = {
            'model': {
                'in_dim': 4,
                'cond_dim': 0,
                'hidden_dim': 128,
                'hidden_layers': 4,
                'w0_initial': 30.0
            },
            'data': {
                'samples_per_frame': 10000,
                'near_surface_ratio': 0.7,
                'surface_band': 0.1,
                'val_split': 0.1
            },
            'training': {
                'epochs': 50,
                'batch_size': 8192,
                'learning_rate': 1e-4,
                'log_interval': 10,
                'val_interval': 5,
                'checkpoint_interval': 10,
                'loss_weights': {'sdf': 1.0, 'eikonal': 0.01}
            },
            'hardware': {
                'device': 'cuda',
                'num_workers': 0,
                'pin_memory': True
            }
        }
    
    # Override data_dir if specified
    config['data']['data_dir'] = args.data_dir
    
    # Create trainer
    trainer = Trainer(
        config=config,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Resume if specified
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        epoch, loss = load_checkpoint(args.resume, trainer.model, trainer.optimizer)
        trainer.current_epoch = epoch
        print(f"Resumed at epoch {epoch}, loss {loss:.6f}")
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
