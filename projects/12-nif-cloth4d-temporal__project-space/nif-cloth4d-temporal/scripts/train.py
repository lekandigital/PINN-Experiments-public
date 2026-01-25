#!/usr/bin/env python3
"""
Training script for NIF-Cloth4D-Temporal.

Usage:
    python scripts/train.py --epochs 100 --use_gru --use_wandb
    python scripts/train.py --config fast_dev  # Quick test
    python scripts/train.py --config l40s      # L40S optimized

For hyperparameter search:
    wandb sweep wandb_sweep.yaml
    wandb agent <sweep_id>
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.models import FourierFeatureMLP
from src.training import ScheduledSamplingTrainer, TrainingConfig
from src.training.config import get_default_config, get_fast_dev_config, get_l40s_config
from src.data import ClothSDFDataset
from src.data.sdf_dataset import create_dataloaders, SyntheticClothDataset
from src.utils import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train NIF-Cloth4D-Temporal neural cloth simulator'
    )
    
    # Configuration preset
    parser.add_argument(
        '--config', type=str, default='default',
        choices=['default', 'fast_dev', 'l40s'],
        help='Configuration preset to use'
    )
    
    # Model architecture
    parser.add_argument('--hidden_dim', type=int, help='Hidden layer dimension')
    parser.add_argument('--num_layers', type=int, help='Number of SIREN layers')
    parser.add_argument('--num_freqs', type=int, help='Number of Fourier frequencies')
    parser.add_argument('--omega_0', type=float, help='SIREN omega parameter')
    parser.add_argument('--use_gru', action='store_true', help='Enable GRU conditioning')
    parser.add_argument('--gru_hidden', type=int, help='GRU hidden dimension')
    
    # Training
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--max_steps', type=int, help='Maximum training steps (overrides epochs)')
    
    # Scheduled sampling
    parser.add_argument('--eps_start', type=float, help='Initial teacher forcing probability')
    parser.add_argument('--eps_min', type=float, help='Minimum teacher forcing probability')
    parser.add_argument('--eps_decay', type=float, help='Teacher forcing decay rate')
    
    # Physics losses
    parser.add_argument('--lambda_stretch', type=float, help='Stretch loss weight')
    parser.add_argument('--lambda_bend', type=float, help='Bend loss weight')
    parser.add_argument('--lambda_momentum', type=float, help='Momentum loss weight')
    parser.add_argument('--lambda_collision', type=float, help='Collision loss weight')
    
    # Data
    parser.add_argument('--data_path', type=str, help='Path to HDF5 dataset')
    parser.add_argument('--use_synthetic', action='store_true', 
                       help='Use synthetic data (no dataset file needed)')
    parser.add_argument('--sequence_length', type=int, help='Frames per sequence')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--wandb_project', type=str, help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, help='W&B run name')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, help='Checkpoint directory')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    # Hardware
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps'], 
                       help='Device to use')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision')
    
    return parser.parse_args()


def update_config_from_args(config: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    """Update configuration from command line arguments."""
    # Model
    if args.hidden_dim is not None:
        config.model.hidden_dim = args.hidden_dim
    if args.num_layers is not None:
        config.model.num_layers = args.num_layers
    if args.num_freqs is not None:
        config.model.num_freqs = args.num_freqs
    if args.omega_0 is not None:
        config.model.omega_0 = args.omega_0
    if args.use_gru:
        config.model.use_gru = True
    if args.gru_hidden is not None:
        config.model.gru_hidden = args.gru_hidden
    
    # Training
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.sequence_length is not None:
        config.rollout_length = args.sequence_length
    
    # Scheduled sampling
    if args.eps_start is not None:
        config.loss.eps_start = args.eps_start
    if args.eps_min is not None:
        config.loss.eps_min = args.eps_min
    if args.eps_decay is not None:
        config.loss.eps_decay = args.eps_decay
    
    # Physics losses
    if args.lambda_stretch is not None:
        config.loss.lambda_stretch = args.lambda_stretch
    if args.lambda_bend is not None:
        config.loss.lambda_bend = args.lambda_bend
    if args.lambda_momentum is not None:
        config.loss.lambda_momentum = args.lambda_momentum
    if args.lambda_collision is not None:
        config.loss.lambda_collision = args.lambda_collision
    
    # Logging
    if args.use_wandb:
        config.use_wandb = True
    if args.wandb_project is not None:
        config.wandb_project = args.wandb_project
    if args.wandb_run_name is not None:
        config.wandb_run_name = args.wandb_run_name
    
    # Checkpointing
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    
    # Hardware
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.no_amp:
        config.use_amp = False
    
    return config


def main():
    """Main training entry point."""
    args = parse_args()
    
    # Load configuration preset
    if args.config == 'fast_dev':
        config = get_fast_dev_config()
        print("Using fast development configuration")
    elif args.config == 'l40s':
        config = get_l40s_config()
        print("Using L40S optimized configuration")
    else:
        config = get_default_config()
        print("Using default configuration")
    
    # Update config from CLI args
    config = update_config_from_args(config, args)
    
    # Set random seed
    set_seed(config.seed)
    
    # Print configuration
    print("\n" + "=" * 60)
    print("NIF-Cloth4D-Temporal Training")
    print("=" * 60)
    print(f"Device: {config.get_device()}")
    print(f"Model: hidden_dim={config.model.hidden_dim}, layers={config.model.num_layers}")
    print(f"GRU: {config.model.use_gru} (hidden={config.model.gru_hidden})")
    print(f"Training: {config.epochs} epochs, batch_size={config.batch_size}")
    print(f"AMP: {config.use_amp}")
    print("=" * 60 + "\n")
    
    # Create model
    model = FourierFeatureMLP(
        in_dim=config.model.in_dim,
        hidden_dim=config.model.hidden_dim,
        out_dim=config.model.out_dim,
        num_layers=config.model.num_layers,
        num_freqs=config.model.num_freqs,
        fourier_scale=config.model.fourier_scale,
        omega_0=config.model.omega_0,
        use_gru=config.model.use_gru,
        gru_hidden=config.model.gru_hidden,
        dropout=config.model.dropout,
    )
    
    print(model)
    print(f"Total parameters: {model.count_parameters():,}\n")
    
    # Create data loaders
    if args.use_synthetic:
        print("Using synthetic dataset (on-the-fly generation)")
        from torch.utils.data import DataLoader
        
        train_dataset = SyntheticClothDataset(
            num_sequences=1000,
            sequence_length=config.rollout_length,
            cloth_res=16,
            num_samples_per_frame=1024,
            seed=config.seed,
        )
        val_dataset = SyntheticClothDataset(
            num_sequences=100,
            sequence_length=config.rollout_length,
            cloth_res=16,
            num_samples_per_frame=1024,
            seed=config.seed + 1,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
    elif args.data_path:
        print(f"Loading dataset from: {args.data_path}")
        train_loader, val_loader = create_dataloaders(
            data_path=args.data_path,
            batch_size=config.batch_size,
            sequence_length=config.rollout_length,
            num_workers=config.num_workers,
            seed=config.seed,
        )
    else:
        print("No dataset specified. Using synthetic data for demo.")
        print("Use --data_path to specify HDF5 dataset or --use_synthetic for synthetic data.")
        from torch.utils.data import DataLoader
        
        train_dataset = SyntheticClothDataset(
            num_sequences=100,
            sequence_length=config.rollout_length,
            cloth_res=16,
            seed=config.seed,
        )
        val_dataset = SyntheticClothDataset(
            num_sequences=20,
            sequence_length=config.rollout_length,
            cloth_res=16,
            seed=config.seed + 1,
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")
    
    # Create trainer
    trainer = ScheduledSamplingTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    
    # Resume from checkpoint
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    try:
        history = trainer.train()
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation loss: {trainer.best_val_loss:.6f}")
        print(f"Final model saved to: {config.checkpoint_dir}/final.pt")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted. Saving checkpoint...")
        trainer._save_checkpoint('interrupted.pt')
        print("Checkpoint saved. Run with --resume to continue.")


if __name__ == '__main__':
    main()
