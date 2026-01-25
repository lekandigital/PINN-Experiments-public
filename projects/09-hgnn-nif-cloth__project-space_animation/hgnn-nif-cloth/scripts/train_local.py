"""
Local Training Script for HGNN-NIF-Cloth

Usage:
    python scripts/train_local.py --epochs 100 --batch_size 8
    python scripts/train_local.py --test_mode  # Quick 3-epoch test
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.models.hybrid_model import HGNN_NIF_ClothModel
from src.data.synthetic_data import generate_synthetic_cloth_data
from src.data.dataset import create_dataloader, H5ClothDataset
from src.training.trainer import Trainer
from src.training.losses import PhysicsLoss


def main():
    parser = argparse.ArgumentParser(description='Train HGNN-NIF-Cloth model')
    
    # Data arguments
    parser.add_argument('--data', type=str, default='data/train_data.h5',
                        help='Path to training data (HDF5)')
    parser.add_argument('--generate_data', action='store_true',
                        help='Generate synthetic data if file not found')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to generate')
    
    # Model arguments
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Latent dimension')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--siren_hidden', type=int, default=128,
                        help='SIREN hidden dimension')
    parser.add_argument('--siren_layers', type=int, default=3,
                        help='Number of SIREN hidden layers')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--use_curriculum', action='store_true',
                        help='Use curriculum learning')
    
    # Loss arguments
    parser.add_argument('--lambda_spring', type=float, default=0.1,
                        help='Spring loss weight')
    parser.add_argument('--lambda_sdf', type=float, default=1.0,
                        help='SDF loss weight')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for checkpoints')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    
    # Test mode
    parser.add_argument('--test_mode', action='store_true',
                        help='Quick test with 3 epochs')
    
    args = parser.parse_args()
    
    # Test mode overrides
    if args.test_mode:
        args.epochs = 3
        args.batch_size = 2
        args.num_samples = 20
        args.generate_data = True
        print("=" * 60)
        print("TEST MODE: Running quick validation with 3 epochs")
        print("=" * 60)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check/generate data
    data_path = Path(args.data)
    if not data_path.exists():
        if args.generate_data:
            print(f"\nGenerating synthetic data ({args.num_samples} samples)...")
            data_path.parent.mkdir(parents=True, exist_ok=True)
            generate_synthetic_cloth_data(
                num_samples=args.num_samples,
                output_path=str(data_path)
            )
        else:
            print(f"ERROR: Data file not found: {data_path}")
            print("Use --generate_data to create synthetic data")
            sys.exit(1)
    
    print(f"\nLoading data from: {data_path}")
    
    # Create data loaders
    train_loader = create_dataloader(
        str(data_path), 'train',
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    val_loader = create_dataloader(
        str(data_path), 'val',
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = HGNN_NIF_ClothModel(
        node_feat_dim=3,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        siren_hidden_dim=args.siren_hidden,
        siren_layers=args.siren_layers
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )
    
    # Create loss function
    loss_fn = PhysicsLoss(
        lambda_spring=args.lambda_spring,
        lambda_sdf=args.lambda_sdf
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=str(device),
        output_dir=args.output_dir,
        use_amp=args.use_amp,
        gradient_clip=1.0,
        save_interval=10 if not args.test_mode else 1
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    history = trainer.train(
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        use_curriculum=args.use_curriculum
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"Best val loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")
    
    return history


if __name__ == '__main__':
    main()
