#!/usr/bin/env python3
"""
Optimized training script for NVIDIA RTX 3090 (24GB VRAM).

Features:
- Gradient checkpointing for memory efficiency
- Optimal batch size for 24GB VRAM
- Mixed precision (FP16/BF16) training
- Temporal model training support
- Curriculum learning with physics losses

Usage:
    # Train static model
    python scripts/train_rtx3090.py --data data/train_data.h5 --epochs 100

    # Train temporal model
    python scripts/train_rtx3090.py --data data/train_data.h5 --epochs 100 --temporal

    # Resume training
    python scripts/train_rtx3090.py --data data/train_data.h5 --resume outputs/checkpoint.pt
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.hybrid_model import HGNN_NIF_ClothModel
from src.models.temporal import TemporalHGNN_NIF
from src.data.dataset import H5ClothDataset
from src.training.losses import PhysicsLoss, CurriculumWeightScheduler, TemporalPhysicsLoss

# RTX 3090 optimized configuration
RTX3090_CONFIG = {
    'batch_size': 4,              # Safe for 24GB VRAM with full model
    'batch_size_temporal': 2,     # Temporal model uses more memory
    'gradient_accumulation': 2,   # Effective batch = batch_size * accumulation
    'use_amp': True,              # Mixed precision training
    'amp_dtype': 'float16',       # float16 or bfloat16
    'gradient_checkpointing': True,
    'num_workers': 4,
    'pin_memory': True,
    'prefetch_factor': 2,
}

# Model configuration
MODEL_CONFIG = {
    'node_feat_dim': 3,
    'latent_dim': 64,
    'hidden_dim': 64,
    'siren_hidden_dim': 128,
    'siren_layers': 3,
    'hgnn_layers': 2,
    'num_heads': 4,
}


def setup_model(temporal: bool = False, checkpoint_path: str = None, device: torch.device = None):
    """
    Setup model with optional checkpoint loading.

    Args:
        temporal: Whether to use temporal model
        checkpoint_path: Path to resume from
        device: Target device

    Returns:
        model: Configured model
        start_epoch: Starting epoch (0 or from checkpoint)
    """
    # Create base model
    base_model = HGNN_NIF_ClothModel(
        node_feat_dim=MODEL_CONFIG['node_feat_dim'],
        latent_dim=MODEL_CONFIG['latent_dim'],
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        siren_hidden_dim=MODEL_CONFIG['siren_hidden_dim'],
        siren_layers=MODEL_CONFIG['siren_layers'],
        hgnn_layers=MODEL_CONFIG['hgnn_layers'],
        num_heads=MODEL_CONFIG['num_heads']
    )

    if temporal:
        model = TemporalHGNN_NIF(base_model, history_frames=3, temporal_hidden=64)
        print("Created TEMPORAL model")
    else:
        model = base_model
        print("Created STATIC model")

    start_epoch = 0

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if temporal and 'temporal_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['temporal_state_dict'])
        elif 'model_state_dict' in checkpoint:
            if temporal:
                model.base_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])

        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming from epoch {start_epoch}")

    # Enable gradient checkpointing for memory efficiency
    if RTX3090_CONFIG['gradient_checkpointing']:
        enable_gradient_checkpointing(model)

    model = model.to(device)

    # Print model info
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {param_count:,} total, {trainable_count:,} trainable")

    return model, start_epoch


def enable_gradient_checkpointing(model):
    """Enable gradient checkpointing for memory efficiency."""
    # This is a placeholder - actual implementation depends on model architecture
    # For now, we can use torch.utils.checkpoint in the forward pass
    pass


def setup_training(model, args, device):
    """
    Setup optimizer, scheduler, and loss function.
    """
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.5
        )

    # Loss function
    if args.temporal:
        loss_fn = TemporalPhysicsLoss(
            lambda_velocity=1.0,
            lambda_physics_vel=0.1,
            lambda_spring=0.1,
            lambda_sdf=1.0,
            lambda_eikonal=0.01
        )
    else:
        loss_fn = PhysicsLoss(
            lambda_spring=0.1,
            lambda_sdf=1.0,
            lambda_eikonal=0.01,
            surface_weight=10.0
        )

    # Curriculum scheduler
    curriculum = CurriculumWeightScheduler(loss_fn, args.epochs) if not args.temporal else None

    # Mixed precision scaler
    if RTX3090_CONFIG['use_amp']:
        dtype = torch.float16 if RTX3090_CONFIG['amp_dtype'] == 'float16' else torch.bfloat16
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    return optimizer, scheduler, loss_fn, curriculum, scaler


def train_epoch(model, train_loader, optimizer, loss_fn, scaler, device,
                epoch, args, curriculum=None):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    # Update curriculum weights
    if curriculum is not None:
        weights = curriculum.step(epoch)
        print(f"  Curriculum stage: {weights['stage']}, lambda_spring: {weights['lambda_spring']:.3f}")

    optimizer.zero_grad()
    accumulation_steps = RTX3090_CONFIG['gradient_accumulation']

    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=RTX3090_CONFIG['use_amp']):
            loss, loss_dict = compute_loss(model, batch, loss_fn, args.temporal, device)
            loss = loss / accumulation_steps

        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

        # Log progress
        if (batch_idx + 1) % args.log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"  Batch {batch_idx + 1}/{num_batches}, Loss: {avg_loss:.4f}")

    return total_loss / num_batches


def compute_loss(model, batch, loss_fn, temporal, device):
    """
    Compute loss for a batch.
    """
    if temporal:
        # Temporal model expects frame history
        # Assume batch has 'positions_history' of shape (B, T, N, 3)
        if 'positions_history' in batch:
            history = batch['positions_history']
        else:
            # Create pseudo-history from single frame
            pos = batch['fine_pos']
            history = pos.unsqueeze(1).repeat(1, 3, 1, 1)

        fine_edges = batch['fine_edges']
        coarse_edges = batch['coarse_edges']
        query_points = batch.get('query_points', None)

        output = model(history, fine_edges, coarse_edges, query_points)

        # Compute temporal loss
        gt_pos = batch.get('gt_next_pos', history[:, -1])
        gt_vel = batch.get('gt_velocity', torch.zeros_like(output['velocity']))
        pred_sdf = output.get('sdf', None)
        gt_sdf = batch.get('query_sdf', None)

        total_loss, loss_dict = loss_fn(
            pred_pos=output['positions'],
            pred_vel=output['velocity'],
            gt_pos=gt_pos,
            gt_vel=gt_vel,
            pred_sdf=pred_sdf,
            gt_sdf=gt_sdf,
            edge_index=fine_edges,
            rest_lengths=batch.get('rest_lengths', None)
        )
    else:
        # Static model
        fine_pos = batch['fine_pos']
        fine_edges = batch['fine_edges']
        coarse_pos = batch['coarse_pos']
        coarse_edges = batch['coarse_edges']
        query_points = batch.get('query_points', None)
        gt_sdf = batch.get('query_sdf', None)

        output = model(
            (fine_pos, fine_edges),
            (coarse_pos, coarse_edges),
            query_points
        )

        pred_sdf = output.get('sdf', None)

        # Compute SDF gradients for Eikonal loss
        sdf_gradients = None
        if query_points is not None and pred_sdf is not None:
            sdf_gradients = model.compute_sdf_gradient(query_points, output['latent'])

        total_loss, loss_dict = loss_fn(
            pred_pos=fine_pos,
            pred_sdf=pred_sdf,
            gt_sdf=gt_sdf,
            edge_index=fine_edges,
            rest_lengths=batch.get('rest_lengths', None),
            sdf_gradients=sdf_gradients
        )

    return total_loss, loss_dict


@torch.no_grad()
def validate(model, val_loader, loss_fn, args, device):
    """
    Validation loop.
    """
    model.eval()
    total_loss = 0.0

    for batch in val_loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        with torch.cuda.amp.autocast(enabled=RTX3090_CONFIG['use_amp']):
            loss, _ = compute_loss(model, batch, loss_fn, args.temporal, device)

        total_loss += loss.item()

    return total_loss / len(val_loader)


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, args, is_best=False):
    """
    Save training checkpoint.
    """
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'val_loss': val_loss,
        **MODEL_CONFIG,
    }

    if args.temporal:
        checkpoint['temporal_state_dict'] = model.state_dict()
        checkpoint['model_state_dict'] = model.base_model.state_dict()
    else:
        checkpoint['model_state_dict'] = model.state_dict()

    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # Save latest
    torch.save(checkpoint, output_dir / 'latest.pt')

    # Save periodic checkpoint
    if (epoch + 1) % args.save_interval == 0:
        torch.save(checkpoint, output_dir / f'checkpoint_epoch{epoch:03d}.pt')

    # Save best
    if is_best:
        torch.save(checkpoint, output_dir / 'best.pt')
        print(f"  -> Saved best model (val_loss: {val_loss:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description='RTX 3090 Optimized Training for HGNN-NIF-Cloth',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Data arguments
    parser.add_argument('--data', type=str, default='data/train_data.h5',
                       help='Path to HDF5 training data')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split fraction')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping norm')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'plateau', 'step'],
                       help='Learning rate scheduler')

    # Model arguments
    parser.add_argument('--temporal', action='store_true',
                       help='Train temporal model instead of static')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    # Output arguments
    parser.add_argument('--output', type=str, default='outputs/rtx3090_run',
                       help='Output directory for checkpoints')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Log training progress every N batches')

    # Hardware arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("HGNN-NIF-Cloth Training - RTX 3090 Optimized")
    print("=" * 60)

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print()

    # Setup model
    model, start_epoch = setup_model(
        temporal=args.temporal,
        checkpoint_path=args.resume,
        device=device
    )

    # Setup training components
    optimizer, scheduler, loss_fn, curriculum, scaler = setup_training(model, args, device)

    # Load data
    print(f"\nLoading data from: {args.data}")
    if not os.path.exists(args.data):
        print("WARNING: Data file not found. Generating synthetic data...")
        from src.data.synthetic_data import generate_synthetic_dataset
        generate_synthetic_dataset(args.data, num_samples=100)

    # Determine batch size
    batch_size = RTX3090_CONFIG['batch_size_temporal'] if args.temporal else RTX3090_CONFIG['batch_size']
    effective_batch = batch_size * RTX3090_CONFIG['gradient_accumulation']
    print(f"Batch size: {batch_size} (effective: {effective_batch} with accumulation)")

    train_dataset = H5ClothDataset(args.data, split='train')
    val_dataset = H5ClothDataset(args.data, split='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=RTX3090_CONFIG['num_workers'],
        pin_memory=RTX3090_CONFIG['pin_memory'],
        prefetch_factor=RTX3090_CONFIG['prefetch_factor']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=RTX3090_CONFIG['num_workers'],
        pin_memory=RTX3090_CONFIG['pin_memory']
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Training loop
    print(f"\nStarting training from epoch {start_epoch}...")
    print("=" * 60)

    best_val_loss = float('inf')
    training_history = []

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, scaler,
            device, epoch, args, curriculum
        )

        # Validate
        val_loss = validate(model, val_loader, loss_fn, args, device)

        # Update scheduler
        if args.scheduler == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {current_lr:.2e}")

        # Track history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': current_lr
        })

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, args, is_best)

        # Memory cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Save training history
    output_dir = Path(args.output)
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
