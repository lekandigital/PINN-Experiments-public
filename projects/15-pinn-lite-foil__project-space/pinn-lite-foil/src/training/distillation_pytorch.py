"""
Knowledge Distillation for PINN Compression - PyTorch Version
Trains a compact student model using teacher predictions + physics constraints.

Student Architecture: 4 layers x 32 units (vs Teacher: 8 layers x 128 units)

Loss:
    L_distill = alpha * MSE(student, teacher) + (1-alpha) * lambda_pde * L_pde

Usage:
    python distillation_pytorch.py --teacher models/teacher/baseline_pinn.pt --data data/processed/training_data.h5 --output models/student/
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baseline_pinn_pytorch import NavierStokesPINN, load_dataset


def create_student_model(
    hidden_layers: int = 4,
    hidden_units: int = 32,
    activation: str = 'tanh',
    nu: float = 1e-3,
    device: str = 'cuda'
) -> NavierStokesPINN:
    """Create compact student PINN model."""
    return NavierStokesPINN(
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        activation=activation,
        nu=nu
    ).to(device)


def compute_model_size(model: nn.Module) -> float:
    """Compute model size in MB."""
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = (total_params * 4) / (1024 * 1024)  # float32 = 4 bytes
    return size_mb


def train_distillation(
    teacher_path: str,
    data_path: str,
    output_dir: str,
    epochs: int = 5000,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    student_layers: int = 4,
    student_units: int = 32,
    temperature: float = 2.0,
    alpha_distill: float = 0.7,
    lambda_pde: float = 10.0,
    checkpoint_interval: int = 500,
    log_interval: int = 100,
    device: str = 'cuda',
    quick_test: bool = False
) -> Dict:
    """
    Train compressed student model via knowledge distillation.

    Args:
        teacher_path: Path to trained teacher model
        data_path: Path to HDF5 dataset
        output_dir: Output directory
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        student_layers: Student hidden layers
        student_units: Student units per layer
        temperature: Distillation temperature
        alpha_distill: Weight for distillation loss (1-alpha for physics)
        lambda_pde: PDE residual weight
        checkpoint_interval: Epochs between checkpoints
        log_interval: Epochs between logging
        device: Device (cuda/cpu)
        quick_test: Quick test mode

    Returns:
        Training history
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if quick_test:
        epochs = min(50, epochs)

    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load teacher model config and weights
    teacher_config_path = Path(teacher_path).parent / 'model_config.json'
    with open(teacher_config_path, 'r') as f:
        teacher_config = json.load(f)

    print(f"\nLoading teacher model from {teacher_path}...")
    teacher = NavierStokesPINN(
        hidden_layers=teacher_config['hidden_layers'],
        hidden_units=teacher_config['hidden_units'],
        nu=teacher_config['nu']
    ).to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.eval()

    teacher_size = compute_model_size(teacher)
    print(f"  Teacher size: {teacher_size:.2f} MB")
    print(f"  Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")

    # Create student model
    print(f"\nCreating student model ({student_layers} layers x {student_units} units)...")
    student = create_student_model(
        hidden_layers=student_layers,
        hidden_units=student_units,
        nu=teacher_config['nu'],
        device=device
    )

    student_size = compute_model_size(student)
    print(f"  Student size: {student_size:.4f} MB")
    print(f"  Student params: {sum(p.numel() for p in student.parameters()):,}")
    print(f"  Compression ratio: {teacher_size / student_size:.2f}x")

    # Load data
    print(f"\nLoading dataset from {data_path}...")
    inputs, targets, stats = load_dataset(data_path, normalize=True, device=device)

    # Split train/val
    n_samples = len(inputs)
    n_train = int(0.9 * n_samples)

    indices = torch.randperm(n_samples)
    train_inputs = inputs[indices[:n_train]]
    val_inputs = inputs[indices[n_train:]]

    print(f"  Train samples: {len(train_inputs):,}")
    print(f"  Val samples: {len(val_inputs):,}")

    # Data loader
    train_dataset = TensorDataset(train_inputs)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    # Training history
    history = {
        'total_loss': [], 'distill_loss': [], 'pde_loss': [], 'val_loss': []
    }

    print("\n" + "="*60)
    print("Knowledge Distillation Training Started")
    print("="*60)

    start_time = time.time()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        student.train()
        epoch_total_loss = 0.0
        epoch_distill_loss = 0.0
        epoch_pde_loss = 0.0
        n_batches = 0

        for (batch_inputs,) in train_loader:
            optimizer.zero_grad()

            # Get teacher predictions (no gradient)
            with torch.no_grad():
                teacher_output = teacher(batch_inputs)

            # Student predictions
            student_output = student(batch_inputs)

            # Distillation loss with temperature
            student_scaled = student_output / temperature
            teacher_scaled = teacher_output / temperature
            distill_loss = torch.mean((student_scaled - teacher_scaled) ** 2) * (temperature ** 2)

            # PDE loss (on subset)
            pde_batch_size = min(512, len(batch_inputs))
            pde_idx = torch.randint(0, len(batch_inputs), (pde_batch_size,))

            x = batch_inputs[pde_idx, 0]
            y = batch_inputs[pde_idx, 1]
            aoa = batch_inputs[pde_idx, 2]

            cont, mom_x, mom_y = student.compute_pde_residuals(x, y, aoa)
            pde_loss = (
                torch.mean(cont ** 2) +
                torch.mean(mom_x ** 2) +
                torch.mean(mom_y ** 2)
            )

            # Combined loss
            total_loss = (
                alpha_distill * distill_loss +
                (1 - alpha_distill) * lambda_pde * pde_loss
            )

            # Backward
            total_loss.backward()
            optimizer.step()

            epoch_total_loss += total_loss.item()
            epoch_distill_loss += distill_loss.item()
            epoch_pde_loss += pde_loss.item()
            n_batches += 1

        # Average losses
        epoch_total_loss /= n_batches
        epoch_distill_loss /= n_batches
        epoch_pde_loss /= n_batches

        history['total_loss'].append(epoch_total_loss)
        history['distill_loss'].append(epoch_distill_loss)
        history['pde_loss'].append(epoch_pde_loss)

        # Validation
        student.eval()
        with torch.no_grad():
            teacher_val = teacher(val_inputs)
            student_val = student(val_inputs)
            val_loss = torch.mean((student_val - teacher_val) ** 2).item()

        history['val_loss'].append(val_loss)

        # Learning rate scheduling
        if (epoch + 1) % 50 == 0:
            scheduler.step()

        # Logging
        if (epoch + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:5d} | "
                  f"Loss: {epoch_total_loss:.6f} | "
                  f"Distill: {epoch_distill_loss:.6f} | "
                  f"PDE: {epoch_pde_loss:.6f} | "
                  f"Val: {val_loss:.6f} | "
                  f"Time: {elapsed:.1f}s")

        # Checkpointing
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = output_dir / f'student_epoch_{epoch+1:05d}.pt'
            torch.save(student.state_dict(), checkpoint_path)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(student.state_dict(), output_dir / 'best_student.pt')

    elapsed = time.time() - start_time
    print("="*60)
    print(f"Distillation Complete! Total time: {elapsed:.1f}s")
    print("="*60 + "\n")

    # Save final model
    torch.save(student.state_dict(), output_dir / 'compressed_pinn.pt')

    # Save config
    config = {
        'student_layers': student_layers,
        'student_units': student_units,
        'temperature': temperature,
        'alpha_distill': alpha_distill,
        'teacher_size_mb': float(teacher_size),
        'student_size_mb': float(student_size),
        'compression_ratio': float(teacher_size / student_size),
        'nu': teacher_config['nu']
    }

    with open(output_dir / 'distillation_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Save history
    with open(output_dir / 'distillation_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nCompressed model saved to {output_dir}")
    print(f"  Final distillation loss: {history['distill_loss'][-1]:.6f}")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Compression ratio: {teacher_size / student_size:.2f}x")

    return history


def main():
    parser = argparse.ArgumentParser(description='Knowledge distillation (PyTorch)')
    parser.add_argument('--teacher', type=str, required=True, help='Path to teacher model')
    parser.add_argument('--data', type=str, required=True, help='Path to HDF5 dataset')
    parser.add_argument('--output', type=str, default='models/student/', help='Output directory')
    parser.add_argument('--epochs', type=int, default=5000, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--student_layers', type=int, default=4, help='Student layers')
    parser.add_argument('--student_units', type=int, default=32, help='Student units')
    parser.add_argument('--temperature', type=float, default=2.0, help='Distillation temperature')
    parser.add_argument('--alpha', type=float, default=0.7, help='Distillation loss weight')
    parser.add_argument('--lambda_pde', type=float, default=10.0, help='PDE loss weight')
    parser.add_argument('--checkpoint_interval', type=int, default=500, help='Checkpoint frequency')
    parser.add_argument('--log_interval', type=int, default=100, help='Log frequency')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--quick_test', action='store_true', help='Quick test mode')

    args = parser.parse_args()

    train_distillation(
        teacher_path=args.teacher,
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        student_layers=args.student_layers,
        student_units=args.student_units,
        temperature=args.temperature,
        alpha_distill=args.alpha,
        lambda_pde=args.lambda_pde,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        device=args.device,
        quick_test=args.quick_test
    )


if __name__ == '__main__':
    main()
