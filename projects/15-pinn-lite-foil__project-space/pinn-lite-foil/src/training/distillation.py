"""
Knowledge Distillation for PINN Compression
Trains a compact student model using teacher predictions + physics constraints.

Student Architecture: 4 layers × 32 units (vs Teacher: 8 layers × 128 units)

Loss:
    L_distill = MSE(student, teacher) + λ_phys * L_pde + λ_bc * L_bc

Usage:
    python distillation.py --teacher models/teacher/baseline_pinn.keras --epochs 5000 --output models/student/
"""

import argparse
import os
import time
import json
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers, callbacks
import h5py

from baseline_pinn import NavierStokesPINN, load_dataset, PINNCallback


class DistillationTrainer:
    """
    Knowledge distillation trainer for PINN compression.
    
    Uses soft targets from teacher model combined with physics-informed loss
    to train a compact student model.
    """
    
    def __init__(
        self,
        teacher: Model,
        student: Model,
        temperature: float = 2.0,
        alpha_distill: float = 0.7,
        alpha_physics: float = 0.3,
        lambda_pde: float = 100.0
    ):
        """
        Initialize distillation trainer.
        
        Args:
            teacher: Trained teacher PINN model
            student: Student PINN model to train
            temperature: Softening temperature for distillation
            alpha_distill: Weight for distillation loss
            alpha_physics: Weight for physics loss
            lambda_pde: PDE residual weight
        """
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha_distill = alpha_distill
        self.alpha_physics = alpha_physics
        self.lambda_pde = lambda_pde
        
        # Freeze teacher
        self.teacher.trainable = False
    
    @tf.function
    def distillation_loss(
        self,
        student_output: tf.Tensor,
        teacher_output: tf.Tensor,
        temperature: float = 1.0
    ) -> tf.Tensor:
        """
        Compute distillation loss with temperature scaling.
        
        For regression, we use MSE with temperature-scaled outputs.
        """
        # Temperature scaling (softens the distribution)
        student_scaled = student_output / temperature
        teacher_scaled = teacher_output / temperature
        
        # MSE loss (scaled by T^2 as per Hinton et al.)
        loss = tf.reduce_mean(tf.square(student_scaled - teacher_scaled))
        return loss * (temperature ** 2)
    
    @tf.function
    def compute_pde_residuals(
        self,
        model: Model,
        x: tf.Tensor,
        y: tf.Tensor,
        aoa: tf.Tensor,
        nu: float = 1e-3,
        rho: float = 1.0
    ) -> tf.Tensor:
        """Compute Navier-Stokes PDE residuals for student model."""
        x = tf.reshape(x, (-1, 1))
        y = tf.reshape(y, (-1, 1))
        aoa = tf.reshape(aoa, (-1, 1))
        
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, y])
            
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x, y])
                
                inputs = tf.concat([x, y, aoa], axis=-1)
                outputs = model(inputs, training=True)
                
                u = outputs[:, 0:1]
                v = outputs[:, 1:2]
                p = outputs[:, 2:3]
            
            u_x = tape1.gradient(u, x)
            u_y = tape1.gradient(u, y)
            v_x = tape1.gradient(v, x)
            v_y = tape1.gradient(v, y)
            p_x = tape1.gradient(p, x)
            p_y = tape1.gradient(p, y)
            
            del tape1
        
        u_xx = tape2.gradient(u_x, x)
        u_yy = tape2.gradient(u_y, y)
        v_xx = tape2.gradient(v_x, x)
        v_yy = tape2.gradient(v_y, y)
        
        del tape2
        
        # Continuity
        continuity = u_x + v_y
        
        # Momentum
        momentum_x = u * u_x + v * u_y + (1.0 / rho) * p_x - nu * (u_xx + u_yy)
        momentum_y = u * v_x + v * v_y + (1.0 / rho) * p_y - nu * (v_xx + v_yy)
        
        pde_loss = (
            tf.reduce_mean(tf.square(continuity)) +
            tf.reduce_mean(tf.square(momentum_x)) +
            tf.reduce_mean(tf.square(momentum_y))
        )
        
        return pde_loss
    
    def train_step(
        self,
        inputs: tf.Tensor,
        optimizer: optimizers.Optimizer
    ) -> Dict[str, float]:
        """
        Single training step with distillation + physics loss.
        
        Args:
            inputs: Input tensor [x, y, aoa]
            optimizer: Optimizer instance
        
        Returns:
            Dictionary of loss values
        """
        x = inputs[:, 0]
        y = inputs[:, 1]
        aoa = inputs[:, 2]
        
        # Get teacher predictions (no gradient)
        teacher_output = self.teacher(inputs, training=False)
        
        with tf.GradientTape() as tape:
            # Student predictions
            student_output = self.student(inputs, training=True)
            
            # Distillation loss
            distill_loss = self.distillation_loss(
                student_output, 
                teacher_output,
                self.temperature
            )
            
            # Physics loss
            pde_loss = self.compute_pde_residuals(self.student, x, y, aoa)
            
            # Combined loss
            total_loss = (
                self.alpha_distill * distill_loss +
                self.alpha_physics * self.lambda_pde * pde_loss
            )
        
        # Update student weights
        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
        
        return {
            'total_loss': float(total_loss),
            'distill_loss': float(distill_loss),
            'pde_loss': float(pde_loss)
        }
    
    def train(
        self,
        train_data: tf.data.Dataset,
        val_data: Optional[tf.data.Dataset] = None,
        epochs: int = 5000,
        learning_rate: float = 1e-3,
        log_interval: int = 100,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_interval: int = 500
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            epochs: Number of epochs
            learning_rate: Initial learning rate
            log_interval: Epochs between logging
            checkpoint_dir: Directory for checkpoints
            checkpoint_interval: Epochs between checkpoints
        
        Returns:
            Training history dictionary
        """
        # Learning rate schedule
        lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=epochs // 10,
            decay_rate=0.9
        )
        optimizer = optimizers.Adam(learning_rate=lr_schedule)
        
        history = {
            'total_loss': [],
            'distill_loss': [],
            'pde_loss': [],
            'val_loss': []
        }
        
        start_time = time.time()
        print("\n" + "="*60)
        print("Knowledge Distillation Training Started")
        print("="*60)
        
        for epoch in range(epochs):
            epoch_losses = {'total_loss': [], 'distill_loss': [], 'pde_loss': []}
            
            for batch in train_data:
                losses = self.train_step(batch, optimizer)
                for k, v in losses.items():
                    epoch_losses[k].append(v)
            
            # Average losses
            for k in epoch_losses:
                history[k].append(np.mean(epoch_losses[k]))
            
            # Validation
            if val_data is not None:
                val_losses = []
                for batch in val_data:
                    teacher_out = self.teacher(batch, training=False)
                    student_out = self.student(batch, training=False)
                    val_loss = tf.reduce_mean(tf.square(student_out - teacher_out))
                    val_losses.append(float(val_loss))
                history['val_loss'].append(np.mean(val_losses))
            
            # Logging
            if (epoch + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                msg = (f"Epoch {epoch+1:5d} | "
                       f"Loss: {history['total_loss'][-1]:.6f} | "
                       f"Distill: {history['distill_loss'][-1]:.6f} | "
                       f"PDE: {history['pde_loss'][-1]:.6f}")
                if val_data is not None:
                    msg += f" | Val: {history['val_loss'][-1]:.6f}"
                msg += f" | Time: {elapsed:.1f}s"
                print(msg)
            
            # Checkpointing
            if checkpoint_dir and (epoch + 1) % checkpoint_interval == 0:
                self.student.save_weights(
                    str(checkpoint_dir / f'student_epoch_{epoch+1:05d}.weights.h5')
                )
        
        elapsed = time.time() - start_time
        print("="*60)
        print(f"Distillation Complete! Total time: {elapsed:.1f}s")
        print("="*60 + "\n")
        
        return history


def create_student_model(
    hidden_layers: int = 4,
    hidden_units: int = 32,
    activation: str = 'tanh',
    nu: float = 1e-3
) -> NavierStokesPINN:
    """Create compact student PINN model."""
    return NavierStokesPINN(
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        activation=activation,
        nu=nu,
        lambda_data=1.0,
        lambda_pde=100.0,
        lambda_bc=10.0
    )


def apply_svd_compression(
    model: Model,
    rank: int = 10
) -> Model:
    """
    Apply low-rank SVD compression to dense layers.
    
    Args:
        model: Model to compress
        rank: Target rank for SVD truncation
    
    Returns:
        Compressed model
    """
    print(f"\nApplying SVD compression (rank={rank})...")
    
    compressed_weights = []
    
    for layer in model.layers:
        if isinstance(layer, layers.Dense):
            weights = layer.get_weights()
            if len(weights) == 2:  # weight matrix + bias
                W, b = weights
                
                # SVD decomposition
                U, S, Vt = np.linalg.svd(W, full_matrices=False)
                
                # Truncate to rank
                k = min(rank, len(S))
                U_k = U[:, :k]
                S_k = S[:k]
                Vt_k = Vt[:k, :]
                
                # Reconstruct
                W_compressed = U_k @ np.diag(S_k) @ Vt_k
                
                # Calculate compression ratio
                original_params = W.shape[0] * W.shape[1]
                compressed_params = k * (W.shape[0] + W.shape[1])
                ratio = original_params / compressed_params
                
                print(f"  {layer.name}: {W.shape} -> rank {k} (compression: {ratio:.2f}x)")
                
                compressed_weights.append([W_compressed, b])
            else:
                compressed_weights.append(weights)
        else:
            if hasattr(layer, 'get_weights'):
                compressed_weights.append(layer.get_weights())
    
    # Set compressed weights
    weight_idx = 0
    for layer in model.layers:
        if hasattr(layer, 'set_weights') and len(compressed_weights[weight_idx]) > 0:
            layer.set_weights(compressed_weights[weight_idx])
        weight_idx += 1
    
    return model


def compute_model_size(model: Model) -> float:
    """Compute model size in MB."""
    total_params = 0
    for layer in model.layers:
        for weight in layer.get_weights():
            total_params += weight.size
    
    # Assuming float32 (4 bytes per param)
    size_mb = (total_params * 4) / (1024 * 1024)
    return size_mb


def train_distillation(
    teacher_path: str,
    data_path: str,
    output_dir: str,
    epochs: int = 5000,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    student_layers: int = 4,
    student_units: int = 32,
    temperature: float = 2.0,
    svd_rank: int = 10,
    apply_svd: bool = True,
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
        svd_rank: Rank for SVD compression
        apply_svd: Whether to apply SVD compression
        quick_test: Quick test mode
    
    Returns:
        Training history
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if quick_test:
        epochs = min(50, epochs)
    
    # Load teacher model
    print(f"\nLoading teacher model from {teacher_path}...")
    teacher = keras.models.load_model(
        teacher_path,
        custom_objects={'NavierStokesPINN': NavierStokesPINN}
    )
    teacher_size = compute_model_size(teacher)
    print(f"  Teacher size: {teacher_size:.2f} MB")
    
    # Create student model
    print(f"\nCreating student model ({student_layers} layers × {student_units} units)...")
    student = create_student_model(
        hidden_layers=student_layers,
        hidden_units=student_units
    )
    student(tf.zeros((1, 3)))  # Build model
    student.summary()
    
    student_size = compute_model_size(student)
    print(f"  Student size: {student_size:.2f} MB")
    print(f"  Compression ratio: {teacher_size / student_size:.2f}x")
    
    # Load data
    print(f"\nLoading dataset from {data_path}...")
    train_inputs, _, _ = load_dataset(data_path, 'train')
    val_inputs, _, _ = load_dataset(data_path, 'test')
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(train_inputs)
    train_dataset = train_dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices(val_inputs)
    val_dataset = val_dataset.batch(batch_size)
    
    # Create trainer
    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        temperature=temperature,
        alpha_distill=0.7,
        alpha_physics=0.3,
        lambda_pde=100.0
    )
    
    # Train
    history = trainer.train(
        train_data=train_dataset,
        val_data=val_dataset,
        epochs=epochs,
        learning_rate=learning_rate,
        log_interval=100,
        checkpoint_dir=output_dir,
        checkpoint_interval=500
    )
    
    # Apply SVD compression
    if apply_svd:
        student = apply_svd_compression(student, rank=svd_rank)
        compressed_size = compute_model_size(student)
        print(f"\n  Compressed student size: {compressed_size:.2f} MB")
    
    # Save student model
    student.save_weights(str(output_dir / 'compressed_pinn.weights.h5'))
    student.save(str(output_dir / 'compressed_pinn.keras'))
    
    # Save config
    config = {
        'student_layers': student_layers,
        'student_units': student_units,
        'temperature': temperature,
        'svd_rank': svd_rank if apply_svd else None,
        'teacher_size_mb': float(teacher_size),
        'student_size_mb': float(student_size),
        'compression_ratio': float(teacher_size / student_size)
    }
    
    with open(output_dir / 'distillation_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save history
    with open(output_dir / 'distillation_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Compressed model saved to {output_dir}")
    print(f"  Final distillation loss: {history['distill_loss'][-1]:.6f}")
    
    return history


def main():
    parser = argparse.ArgumentParser(
        description='Knowledge distillation for PINN compression'
    )
    parser.add_argument(
        '--teacher', type=str, required=True,
        help='Path to trained teacher model'
    )
    parser.add_argument(
        '--data', type=str, required=True,
        help='Path to HDF5 dataset'
    )
    parser.add_argument(
        '--output', type=str, default='models/student/',
        help='Output directory'
    )
    parser.add_argument(
        '--epochs', type=int, default=5000,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1024,
        help='Batch size'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--student_layers', type=int, default=4,
        help='Student hidden layers'
    )
    parser.add_argument(
        '--student_units', type=int, default=32,
        help='Student units per layer'
    )
    parser.add_argument(
        '--temperature', type=float, default=2.0,
        help='Distillation temperature'
    )
    parser.add_argument(
        '--svd_rank', type=int, default=10,
        help='SVD compression rank'
    )
    parser.add_argument(
        '--no_svd', action='store_true',
        help='Disable SVD compression'
    )
    parser.add_argument(
        '--quick_test', action='store_true',
        help='Quick test mode'
    )
    
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
        svd_rank=args.svd_rank,
        apply_svd=not args.no_svd,
        quick_test=args.quick_test
    )


if __name__ == '__main__':
    main()
