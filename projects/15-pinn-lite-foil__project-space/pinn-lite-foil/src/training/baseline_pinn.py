"""
Baseline Physics-Informed Neural Network (PINN) for 2D Airfoil Flow
Implements the teacher model with 8 layers × 128 units.

Architecture:
    Input: [x, y, AoA] → 8 × 128 (tanh) → Output: [u, v, p]

Loss:
    L_total = λ_data * L_data + λ_pde * L_pde + λ_bc * L_bc

Usage:
    python baseline_pinn.py --data data/processed/dataset.h5 --epochs 10000 --output models/teacher/
"""

import argparse
import os
import time
import json
from pathlib import Path
from typing import Tuple, Dict, Optional, Callable

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers, callbacks
import h5py


# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


class NavierStokesPINN(Model):
    """
    Physics-Informed Neural Network for 2D incompressible Navier-Stokes equations.
    
    Enforces:
        - Continuity: ∂u/∂x + ∂v/∂y = 0
        - X-Momentum: u∂u/∂x + v∂u/∂y + (1/ρ)∂p/∂x - ν(∂²u/∂x² + ∂²u/∂y²) = 0
        - Y-Momentum: u∂v/∂x + v∂v/∂y + (1/ρ)∂p/∂y - ν(∂²v/∂x² + ∂²v/∂y²) = 0
    """
    
    def __init__(
        self,
        hidden_layers: int = 8,
        hidden_units: int = 128,
        activation: str = 'tanh',
        nu: float = 1e-3,
        rho: float = 1.0,
        lambda_data: float = 1.0,
        lambda_pde: float = 100.0,
        lambda_bc: float = 10.0,
        **kwargs
    ):
        """
        Initialize PINN model.
        
        Args:
            hidden_layers: Number of hidden layers
            hidden_units: Units per hidden layer
            activation: Activation function
            nu: Kinematic viscosity
            rho: Density
            lambda_data: Weight for data loss
            lambda_pde: Weight for PDE residual loss
            lambda_bc: Weight for boundary condition loss
        """
        super().__init__(**kwargs)
        
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.nu = tf.constant(nu, dtype=tf.float32)
        self.rho = tf.constant(rho, dtype=tf.float32)
        self.lambda_data = lambda_data
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        
        # Build network
        self.dense_layers = []
        for i in range(hidden_layers):
            self.dense_layers.append(
                layers.Dense(
                    hidden_units,
                    activation=activation,
                    kernel_initializer='glorot_normal',
                    name=f'hidden_{i}'
                )
            )
        
        # Output heads (u, v, p)
        self.output_u = layers.Dense(1, name='output_u')
        self.output_v = layers.Dense(1, name='output_v')
        self.output_p = layers.Dense(1, name='output_p')
        
        # Store config for serialization
        self._config = {
            'hidden_layers': hidden_layers,
            'hidden_units': hidden_units,
            'activation': activation,
            'nu': float(nu),
            'rho': float(rho),
            'lambda_data': lambda_data,
            'lambda_pde': lambda_pde,
            'lambda_bc': lambda_bc
        }
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Tensor of shape (batch, 3) with [x, y, aoa]
        
        Returns:
            Tensor of shape (batch, 3) with [u, v, p]
        """
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        
        u = self.output_u(x)
        v = self.output_v(x)
        p = self.output_p(x)
        
        return tf.concat([u, v, p], axis=-1)
    
    @tf.function
    def compute_pde_residuals(
        self, 
        x: tf.Tensor, 
        y: tf.Tensor, 
        aoa: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Compute Navier-Stokes PDE residuals using automatic differentiation.
        
        Args:
            x: x-coordinates (batch,)
            y: y-coordinates (batch,)
            aoa: angles of attack (batch,)
        
        Returns:
            Tuple of (continuity, momentum_x, momentum_y) residuals
        """
        x = tf.reshape(x, (-1, 1))
        y = tf.reshape(y, (-1, 1))
        aoa = tf.reshape(aoa, (-1, 1))
        
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, y])
            
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x, y])
                
                inputs = tf.concat([x, y, aoa], axis=-1)
                outputs = self(inputs, training=True)
                
                u = outputs[:, 0:1]
                v = outputs[:, 1:2]
                p = outputs[:, 2:3]
            
            # First derivatives
            u_x = tape1.gradient(u, x)
            u_y = tape1.gradient(u, y)
            v_x = tape1.gradient(v, x)
            v_y = tape1.gradient(v, y)
            p_x = tape1.gradient(p, x)
            p_y = tape1.gradient(p, y)
            
            del tape1
        
        # Second derivatives
        u_xx = tape2.gradient(u_x, x)
        u_yy = tape2.gradient(u_y, y)
        v_xx = tape2.gradient(v_x, x)
        v_yy = tape2.gradient(v_y, y)
        
        del tape2
        
        # Continuity: ∂u/∂x + ∂v/∂y = 0
        continuity = u_x + v_y
        
        # X-Momentum: u∂u/∂x + v∂u/∂y + (1/ρ)∂p/∂x - ν(∂²u/∂x² + ∂²u/∂y²) = 0
        momentum_x = (
            u * u_x + v * u_y 
            + (1.0 / self.rho) * p_x 
            - self.nu * (u_xx + u_yy)
        )
        
        # Y-Momentum: u∂v/∂x + v∂v/∂y + (1/ρ)∂p/∂y - ν(∂²v/∂x² + ∂²v/∂y²) = 0
        momentum_y = (
            u * v_x + v * v_y 
            + (1.0 / self.rho) * p_y 
            - self.nu * (v_xx + v_yy)
        )
        
        return continuity, momentum_x, momentum_y
    
    def train_step(self, data):
        """Custom training step with physics-informed loss."""
        # Unpack data
        if len(data) == 2:
            inputs, targets = data
            sample_weight = None
        else:
            inputs, targets, sample_weight = data
        
        x = inputs[:, 0]
        y = inputs[:, 1]
        aoa = inputs[:, 2]
        
        with tf.GradientTape() as tape:
            # Data loss
            predictions = self(inputs, training=True)
            data_loss = tf.reduce_mean(tf.square(predictions - targets))
            
            # PDE residual loss
            cont, mom_x, mom_y = self.compute_pde_residuals(x, y, aoa)
            pde_loss = (
                tf.reduce_mean(tf.square(cont)) +
                tf.reduce_mean(tf.square(mom_x)) +
                tf.reduce_mean(tf.square(mom_y))
            )
            
            # Total loss
            total_loss = (
                self.lambda_data * data_loss +
                self.lambda_pde * pde_loss
            )
        
        # Compute gradients and update weights
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update metrics
        return {
            'loss': total_loss,
            'data_loss': data_loss,
            'pde_loss': pde_loss
        }
    
    def test_step(self, data):
        """Custom test step."""
        inputs, targets = data
        
        x = inputs[:, 0]
        y = inputs[:, 1]
        aoa = inputs[:, 2]
        
        predictions = self(inputs, training=False)
        data_loss = tf.reduce_mean(tf.square(predictions - targets))
        
        cont, mom_x, mom_y = self.compute_pde_residuals(x, y, aoa)
        pde_loss = (
            tf.reduce_mean(tf.square(cont)) +
            tf.reduce_mean(tf.square(mom_x)) +
            tf.reduce_mean(tf.square(mom_y))
        )
        
        total_loss = self.lambda_data * data_loss + self.lambda_pde * pde_loss
        
        return {
            'loss': total_loss,
            'data_loss': data_loss,
            'pde_loss': pde_loss
        }
    
    def get_config(self):
        """Return model configuration."""
        return self._config
    
    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        return cls(**config)


def load_dataset(
    filepath: str,
    split: str = 'train',
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load dataset from HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
        split: 'train' or 'test'
        normalize: Whether to normalize data
    
    Returns:
        Tuple of (inputs, targets, normalization_stats)
    """
    all_x, all_y, all_aoa = [], [], []
    all_u, all_v, all_p = [], [], []
    
    with h5py.File(filepath, 'r') as f:
        grp = f[split]
        
        for case_name in grp.keys():
            case = grp[case_name]
            
            all_x.append(case['x'][:])
            all_y.append(case['y'][:])
            all_u.append(case['u'][:])
            all_v.append(case['v'][:])
            all_p.append(case['p'][:])
            
            aoa = case.attrs['aoa']
            all_aoa.append(np.full_like(case['x'][:], aoa))
    
    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    aoa = np.concatenate(all_aoa)
    u = np.concatenate(all_u)
    v = np.concatenate(all_v)
    p = np.concatenate(all_p)
    
    # Stack inputs and outputs
    inputs = np.stack([x, y, aoa], axis=-1).astype(np.float32)
    targets = np.stack([u, v, p], axis=-1).astype(np.float32)
    
    stats = {}
    
    if normalize:
        # Normalize inputs
        input_mean = inputs.mean(axis=0)
        input_std = inputs.std(axis=0) + 1e-8
        inputs = (inputs - input_mean) / input_std
        
        # Normalize outputs
        target_mean = targets.mean(axis=0)
        target_std = targets.std(axis=0) + 1e-8
        targets = (targets - target_mean) / target_std
        
        stats = {
            'input_mean': input_mean.tolist(),
            'input_std': input_std.tolist(),
            'target_mean': target_mean.tolist(),
            'target_std': target_std.tolist()
        }
    
    return inputs, targets, stats


class PINNCallback(callbacks.Callback):
    """Custom callback for PINN training progress."""
    
    def __init__(self, log_interval: int = 100):
        super().__init__()
        self.log_interval = log_interval
        self.start_time = None
    
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print("\n" + "="*60)
        print("PINN Training Started")
        print("="*60)
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_interval == 0:
            elapsed = time.time() - self.start_time
            print(f"Epoch {epoch+1:5d} | "
                  f"Loss: {logs['loss']:.6f} | "
                  f"Data: {logs['data_loss']:.6f} | "
                  f"PDE: {logs['pde_loss']:.6f} | "
                  f"Time: {elapsed:.1f}s")
    
    def on_train_end(self, logs=None):
        elapsed = time.time() - self.start_time
        print("="*60)
        print(f"Training Complete! Total time: {elapsed:.1f}s")
        print("="*60 + "\n")


def train_pinn(
    data_path: str,
    output_dir: str,
    epochs: int = 10000,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    hidden_layers: int = 8,
    hidden_units: int = 128,
    nu: float = 1e-3,
    lambda_data: float = 1.0,
    lambda_pde: float = 100.0,
    checkpoint_interval: int = 1000,
    quick_test: bool = False
) -> Dict:
    """
    Train baseline PINN model.
    
    Args:
        data_path: Path to HDF5 dataset
        output_dir: Output directory for models and logs
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        hidden_layers: Number of hidden layers
        hidden_units: Units per layer
        nu: Kinematic viscosity
        lambda_data: Data loss weight
        lambda_pde: PDE loss weight
        checkpoint_interval: Epochs between checkpoints
        quick_test: Run quick test with reduced epochs
    
    Returns:
        Training history dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Quick test mode
    if quick_test:
        epochs = min(100, epochs)
        checkpoint_interval = 50
    
    print(f"\nLoading dataset from {data_path}...")
    train_inputs, train_targets, stats = load_dataset(data_path, 'train')
    val_inputs, val_targets, _ = load_dataset(data_path, 'test')
    
    print(f"  Train samples: {len(train_inputs):,}")
    print(f"  Val samples: {len(val_inputs):,}")
    
    # Save normalization stats
    with open(output_dir / 'normalization_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Create model
    print(f"\nCreating PINN model ({hidden_layers} layers × {hidden_units} units)...")
    model = NavierStokesPINN(
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        nu=nu,
        lambda_data=lambda_data,
        lambda_pde=lambda_pde
    )
    
    # Build model
    model(tf.zeros((1, 3)))
    model.summary()
    
    # Optimizer with learning rate schedule
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=epochs // 10,
        decay_rate=0.9
    )
    optimizer = optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(optimizer=optimizer)
    
    # Callbacks
    callback_list = [
        PINNCallback(log_interval=100),
        callbacks.ModelCheckpoint(
            filepath=str(output_dir / 'checkpoint_epoch_{epoch:05d}.weights.h5'),
            save_weights_only=True,
            save_freq=checkpoint_interval
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=500,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=200,
            min_lr=1e-6
        )
    ]
    
    # Train
    print(f"\nStarting training for {epochs} epochs...")
    history = model.fit(
        train_inputs, train_targets,
        validation_data=(val_inputs, val_targets),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback_list,
        verbose=0
    )
    
    # Save final model
    model.save_weights(str(output_dir / 'baseline_pinn.weights.h5'))
    
    # Save full model (requires custom objects)
    model.save(str(output_dir / 'baseline_pinn.keras'))
    
    # Save model config
    with open(output_dir / 'model_config.json', 'w') as f:
        json.dump(model.get_config(), f, indent=2)
    
    # Save training history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"\n✓ Model saved to {output_dir}")
    print(f"  Final train loss: {history.history['loss'][-1]:.6f}")
    print(f"  Final val loss: {history.history['val_loss'][-1]:.6f}")
    
    return history_dict


def main():
    parser = argparse.ArgumentParser(
        description='Train baseline PINN for 2D airfoil flow'
    )
    parser.add_argument(
        '--data', type=str, required=True,
        help='Path to HDF5 dataset'
    )
    parser.add_argument(
        '--output', type=str, default='models/teacher/',
        help='Output directory'
    )
    parser.add_argument(
        '--epochs', type=int, default=10000,
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
        '--hidden_layers', type=int, default=8,
        help='Number of hidden layers'
    )
    parser.add_argument(
        '--hidden_units', type=int, default=128,
        help='Units per layer'
    )
    parser.add_argument(
        '--nu', type=float, default=1e-3,
        help='Kinematic viscosity'
    )
    parser.add_argument(
        '--lambda_data', type=float, default=1.0,
        help='Data loss weight'
    )
    parser.add_argument(
        '--lambda_pde', type=float, default=100.0,
        help='PDE loss weight'
    )
    parser.add_argument(
        '--checkpoint_interval', type=int, default=1000,
        help='Epochs between checkpoints'
    )
    parser.add_argument(
        '--quick_test', action='store_true',
        help='Quick test mode (100 epochs)'
    )
    
    args = parser.parse_args()
    
    train_pinn(
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_layers=args.hidden_layers,
        hidden_units=args.hidden_units,
        nu=args.nu,
        lambda_data=args.lambda_data,
        lambda_pde=args.lambda_pde,
        checkpoint_interval=args.checkpoint_interval,
        quick_test=args.quick_test
    )


if __name__ == '__main__':
    main()
