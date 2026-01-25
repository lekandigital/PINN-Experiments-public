"""
SurfPINN Training Script
========================
Mixed-precision training loop for the SurfPINN dual-branch model.
Optimized for NVIDIA GPUs (L40S 48GB / RTX 3090 24GB).

Features:
- Mixed precision (bfloat16) for memory efficiency
- Gradient clipping for stability
- Cosine learning rate schedule
- Checkpointing every N epochs
- TensorBoard logging
"""

import os
import time
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, NamedTuple
from functools import partial

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np

from model import SurfPINN, SurfPINNConfig, create_model, init_model
from physics import total_physics_loss, PhysicsConfig, compute_psnr, compute_rmse
from data_gen import generate_dataset, load_dataset


# ==============================================================================
# Training Configuration
# ==============================================================================

class TrainConfig(NamedTuple):
    """Training hyperparameters."""
    # Optimization
    learning_rate: float = 1e-4
    warmup_epochs: int = 5
    total_epochs: int = 100
    batch_size: int = 8
    
    # Regularization
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # Mixed precision
    use_mixed_precision: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 10
    
    # Data
    data_path: str = "data/synthetic_dam_break.h5"
    
    # Logging
    log_every: int = 10


class TrainState(NamedTuple):
    """Training state container."""
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState
    step: int
    epoch: int
    rng: jax.random.PRNGKey


# ==============================================================================
# Learning Rate Schedule
# ==============================================================================

def create_lr_schedule(
    base_lr: float,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int
) -> optax.Schedule:
    """
    Create learning rate schedule with warmup and cosine decay.
    
    Args:
        base_lr: Base learning rate after warmup
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        steps_per_epoch: Steps per epoch
    
    Returns:
        Optax schedule function
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=warmup_steps
    )
    
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=total_steps - warmup_steps
    )
    
    return optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_steps]
    )


def create_optimizer(config: TrainConfig, steps_per_epoch: int) -> optax.GradientTransformation:
    """
    Create AdamW optimizer with gradient clipping and learning rate schedule.
    
    Args:
        config: Training configuration
        steps_per_epoch: Number of steps per epoch
    
    Returns:
        Optax optimizer chain
    """
    lr_schedule = create_lr_schedule(
        config.learning_rate,
        config.warmup_epochs,
        config.total_epochs,
        steps_per_epoch
    )
    
    return optax.chain(
        optax.clip_by_global_norm(config.gradient_clip),
        optax.adamw(learning_rate=lr_schedule, weight_decay=config.weight_decay)
    )


# ==============================================================================
# Loss Function
# ==============================================================================

def compute_loss(
    params: hk.Params,
    state: hk.State,
    rng: jax.random.PRNGKey,
    model: hk.TransformedWithState,
    batch: Dict[str, jnp.ndarray],
    physics_config: PhysicsConfig,
    is_training: bool = True
) -> Tuple[jnp.ndarray, Tuple[hk.State, Dict[str, jnp.ndarray]]]:
    """
    Compute total loss for a batch.
    
    Args:
        params: Model parameters
        state: Model state
        rng: Random key
        model: Transformed Haiku model
        batch: Dictionary with training data
        physics_config: Physics loss configuration
        is_training: Training mode flag
    
    Returns:
        Tuple of (total_loss, (new_state, loss_dict))
    """
    # Extract batch data
    grid_input = batch['grid_input']      # (batch, Nx, Ny, C)
    particle_pos = batch['particle_pos']  # (batch, N_particles, 3)
    height_true = batch['height_true']    # (batch, Nx, Ny, 1)
    velocity_true = batch['velocity_true']  # (batch, N_particles, 3)
    
    # Forward pass
    (height_pred, velocity_pred, z_eul, z_lag), new_state = model.apply(
        params, state, rng, grid_input, particle_pos, is_training
    )
    
    # Compute physics-informed loss
    losses = total_physics_loss(
        height_pred=height_pred,
        velocity_pred=velocity_pred,
        height_true=height_true,
        velocity_true=velocity_true,
        config=physics_config,
        dx=1.0 / height_pred.shape[1],
        dy=1.0 / height_pred.shape[2]
    )
    
    # Add latent regularization (encourage compact representations)
    latent_reg = 0.001 * (jnp.mean(z_eul ** 2) + jnp.mean(z_lag ** 2))
    losses['latent_reg'] = latent_reg
    losses['total'] = losses['total'] + latent_reg
    
    return losses['total'], (new_state, losses)


# ==============================================================================
# Training Step Factory
# ==============================================================================

def make_train_step(model, optimizer, physics_config):
    """
    Create a JIT-compiled training step function.
    
    Args:
        model: Haiku transformed model
        optimizer: Optax optimizer
        physics_config: Physics configuration
    
    Returns:
        JIT-compiled train_step function
    """
    
    def loss_fn(params, state, rng, batch):
        """Compute total loss."""
        grid_input = batch['grid_input']
        particle_pos = batch['particle_pos']
        height_true = batch['height_true']
        velocity_true = batch['velocity_true']
        
        # Forward pass
        (height_pred, velocity_pred, z_eul, z_lag), new_state = model.apply(
            params, state, rng, grid_input, particle_pos, True
        )
        
        # Compute physics-informed loss
        losses = total_physics_loss(
            height_pred=height_pred,
            velocity_pred=velocity_pred,
            height_true=height_true,
            velocity_true=velocity_true,
            config=physics_config,
            dx=1.0 / height_pred.shape[1],
            dy=1.0 / height_pred.shape[2]
        )
        
        # Add latent regularization
        latent_reg = 0.001 * (jnp.mean(z_eul ** 2) + jnp.mean(z_lag ** 2))
        losses['latent_reg'] = latent_reg
        losses['total'] = losses['total'] + latent_reg
        
        return losses['total'], (new_state, losses)
    
    @jax.jit
    def train_step(params, state, opt_state, rng, batch):
        """Single training step."""
        rng, step_rng = jax.random.split(rng)
        
        # Compute gradients
        (loss, (new_state, losses)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params, state, step_rng, batch)
        
        # Apply gradients
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        # Compute grad norm
        grad_norm = optax.global_norm(grads)
        
        return new_params, new_state, new_opt_state, rng, losses, grad_norm
    
    return train_step


def make_eval_step(model):
    """Create JIT-compiled evaluation step."""
    
    @jax.jit
    def eval_step(params, state, batch):
        """Evaluation step."""
        grid_input = batch['grid_input']
        particle_pos = batch['particle_pos']
        height_true = batch['height_true']
        velocity_true = batch['velocity_true']
        
        # Forward pass (eval mode)
        (height_pred, velocity_pred, _, _), _ = model.apply(
            params, state, None, grid_input, particle_pos, False
        )
        
        # Compute metrics
        psnr = compute_psnr(height_pred, height_true)
        rmse_height = compute_rmse(height_pred, height_true)
        rmse_velocity = compute_rmse(velocity_pred, velocity_true)
        
        return {
            'psnr': psnr,
            'rmse_height': rmse_height,
            'rmse_velocity': rmse_velocity
        }
    
    return eval_step


# ==============================================================================
# Data Loading
# ==============================================================================

def prepare_batches(
    data: Dict,
    batch_size: int,
    n_time_samples: int = 4,
    rng: Optional[jax.random.PRNGKey] = None
) -> list:
    """
    Prepare training batches from dataset.
    
    Args:
        data: Dataset dictionary
        batch_size: Batch size
        n_time_samples: Number of time steps per sample
        rng: Random key for shuffling
    
    Returns:
        List of batch dictionaries
    """
    eulerian = data['eulerian']
    lagrangian = data['lagrangian']
    
    grid_coords = eulerian['grid_coords']  # (Nx, Ny, Nt, 3)
    height = eulerian['height']            # (Nx, Ny, Nt, 1)
    velocity_grid = eulerian['velocity']   # (Nx, Ny, Nt, 2)
    
    positions = lagrangian['positions']    # (N_particles, Nt, 3)
    velocities = lagrangian['velocities']  # (N_particles, Nt, 3)
    
    Nx, Ny, Nt, _ = grid_coords.shape
    N_particles = positions.shape[0]
    
    batches = []
    
    # Sample time steps
    for t in range(0, Nt - n_time_samples + 1, n_time_samples):
        # Create batch for this time window
        batch = {
            'grid_input': np.concatenate([
                grid_coords[:, :, t, :],
                height[:, :, t, :]
            ], axis=-1).astype(np.float32),  # (Nx, Ny, 4)
            'height_true': height[:, :, t, :].astype(np.float32),  # (Nx, Ny, 1)
            'particle_pos': positions[:, t, :].astype(np.float32),  # (N_particles, 3)
            'velocity_true': velocities[:, t, :].astype(np.float32)  # (N_particles, 3)
        }
        
        # Add batch dimension
        batch = {k: v[None, ...] for k, v in batch.items()}
        
        batches.append(batch)
    
    # Pad to batch_size by repeating
    while len(batches) < batch_size:
        batches.extend(batches[:batch_size - len(batches)])
    
    return batches[:max(len(batches), 1)]


# ==============================================================================
# Checkpointing
# ==============================================================================

def save_checkpoint(train_state: TrainState, path: str):
    """Save training state to file."""
    import pickle
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy for serialization
    state_dict = {
        'params': jax.tree_util.tree_map(np.array, train_state.params),
        'state': jax.tree_util.tree_map(np.array, train_state.state),
        'step': train_state.step,
        'epoch': train_state.epoch
    }
    
    with open(path, 'wb') as f:
        pickle.dump(state_dict, f)
    
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path: str, train_state: TrainState) -> TrainState:
    """Load training state from file."""
    import pickle
    
    with open(path, 'rb') as f:
        state_dict = pickle.load(f)
    
    return TrainState(
        params=jax.tree_util.tree_map(jnp.array, state_dict['params']),
        state=jax.tree_util.tree_map(jnp.array, state_dict['state']),
        opt_state=train_state.opt_state,
        step=state_dict['step'],
        epoch=state_dict['epoch'],
        rng=train_state.rng
    )


# ==============================================================================
# Main Training Loop
# ==============================================================================

def train(config: TrainConfig, model_config: SurfPINNConfig = None):
    """
    Main training function.
    
    Args:
        config: Training configuration
        model_config: Model architecture configuration
    """
    print("=" * 60)
    print("SurfPINN Training")
    print("=" * 60)
    
    # Check for GPU
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    if any('gpu' in str(d).lower() for d in devices):
        print("GPU detected!")
    else:
        print("Warning: No GPU detected, training will be slow")
    
    # Enable mixed precision
    if config.use_mixed_precision:
        jax.config.update("jax_default_matmul_precision", "bfloat16")
        print("Mixed precision enabled (bfloat16)")
    
    # Generate or load data
    if not Path(config.data_path).exists():
        print(f"\nGenerating synthetic dataset...")
        generate_dataset(config.data_path)
    
    print(f"\nLoading data from {config.data_path}")
    data = load_dataset(config.data_path)
    print(f"  Eulerian grid: {data['eulerian']['height'].shape}")
    print(f"  Lagrangian particles: {data['lagrangian']['positions'].shape}")
    
    # Prepare batches
    batches = prepare_batches(data, config.batch_size)
    steps_per_epoch = len(batches)
    print(f"  Steps per epoch: {steps_per_epoch}")
    
    # Initialize model
    model_config = model_config or SurfPINNConfig()
    model = create_model(model_config)
    
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    # Get shapes from data
    sample_batch = batches[0]
    grid_shape = sample_batch['grid_input'].shape[1:]  # (Nx, Ny, C)
    n_particles = sample_batch['particle_pos'].shape[1]
    
    print(f"\nInitializing model...")
    print(f"  Grid input shape: {grid_shape}")
    print(f"  Particles: {n_particles}")
    
    params, state = model.init(
        init_rng,
        jnp.zeros((1,) + grid_shape),
        jnp.zeros((1, n_particles, 3)),
        True
    )
    
    # Count parameters
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Total parameters: {n_params:,}")
    
    # Create optimizer
    optimizer = create_optimizer(config, steps_per_epoch)
    opt_state = optimizer.init(params)
    
    # Physics config
    physics_config = PhysicsConfig()
    
    # Create JIT-compiled training and eval functions
    train_step = make_train_step(model, optimizer, physics_config)
    eval_step = make_eval_step(model)
    
    # Training loop
    print(f"\nStarting training for {config.total_epochs} epochs...")
    print("-" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(config.total_epochs):
        epoch_start = time.time()
        epoch_losses = []
        
        for batch in batches:
            # Convert to JAX arrays
            batch_jax = {k: jnp.array(v) for k, v in batch.items()}
            
            # Training step
            params, state, opt_state, rng, losses, grad_norm = train_step(
                params, state, opt_state, rng, batch_jax
            )
            
            # Collect metrics
            metrics = {k: float(v) for k, v in losses.items()}
            metrics['grad_norm'] = float(grad_norm)
            epoch_losses.append(metrics)
        
        # Aggregate epoch metrics
        avg_loss = np.mean([m['total'] for m in epoch_losses])
        avg_height_mse = np.mean([m['height_mse'] for m in epoch_losses])
        avg_curv = np.mean([m['curvature'] for m in epoch_losses])
        avg_grad = np.mean([m['grad_norm'] for m in epoch_losses])
        
        epoch_time = time.time() - epoch_start
        
        # Logging
        if (epoch + 1) % config.log_every == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d} | Loss: {avg_loss:.6f} | "
                  f"H_MSE: {avg_height_mse:.6f} | Curv: {avg_curv:.6f} | "
                  f"Grad: {avg_grad:.4f} | Time: {epoch_time:.2f}s")
        
        # Checkpointing
        if (epoch + 1) % config.checkpoint_every == 0:
            train_state = TrainState(params=params, state=state, opt_state=opt_state,
                                    step=(epoch+1)*steps_per_epoch, epoch=epoch+1, rng=rng)
            ckpt_path = f"{config.checkpoint_dir}/surfpinn_epoch_{epoch+1}.pkl"
            save_checkpoint(train_state, ckpt_path)
        
        # Track best
        if avg_loss < best_loss:
            best_loss = avg_loss
            train_state = TrainState(params=params, state=state, opt_state=opt_state,
                                    step=(epoch+1)*steps_per_epoch, epoch=epoch+1, rng=rng)
            save_checkpoint(train_state, f"{config.checkpoint_dir}/surfpinn_best.pkl")
    
    print("-" * 60)
    print(f"Training complete! Best loss: {best_loss:.6f}")
    
    # Final evaluation
    print("\nFinal evaluation...")
    eval_batch = {k: jnp.array(v) for k, v in batches[0].items()}
    eval_metrics = eval_step(params, state, eval_batch)
    print(f"  PSNR: {float(eval_metrics['psnr']):.2f} dB")
    print(f"  Height RMSE: {float(eval_metrics['rmse_height']):.6f}")
    print(f"  Velocity RMSE: {float(eval_metrics['rmse_velocity']):.6f}")
    
    # Return final state
    final_state = TrainState(params=params, state=state, opt_state=opt_state,
                            step=config.total_epochs*steps_per_epoch, 
                            epoch=config.total_epochs, rng=rng)
    return final_state


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train SurfPINN model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--data", type=str, default="data/synthetic_dam_break.h5",
                       help="Path to dataset")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Checkpoint directory")
    parser.add_argument("--test-mode", action="store_true",
                       help="Run quick test (10 epochs)")
    parser.add_argument("--no-mixed-precision", action="store_true",
                       help="Disable mixed precision")
    
    args = parser.parse_args()
    
    config = TrainConfig(
        total_epochs=10 if args.test_mode else args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_path=args.data,
        checkpoint_dir=args.checkpoint_dir,
        use_mixed_precision=not args.no_mixed_precision,
        log_every=1 if args.test_mode else 10
    )
    
    train(config)


if __name__ == "__main__":
    main()
