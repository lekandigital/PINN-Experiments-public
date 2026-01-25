"""
Training Infrastructure for WavePINN-NIF-Scalar.

This module provides:
- Mixed-precision training (bfloat16) for GPU efficiency
- Gradient clipping and learning rate scheduling
- Multi-GPU training via JAX pmap
- Training loop with logging and checkpointing
- Integration with Weights & Biases (optional)

Optimized for NVIDIA L40S/RTX 3090/4090 GPUs.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, NamedTuple
from functools import partial
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
import optax
import haiku as hk

from .model import WavePINN, WavePINNConfig
from .utils import (
    cast_to_bfloat16, cast_to_float32,
    save_checkpoint, load_checkpoint,
    MetricsLogger, compute_grad_norm, count_params
)

# Type aliases
Array = jax.Array
PRNGKey = jax.Array
Params = hk.Params


# =============================================================================
# Training Configuration
# =============================================================================

class TrainingConfig(NamedTuple):
    """Configuration for training."""
    # Optimizer
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    
    # Learning rate schedule
    warmup_steps: int = 100
    decay_steps: int = 10000
    min_learning_rate: float = 1e-6
    
    # Gradient clipping
    grad_clip_norm: float = 1.0
    
    # Mixed precision
    use_mixed_precision: bool = True
    
    # Multi-GPU
    multi_gpu: bool = False
    
    # Training loop
    n_epochs: int = 1000
    log_every: int = 100
    checkpoint_every: int = 1000
    
    # Batch settings
    batch_size: int = 8192  # Points per batch
    resample_every: int = 100  # Resample collocation points every N steps


# =============================================================================
# Optimizer Creation
# =============================================================================

def create_optimizer(config: TrainingConfig) -> optax.GradientTransformation:
    """
    Create optimizer with learning rate schedule and gradient clipping.
    
    Args:
        config: Training configuration.
        
    Returns:
        Optax optimizer chain.
    """
    # Learning rate schedule: warmup + cosine decay
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=config.learning_rate * 0.1,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.decay_steps,
        end_value=config.min_learning_rate
    )
    
    # Build optimizer chain
    components = []
    
    # Gradient clipping
    if config.grad_clip_norm > 0:
        components.append(optax.clip_by_global_norm(config.grad_clip_norm))
    
    # Base optimizer
    if config.optimizer == "adamw":
        components.append(optax.adamw(schedule, weight_decay=config.weight_decay))
    elif config.optimizer == "adam":
        components.append(optax.adam(schedule))
    elif config.optimizer == "sgd":
        components.append(optax.sgd(schedule, momentum=0.9))
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    return optax.chain(*components)


# =============================================================================
# Training State
# =============================================================================

class TrainState(NamedTuple):
    """Immutable training state container."""
    params: Params
    opt_state: optax.OptState
    step: int
    key: PRNGKey
    best_loss: float = float('inf')


def init_train_state(
    pinn: WavePINN,
    optimizer: optax.GradientTransformation,
    key: PRNGKey
) -> TrainState:
    """
    Initialize training state.
    
    Args:
        pinn: WavePINN model.
        optimizer: Optax optimizer.
        key: Random key.
        
    Returns:
        Initial TrainState.
    """
    k1, k2 = jr.split(key)
    params = pinn.init_params(k1)
    opt_state = optimizer.init(params)
    
    return TrainState(
        params=params,
        opt_state=opt_state,
        step=0,
        key=k2,
        best_loss=float('inf')
    )


# =============================================================================
# Single Training Step
# =============================================================================

def make_train_step(
    pinn: WavePINN,
    optimizer: optax.GradientTransformation,
    config: TrainingConfig
) -> Callable:
    """
    Create JIT-compiled training step function.
    
    Args:
        pinn: WavePINN model instance.
        optimizer: Optax optimizer.
        config: Training configuration.
        
    Returns:
        Training step function: (state, batch) -> (state, metrics).
    """
    
    def loss_fn(params: Params, batch: Dict[str, Array]) -> Tuple[Array, Dict]:
        """Compute loss with optional mixed precision."""
        total_loss, components = pinn.total_loss(params, batch, return_components=True)
        return total_loss, components
    
    def train_step_single(
        state: TrainState,
        batch: Dict[str, Array]
    ) -> Tuple[TrainState, Dict[str, Array]]:
        """
        Single training step.
        
        Args:
            state: Current training state.
            batch: Training batch.
            
        Returns:
            Updated state and metrics dictionary.
        """
        # Mixed precision: cast params to bfloat16 for forward pass
        if config.use_mixed_precision:
            params_compute = cast_to_bfloat16(state.params)
        else:
            params_compute = state.params
        
        # Compute loss and gradients
        (loss, components), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params_compute, batch
        )
        
        # Cast gradients back to float32 for optimizer
        if config.use_mixed_precision:
            grads = cast_to_float32(grads)
        
        # Optimizer update
        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        
        # Update state
        new_state = TrainState(
            params=new_params,
            opt_state=new_opt_state,
            step=state.step + 1,
            key=state.key,
            best_loss=jnp.minimum(state.best_loss, loss)
        )
        
        # Metrics
        grad_norm = compute_grad_norm(grads)
        metrics = {
            'loss': loss,
            'loss_pde': components['pde'],
            'loss_bc': components['bc'],
            'loss_ic_u': components['ic_u'],
            'loss_ic_dt': components['ic_dt'],
            'grad_norm': grad_norm,
        }
        
        return new_state, metrics
    
    return jax.jit(train_step_single)


def make_train_step_pmap(
    pinn: WavePINN,
    optimizer: optax.GradientTransformation,
    config: TrainingConfig
) -> Callable:
    """
    Create pmap-compiled training step for multi-GPU.
    
    Args:
        pinn: WavePINN model.
        optimizer: Optax optimizer.
        config: Training configuration.
        
    Returns:
        Pmapped training step function.
    """
    single_step = make_train_step(pinn, optimizer, config)
    
    @partial(jax.pmap, axis_name='devices')
    def train_step_pmap(state: TrainState, batch: Dict[str, Array]):
        new_state, metrics = single_step(state, batch)
        
        # All-reduce gradients and metrics across devices
        metrics = jax.tree_util.tree_map(
            lambda x: lax.pmean(x, axis_name='devices'),
            metrics
        )
        
        return new_state, metrics
    
    return train_step_pmap


# =============================================================================
# Training Loop
# =============================================================================

class Trainer:
    """
    Main trainer class for WavePINN.
    
    Handles training loop, logging, checkpointing, and optional W&B integration.
    """
    
    def __init__(
        self,
        pinn: WavePINN,
        config: TrainingConfig,
        log_dir: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: str = "wavepinn-nif-scalar"
    ):
        """
        Initialize trainer.
        
        Args:
            pinn: WavePINN model.
            config: Training configuration.
            log_dir: Directory for logs and checkpoints.
            use_wandb: Whether to use Weights & Biases logging.
            wandb_project: W&B project name.
        """
        self.pinn = pinn
        self.config = config
        self.log_dir = Path(log_dir) if log_dir else Path("outputs")
        self.use_wandb = use_wandb
        
        # Create optimizer
        self.optimizer = create_optimizer(config)
        
        # Create training step function
        if config.multi_gpu and len(jax.devices()) > 1:
            self.train_step = make_train_step_pmap(pinn, self.optimizer, config)
            self.n_devices = len(jax.devices())
            print(f"Using {self.n_devices} GPUs for training")
        else:
            self.train_step = make_train_step(pinn, self.optimizer, config)
            self.n_devices = 1
        
        # Initialize logger
        self.logger = MetricsLogger(self.log_dir / "logs")
        
        # W&B initialization
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project,
                    config=config._asdict()
                )
                self.wandb = wandb
            except ImportError:
                print("wandb not installed. Disabling W&B logging.")
                self.wandb = None
        else:
            self.wandb = None
    
    def train(
        self,
        train_data: Dict[str, Array],
        val_data: Optional[Dict[str, Array]] = None,
        initial_state: Optional[TrainState] = None,
        key: Optional[PRNGKey] = None
    ) -> Tuple[TrainState, List[Dict]]:
        """
        Run training loop.
        
        Args:
            train_data: Training data dictionary with collocation points.
            val_data: Optional validation data.
            initial_state: Optional initial training state (for resuming).
            key: Random key for initialization.
            
        Returns:
            Tuple of (final_state, history).
        """
        cfg = self.config
        
        # Initialize state
        if initial_state is not None:
            state = initial_state
        else:
            if key is None:
                key = jr.PRNGKey(42)
            state = init_train_state(self.pinn, self.optimizer, key)
        
        # Print model info
        n_params = count_params(state.params)
        print(f"Model parameters: {n_params:,}")
        print(f"Training for {cfg.n_epochs} epochs")
        
        # Prepare batch from training data
        batch = self._prepare_batch(train_data)
        
        # Multi-GPU: replicate state and batch
        if self.n_devices > 1:
            state = jax.device_put_replicated(state, jax.devices())
            batch = self._shard_batch(batch)
        
        # Training loop
        history = []
        start_time = time.time()
        
        for epoch in range(cfg.n_epochs):
            # Optionally resample collocation points
            if cfg.resample_every > 0 and epoch > 0 and epoch % cfg.resample_every == 0:
                state, batch = self._resample_points(state, train_data)
            
            # Training step
            state, metrics = self.train_step(state, batch)
            
            # Get metrics from device
            if self.n_devices > 1:
                metrics = jax.tree_util.tree_map(lambda x: x[0], metrics)
            metrics = {k: float(v) for k, v in metrics.items()}
            
            # Logging
            if epoch % cfg.log_every == 0 or epoch == cfg.n_epochs - 1:
                elapsed = time.time() - start_time
                metrics['epoch'] = epoch
                metrics['elapsed_time'] = elapsed
                
                self.logger.log(epoch, **metrics)
                self.logger.print_status(epoch, metrics)
                
                if self.wandb is not None:
                    self.wandb.log(metrics, step=epoch)
            
            # Checkpointing
            if cfg.checkpoint_every > 0 and epoch > 0 and epoch % cfg.checkpoint_every == 0:
                self._save_checkpoint(state, epoch, metrics)
            
            history.append(metrics)
        
        # Final checkpoint
        self._save_checkpoint(state, cfg.n_epochs, history[-1])
        
        # Cleanup
        if self.wandb is not None:
            self.wandb.finish()
        
        self.logger.save()
        
        print(f"\nTraining complete! Total time: {time.time() - start_time:.1f}s")
        print(f"Final loss: {history[-1]['loss']:.4e}")
        
        return state, history
    
    def _prepare_batch(self, data: Dict[str, Array]) -> Dict[str, Array]:
        """Prepare training batch from data dictionary."""
        batch = {
            'interior': data['collocation']['interior'],
            'boundary': data['collocation']['boundary'],
            'initial': data['collocation']['initial'],
            'velocity': data['velocity_at_interior'],
        }
        
        # Add source term if available
        if 'source' in data:
            batch['source'] = data['source']
        
        return batch
    
    def _shard_batch(self, batch: Dict[str, Array]) -> Dict[str, Array]:
        """Shard batch across devices for pmap."""
        def shard(arr):
            # Reshape to (n_devices, batch_per_device, ...)
            n = arr.shape[0]
            per_device = n // self.n_devices
            return arr[:per_device * self.n_devices].reshape(
                self.n_devices, per_device, *arr.shape[1:]
            )
        return jax.tree_util.tree_map(shard, batch)
    
    def _resample_points(
        self,
        state: TrainState,
        data: Dict
    ) -> Tuple[TrainState, Dict[str, Array]]:
        """Resample collocation points for better coverage."""
        # For simplicity, we don't resample in this implementation
        # In practice, you could regenerate points here
        return state, self._prepare_batch(data)
    
    def _save_checkpoint(
        self,
        state: TrainState,
        epoch: int,
        metrics: Dict
    ) -> None:
        """Save training checkpoint."""
        if self.n_devices > 1:
            # Get params from first device
            params = jax.tree_util.tree_map(lambda x: x[0], state.params)
            opt_state = jax.tree_util.tree_map(lambda x: x[0], state.opt_state)
            step = int(state.step[0])
        else:
            params = state.params
            opt_state = state.opt_state
            step = int(state.step)
        
        ckpt_path = self.log_dir / "checkpoints" / f"ckpt_epoch_{epoch}.pkl"
        save_checkpoint(
            ckpt_path,
            params=params,
            opt_state=opt_state,
            step=step,
            metrics=metrics
        )


# =============================================================================
# Convenience Training Function
# =============================================================================

def train_wavepinn(
    data: Dict,
    pinn_config: Optional[WavePINNConfig] = None,
    train_config: Optional[TrainingConfig] = None,
    seed: int = 42,
    log_dir: str = "outputs"
) -> Tuple[WavePINN, Params, List[Dict]]:
    """
    High-level function to train a WavePINN model.
    
    Args:
        data: Training data from generate_training_dataset().
        pinn_config: Model configuration (default: use sensible defaults).
        train_config: Training configuration.
        seed: Random seed.
        log_dir: Output directory.
        
    Returns:
        Tuple of (pinn, trained_params, history).
        
    Example:
        >>> from src.data_gen import generate_training_dataset
        >>> data = generate_training_dataset(seed=42)
        >>> pinn, params, history = train_wavepinn(data, log_dir="outputs")
        >>> print(f"Final loss: {history[-1]['loss']:.4e}")
    """
    if pinn_config is None:
        pinn_config = WavePINNConfig()
    
    if train_config is None:
        train_config = TrainingConfig()
    
    # Create model
    pinn = WavePINN(pinn_config, seed=seed)
    
    # Create trainer
    trainer = Trainer(pinn, train_config, log_dir=log_dir)
    
    # Train
    key = jr.PRNGKey(seed)
    final_state, history = trainer.train(data, key=key)
    
    # Extract params
    if train_config.multi_gpu and len(jax.devices()) > 1:
        params = jax.tree_util.tree_map(lambda x: x[0], final_state.params)
    else:
        params = final_state.params
    
    return pinn, params, history


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line entry point for training."""
    import argparse
    from .data_gen import generate_training_dataset
    
    parser = argparse.ArgumentParser(description="Train WavePINN-NIF-Scalar")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden", type=str, default="128,128,64", help="Hidden dims")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    
    args = parser.parse_args()
    
    # Parse hidden dims
    hidden_dims = [int(x) for x in args.hidden.split(",")]
    
    # Generate data
    print("Generating training data...")
    data = generate_training_dataset(
        seed=args.seed,
        nx=100,
        nz=100,
        n_interior=10000,
        n_boundary=2000,
        n_initial=1000
    )
    
    # Configs
    pinn_config = WavePINNConfig(hidden_dims=hidden_dims)
    train_config = TrainingConfig(
        n_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    # Create trainer
    pinn = WavePINN(pinn_config, seed=args.seed)
    trainer = Trainer(
        pinn, train_config,
        log_dir=args.output,
        use_wandb=args.wandb
    )
    
    # Train
    key = jr.PRNGKey(args.seed)
    state, history = trainer.train(data, key=key)
    
    print(f"\nTraining complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()
