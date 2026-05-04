"""
Training orchestrator for WavePINN-NIF.

Implements:
- Training loop with mixed precision (optional)
- Learning rate scheduling
- Gradient clipping
- Checkpointing
- Weights & Biases logging
"""

import jax
import jax.numpy as jnp
import optax
import pickle
import os
import time
from typing import Dict, Any, Optional, Callable, Tuple
from pathlib import Path
from tqdm import tqdm
from functools import partial

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .model import create_model, count_parameters
from .physics_loss import WavePDELoss
from .data_generator import SyntheticWaveData, create_data_batch


class WavePINNTrainer:
    """
    Main training orchestrator for WavePINN-NIF.
    
    Handles:
    - Model initialization
    - Optimizer setup
    - Training loop
    - Checkpointing
    - Logging
    """
    
    def __init__(self,
                 config: Dict[str, Any],
                 model: Optional[Any] = None,
                 use_wandb: bool = True):
        """
        Initialize trainer.
        
        Args:
            config: Full configuration dictionary
            model: Optional pre-created model (defaults to create_model())
            use_wandb: Whether to use Weights & Biases logging
        """
        self.config = config
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        # Extract sub-configs
        self.training_config = config.get('training', {})
        self.model_config = config.get('model', {})
        self.loss_config = config.get('loss_weights', {})
        self.domain_config = config.get('domain', {})
        
        # Set random seed
        self.seed = config.get('seed', 42)
        self.rng = jax.random.PRNGKey(self.seed)
        
        # Create model
        if model is None:
            model_cfg = {
                'wave': self.model_config.get('wave_net', {}),
                'media': self.model_config.get('media_net', {})
            }
            self.model = create_model(model_cfg)
        else:
            self.model = model
        
        # Determine input dimension
        ndim = self.domain_config.get('spatial_dims', 2)
        self.input_dim = ndim + 1  # spatial dims + time
        
        # Initialize parameters
        self.rng, init_rng = jax.random.split(self.rng)
        dummy_input = jnp.zeros((1, self.input_dim))
        self.params = self.model.init(init_rng, dummy_input, return_media=True)
        
        print(f"Model initialized with {count_parameters(self.params):,} parameters")
        
        # Create optimizer with optional gradient clipping
        self.optimizer = self._create_optimizer()
        self.opt_state = self.optimizer.init(self.params)
        
        # Create loss computer
        self.loss_computer = WavePDELoss(
            model_apply=self.model.apply,
            lambda_pde=self.loss_config.get('lambda_pde', 1.0),
            lambda_bc=self.loss_config.get('lambda_bc', 10.0),
            lambda_ic=self.loss_config.get('lambda_ic', 10.0),
            lambda_data=self.loss_config.get('lambda_data', 1.0),
            lambda_medium=self.loss_config.get('lambda_medium', 0.0),
            source_config=self.loss_config.get('source_config', None),
            ndim=ndim
        )
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.loss_history = []
        
        # Paths
        self.checkpoint_dir = Path(config.get('paths', {}).get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize W&B
        if self.use_wandb:
            self._init_wandb()
    
    def _create_optimizer(self) -> optax.GradientTransformation:
        """Create optimizer with learning rate schedule and gradient clipping."""
        lr = self.training_config.get('learning_rate', 1e-3)
        grad_clip = self.training_config.get('gradient_clip', 1.0)
        
        # Learning rate schedule
        lr_config = self.training_config.get('lr_scheduler', {})
        schedule_type = lr_config.get('type', 'constant')
        
        if schedule_type == 'exponential':
            decay_rate = lr_config.get('decay_rate', 0.99)
            decay_steps = lr_config.get('decay_steps', 100)
            scheduler = optax.exponential_decay(
                init_value=lr,
                transition_steps=decay_steps,
                decay_rate=decay_rate
            )
        elif schedule_type == 'cosine':
            n_epochs = self.training_config.get('n_epochs', 5000)
            scheduler = optax.cosine_decay_schedule(
                init_value=lr,
                decay_steps=n_epochs
            )
        else:
            scheduler = optax.constant_schedule(lr)
        
        # Build optimizer chain
        optimizer = optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optax.scale_by_adam(),
            optax.scale_by_schedule(scheduler),
            optax.scale(-1.0)  # Gradient descent
        )
        
        return optimizer
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        project = self.config.get('logging', {}).get('wandb_project', 'wavepinn-nif')
        entity = self.config.get('logging', {}).get('wandb_entity', None)
        
        wandb.init(
            project=project,
            entity=entity,
            config=self.config,
            name=self.config.get('experiment_name', 'wavepinn_run')
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _train_step(self,
                    params: Dict,
                    opt_state: Any,
                    rng: jax.Array,
                    data_batch: Dict) -> Tuple[Dict, Any, Dict, jnp.ndarray]:
        """
        Single training step (JIT-compiled).
        
        Args:
            params: Model parameters
            opt_state: Optimizer state
            rng: Random key
            data_batch: Batch of training data
            
        Returns:
            Tuple of (new_params, new_opt_state, loss_dict, grad_norm)
        """
        def loss_fn(p):
            return self.loss_computer.total_loss(p, rng, data_batch)
        
        (total_loss, loss_dict), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        
        # Compute gradient norm for logging
        grad_norm = jnp.sqrt(
            sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads))
        )
        
        # Apply updates
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss_dict, grad_norm
    
    def train_step(self, data_batch: Dict) -> Dict[str, float]:
        """
        Execute single training step.
        
        Args:
            data_batch: Batch of training data
            
        Returns:
            Dictionary of loss values
        """
        self.rng, step_rng = jax.random.split(self.rng)
        
        self.params, self.opt_state, loss_dict, grad_norm = self._train_step(
            self.params, self.opt_state, step_rng, data_batch
        )
        
        self.step += 1
        
        # Convert to Python floats for logging
        metrics = {k: float(v) for k, v in loss_dict.items()}
        metrics['grad_norm'] = float(grad_norm)
        metrics['step'] = self.step
        
        return metrics
    
    def train_epoch(self,
                    data: Dict,
                    batch_size: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            data: Full dataset
            batch_size: Batch size
            
        Returns:
            Average metrics for the epoch
        """
        n_samples = len(data['interior'])
        n_batches = max(1, n_samples // batch_size)
        
        epoch_metrics = {}
        
        for batch_idx in range(n_batches):
            self.rng, batch_rng = jax.random.split(self.rng)
            batch = create_data_batch(data, batch_size, batch_rng)
            
            # Add supervised wavefield labels when the repaired contract provides them.
            if 'x_data' in data and 'u_data' in data:
                self.rng, data_rng = jax.random.split(self.rng)
                data_size = len(data['x_data'])
                data_idx = jax.random.randint(data_rng, (batch_size,), 0, data_size)
                batch['x_data'] = jnp.asarray(data['x_data'])[data_idx]
                batch['u_data'] = jnp.asarray(data['u_data'])[data_idx]

            # Add directly supervised medium labels when known c(x,y) is provided.
            if 'x_media' in data and 'c_data' in data:
                self.rng, media_rng = jax.random.split(self.rng)
                media_size = len(data['x_media'])
                media_idx = jax.random.randint(media_rng, (batch_size,), 0, media_size)
                batch['x_media'] = jnp.asarray(data['x_media'])[media_idx]
                batch['c_data'] = jnp.asarray(data['c_data'])[media_idx]

            # Initial-condition targets are no longer unconditionally hardcoded.
            # Zero is only a fallback for legacy generated datasets.
            if 'u0_target' not in batch:
                batch['u0_target'] = jnp.zeros(len(batch['initial']))
            if 'v0_target' not in batch:
                batch['v0_target'] = jnp.zeros(len(batch['initial']))
            
            metrics = self.train_step(batch)
            
            # Accumulate metrics
            for k, v in metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = []
                epoch_metrics[k].append(v)
        
        # Average metrics
        avg_metrics = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
        avg_metrics['epoch'] = self.epoch
        
        return avg_metrics
    
    def train(self,
              data: Dict,
              n_epochs: Optional[int] = None) -> Dict:
        """
        Full training loop.
        
        Args:
            data: Training data dictionary
            n_epochs: Number of epochs (defaults to config)
            
        Returns:
            Final parameters
        """
        if n_epochs is None:
            n_epochs = self.training_config.get('n_epochs', 5000)
        
        batch_size = self.training_config.get('batch_size', 1024)
        checkpoint_freq = self.training_config.get('checkpoint_freq', 500)
        log_freq = self.config.get('logging', {}).get('log_freq', 10)
        
        print(f"\nStarting training for {n_epochs} epochs...")
        print(f"  Batch size: {batch_size}")
        print(f"  Checkpoint frequency: {checkpoint_freq}")
        print("-" * 60)
        
        start_time = time.time()
        
        for epoch in tqdm(range(n_epochs), desc="Training"):
            self.epoch = epoch
            
            # Train one epoch
            metrics = self.train_epoch(data, batch_size)
            
            # Track best loss
            if metrics['loss_total'] < self.best_loss:
                self.best_loss = metrics['loss_total']
                self.save_checkpoint('best')
            
            self.loss_history.append(metrics)
            
            # Logging
            if epoch % log_freq == 0:
                if self.use_wandb:
                    wandb.log(metrics)
                
                # Print progress
                if epoch % (log_freq * 10) == 0:
                    elapsed = time.time() - start_time
                    print(f"\nEpoch {epoch}/{n_epochs} ({elapsed:.1f}s)")
                    print(f"  Total loss: {metrics['loss_total']:.6f}")
                    print(f"  PDE loss: {metrics['loss_pde']:.6f}")
                    print(f"  BC loss: {metrics['loss_bc']:.6f}")
                    print(f"  IC loss: {metrics['loss_ic']:.6f}")
                    print(f"  Grad norm: {metrics['grad_norm']:.6f}")
            
            # Checkpointing
            if epoch % checkpoint_freq == 0 and epoch > 0:
                self.save_checkpoint(f'epoch_{epoch}')
        
        # Final checkpoint
        self.save_checkpoint('final')
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f}s")
        print(f"Best loss: {self.best_loss:.6f}")
        
        if self.use_wandb:
            wandb.finish()
        
        return self.params
    
    def save_checkpoint(self, name: str):
        """
        Save model checkpoint.
        
        Args:
            name: Checkpoint name
        """
        checkpoint = {
            'params': self.params,
            'opt_state': self.opt_state,
            'step': self.step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'config': self.config,
            'loss_history': self.loss_history[-100:],  # Keep last 100
        }
        
        path = self.checkpoint_dir / f'{name}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"  Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.params = checkpoint['params']
        self.opt_state = checkpoint['opt_state']
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        if 'loss_history' in checkpoint:
            self.loss_history = checkpoint['loss_history']
        
        print(f"Loaded checkpoint from {path}")
        print(f"  Epoch: {self.epoch}, Step: {self.step}, Best loss: {self.best_loss:.6f}")
    
    def evaluate(self,
                 data: Dict) -> Dict[str, float]:
        """
        Evaluate model on data.
        
        Args:
            data: Evaluation data
            
        Returns:
            Evaluation metrics
        """
        self.rng, eval_rng = jax.random.split(self.rng)
        
        # Add targets
        data = dict(data)
        if 'u0_target' not in data:
            data['u0_target'] = jnp.zeros(len(data.get('initial', [])))
        if 'v0_target' not in data:
            data['v0_target'] = jnp.zeros(len(data.get('initial', [])))
        
        total_loss, loss_dict = self.loss_computer.total_loss(
            self.params, eval_rng, data
        )
        
        return {k: float(v) for k, v in loss_dict.items()}
    
    def predict(self,
                x: jnp.ndarray,
                return_media: bool = False) -> jnp.ndarray:
        """
        Make predictions with trained model.
        
        Args:
            x: Input coordinates (batch, ndim+1)
            return_media: Whether to also return wave speed
            
        Returns:
            Wavefield predictions (and wave speed if requested)
        """
        self.rng, pred_rng = jax.random.split(self.rng)
        
        if return_media:
            return self.model.apply(self.params, pred_rng, x, return_media=True)
        else:
            return self.model.apply(self.params, pred_rng, x)


def create_trainer_from_config(config_path: str) -> WavePINNTrainer:
    """
    Create trainer from YAML configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Initialized trainer
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return WavePINNTrainer(config)


def quick_train(n_epochs: int = 100,
                batch_size: int = 256,
                learning_rate: float = 1e-3,
                seed: int = 42) -> Tuple[WavePINNTrainer, Dict]:
    """
    Quick training function for testing.
    
    Args:
        n_epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        seed: Random seed
        
    Returns:
        Tuple of (trainer, final_metrics)
    """
    # Minimal config
    config = {
        'seed': seed,
        'experiment_name': 'quick_test',
        'domain': {
            'spatial_dims': 2,
            'domain_size': [1.0, 1.0],
            't_max': 1.0,
        },
        'model': {
            'wave_net': {
                'hidden_dims': [64, 64, 64],
                'use_fourier': True,
                'fourier_dim': 64,
                'fourier_sigma': 10.0,
                'activation': 'tanh',
                'use_residual': True,
            },
            'media_net': {
                'hidden_dims': [32, 32],
                'activation': 'softplus',
                'c_min': 1.0,
                'c_max': 5.0,
                'use_fourier': True,
                'fourier_dim': 32,
                'fourier_sigma': 5.0,
            }
        },
        'training': {
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'checkpoint_freq': max(n_epochs // 5, 1),
        },
        'loss_weights': {
            'lambda_pde': 1.0,
            'lambda_bc': 10.0,
            'lambda_ic': 10.0,
        },
        'logging': {
            'use_wandb': False,
            'log_freq': 10,
        },
        'paths': {
            'checkpoint_dir': 'checkpoints',
        }
    }
    
    # Create trainer
    trainer = WavePINNTrainer(config, use_wandb=False)
    
    # Generate data
    data_gen = SyntheticWaveData(
        domain_size=(1.0, 1.0),
        n_collocation=1000,
        n_boundary=200,
        n_initial=200,
        t_max=1.0,
        seed=seed
    )
    data = data_gen.sample_collocation_points()
    
    # Train
    trainer.train(data, n_epochs=n_epochs)
    
    # Final evaluation
    eval_data = {
        'interior': data['interior'][:100],
        'boundary': {k: v[:50] for k, v in data['boundary'].items()},
        'initial': data['initial'][:100],
    }
    final_metrics = trainer.evaluate(eval_data)
    
    return trainer, final_metrics


if __name__ == "__main__":
    print("Testing WavePINN trainer...")
    
    trainer, metrics = quick_train(n_epochs=10, batch_size=64)
    
    print("\nFinal metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    
    print("\n✅ Trainer test completed!")
