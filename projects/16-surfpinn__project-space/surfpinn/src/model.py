"""
SurfPINN Dual-Branch Model
==========================
Physics-informed neural network with:
- Eulerian branch: 2D CNN for height field prediction
- Lagrangian branch: MLP for particle velocity prediction
- Shared latent space for consistency between representations

Architecture based on ELPINN (2025) and DeepLag (2024) approaches.
"""
from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp
from typing import Tuple, Optional, NamedTuple, Union
from functools import partial


class SurfPINNConfig(NamedTuple):
    """Configuration for SurfPINN model."""
    latent_dim: int = 128
    eul_channels: Tuple[int, ...] = (32, 64, 128)
    lag_hidden: Tuple[int, ...] = (64, 64, 128)
    decoder_hidden: Tuple[int, ...] = (64, 64)
    use_skip_connections: bool = True
    dropout_rate: float = 0.0


class EulerianEncoder(hk.Module):
    """
    2D CNN encoder for Eulerian height fields.
    
    Processes spatial grid data (height, velocity) and encodes
    into a latent representation for each grid point.
    """
    
    def __init__(
        self, 
        channels: Tuple[int, ...] = (32, 64, 128),
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.channels = channels
    
    def __call__(self, x: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        """
        Encode Eulerian grid data.
        
        Args:
            x: Input tensor of shape (batch, Nx, Ny, C_in)
               where C_in includes height and possibly velocity channels
            is_training: Whether in training mode (for dropout)
        
        Returns:
            Encoded features of shape (batch, Nx, Ny, latent_dim)
        """
        # Ensure 4D input (batch, H, W, C)
        if x.ndim == 3:
            x = x[None, ...]
        
        # Encoder blocks with residual connections
        for i, ch in enumerate(self.channels):
            # Conv block
            h = hk.Conv2D(ch, kernel_shape=3, stride=1, padding='SAME',
                         name=f'conv_{i}')(x if i == 0 else h)
            h = jax.nn.gelu(h)
            
            # Optional batch norm for stability
            h = hk.BatchNorm(
                create_scale=True, 
                create_offset=True,
                decay_rate=0.9,
                name=f'bn_{i}'
            )(h, is_training=is_training)
            
            # Skip connection if dimensions match
            if i > 0 and h.shape[-1] == x.shape[-1]:
                h = h + x
            x = h
        
        return h


class EulerianDecoder(hk.Module):
    """
    Decoder for Eulerian height field from latent representation.
    
    Outputs scalar height at each grid point.
    """
    
    def __init__(
        self,
        hidden_channels: Tuple[int, ...] = (64, 32),
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.hidden_channels = hidden_channels
    
    def __call__(self, z: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        """
        Decode latent features to height field.
        
        Args:
            z: Latent tensor of shape (batch, Nx, Ny, latent_dim)
            is_training: Training mode flag
        
        Returns:
            Height field of shape (batch, Nx, Ny, 1)
        """
        h = z
        for i, ch in enumerate(self.hidden_channels):
            h = hk.Conv2D(ch, kernel_shape=3, stride=1, padding='SAME',
                         name=f'dec_conv_{i}')(h)
            h = jax.nn.gelu(h)
        
        # Final layer: single channel for height
        height = hk.Conv2D(1, kernel_shape=1, stride=1, padding='SAME',
                          name='height_out')(h)
        
        # Ensure positive height with softplus
        height = jax.nn.softplus(height)
        
        return height


class LagrangianEncoder(hk.Module):
    """
    MLP encoder for Lagrangian particle data.
    
    Processes particle positions and encodes into latent representation.
    """
    
    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (64, 64, 128),
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.hidden_dims = hidden_dims
    
    def __call__(self, x: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        """
        Encode particle positions.
        
        Args:
            x: Particle positions of shape (batch, N_particles, 3) or (N_particles, 3)
            is_training: Training mode flag
        
        Returns:
            Encoded features of shape (..., latent_dim)
        """
        h = x
        for i, dim in enumerate(self.hidden_dims):
            h = hk.Linear(dim, name=f'lag_fc_{i}')(h)
            h = jax.nn.gelu(h)
            
            # Layer norm for stability
            if i < len(self.hidden_dims) - 1:
                h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                               name=f'lag_ln_{i}')(h)
        
        return h


class LagrangianDecoder(hk.Module):
    """
    Decoder for Lagrangian particle velocities.
    
    Outputs 3D velocity vector for each particle.
    """
    
    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (64, 64),
        output_dim: int = 3,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
    
    def __call__(self, z: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        """
        Decode latent to velocity.
        
        Args:
            z: Latent features of shape (..., latent_dim)
            is_training: Training mode flag
        
        Returns:
            Velocity predictions of shape (..., 3)
        """
        h = z
        for i, dim in enumerate(self.hidden_dims):
            h = hk.Linear(dim, name=f'vel_fc_{i}')(h)
            h = jax.nn.gelu(h)
        
        # Output layer: 3D velocity
        velocity = hk.Linear(self.output_dim, name='velocity_out')(h)
        
        return velocity


class SurfPINN(hk.Module):
    """
    SurfPINN: Dual-branch physics-informed neural network.
    
    Jointly predicts:
    - Eulerian height fields on regular grid
    - Lagrangian particle velocities
    
    Both branches share a latent dimension for consistency.
    """
    
    def __init__(self, config: Optional[SurfPINNConfig] = None, name: Optional[str] = None):
        super().__init__(name=name)
        self.config = config or SurfPINNConfig()
        
        # Eulerian branch
        self.eul_encoder = EulerianEncoder(channels=self.config.eul_channels)
        self.eul_decoder = EulerianDecoder(hidden_channels=self.config.decoder_hidden)
        
        # Lagrangian branch  
        self.lag_encoder = LagrangianEncoder(hidden_dims=self.config.lag_hidden)
        self.lag_decoder = LagrangianDecoder(hidden_dims=self.config.decoder_hidden)
    
    def encode_eulerian(self, grid_input: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        """Encode Eulerian grid data to latent space."""
        return self.eul_encoder(grid_input, is_training)
    
    def encode_lagrangian(self, particle_pos: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        """Encode Lagrangian particle positions to latent space."""
        return self.lag_encoder(particle_pos, is_training)
    
    def decode_height(self, z_eul: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        """Decode Eulerian latent to height field."""
        return self.eul_decoder(z_eul, is_training)
    
    def decode_velocity(self, z_lag: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        """Decode Lagrangian latent to particle velocities."""
        return self.lag_decoder(z_lag, is_training)
    
    def __call__(
        self,
        grid_input: jnp.ndarray,
        particle_pos: jnp.ndarray,
        is_training: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Forward pass through both branches.
        
        Args:
            grid_input: Eulerian input (batch, Nx, Ny, C_in)
                       C_in typically includes coordinates or initial height
            particle_pos: Lagrangian positions (batch, N_particles, 3) or (N_particles, 3)
            is_training: Training mode flag
        
        Returns:
            Tuple of:
            - height_pred: Predicted height field (batch, Nx, Ny, 1)
            - velocity_pred: Predicted particle velocities (batch, N_particles, 3)
            - z_eul: Eulerian latent representation
            - z_lag: Lagrangian latent representation
        """
        # Eulerian branch
        z_eul = self.encode_eulerian(grid_input, is_training)
        height_pred = self.decode_height(z_eul, is_training)
        
        # Lagrangian branch
        z_lag = self.encode_lagrangian(particle_pos, is_training)
        velocity_pred = self.decode_velocity(z_lag, is_training)
        
        return height_pred, velocity_pred, z_eul, z_lag


def create_model(config: Optional[SurfPINNConfig] = None):
    """
    Create SurfPINN model as Haiku transformed functions.
    
    Args:
        config: Model configuration
    
    Returns:
        Tuple of (init_fn, apply_fn) for the model
    """
    config = config or SurfPINNConfig()
    
    def forward(grid_input, particle_pos, is_training=True):
        model = SurfPINN(config=config)
        return model(grid_input, particle_pos, is_training)
    
    return hk.transform_with_state(forward)


def init_model(
    rng: jax.random.PRNGKey,
    config: Optional[SurfPINNConfig] = None,
    grid_shape: Tuple[int, int, int] = (64, 64, 3),
    n_particles: int = 1000
) -> Tuple[hk.Params, hk.State]:
    """
    Initialize model parameters.
    
    Args:
        rng: Random key for initialization
        config: Model configuration
        grid_shape: Shape of Eulerian grid (Nx, Ny, C_in)
        n_particles: Number of Lagrangian particles
    
    Returns:
        Tuple of (params, state)
    """
    model = create_model(config)
    
    # Dummy inputs for initialization
    dummy_grid = jnp.zeros((1,) + grid_shape)
    dummy_particles = jnp.zeros((1, n_particles, 3))
    
    params, state = model.init(rng, dummy_grid, dummy_particles, is_training=True)
    
    return params, state


# Convenience function for inference
@partial(jax.jit, static_argnums=(4,))
def predict(
    params: hk.Params,
    state: hk.State,
    grid_input: jnp.ndarray,
    particle_pos: jnp.ndarray,
    config: Optional[SurfPINNConfig] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run inference with trained model.
    
    Args:
        params: Model parameters
        state: Model state (batch norm stats)
        grid_input: Eulerian input
        particle_pos: Lagrangian positions
        config: Model configuration
    
    Returns:
        (height_pred, velocity_pred)
    """
    model = create_model(config)
    (height, velocity, _, _), _ = model.apply(
        params, state, None, grid_input, particle_pos, is_training=False
    )
    return height, velocity
