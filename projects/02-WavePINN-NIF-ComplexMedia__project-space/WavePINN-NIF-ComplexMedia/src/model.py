"""
Neural Implicit Field (NIF) model architecture for WavePINN.

Implements:
- FourierFeatureEncoder: Random Fourier features to overcome spectral bias
- WavePINN: Main PINN for wavefield u(x, y, z, t)
- MediaNIF: Neural Implicit Field for wave speed c(x, y, z)
"""

import haiku as hk
import jax
import jax.numpy as jnp
from typing import Optional, List, Tuple, Callable


class FourierFeatureEncoder(hk.Module):
    """
    Fourier feature mapping to overcome spectral bias in neural networks.
    
    Maps input coordinates to high-dimensional Fourier features:
    γ(x) = [cos(2π B x), sin(2π B x)]
    
    where B is a random matrix sampled from N(0, σ²).
    
    Reference: Tancik et al., "Fourier Features Let Networks Learn 
    High Frequency Functions in Low Dimensional Domains" (NeurIPS 2020)
    """
    
    def __init__(self, 
                 output_dim: int = 256,
                 sigma: float = 10.0,
                 trainable: bool = False,
                 name: Optional[str] = None):
        """
        Initialize Fourier feature encoder.
        
        Args:
            output_dim: Dimension of output features (must be even)
            sigma: Standard deviation of random frequencies
            trainable: Whether frequency matrix B is trainable
            name: Module name
        """
        super().__init__(name=name)
        assert output_dim % 2 == 0, "output_dim must be even"
        self.output_dim = output_dim
        self.sigma = sigma
        self.trainable = trainable
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply Fourier feature encoding.
        
        Args:
            x: Input coordinates of shape (batch, input_dim)
            
        Returns:
            Fourier features of shape (batch, output_dim)
        """
        input_dim = x.shape[-1]
        
        # Random frequency matrix B
        if self.trainable:
            B = hk.get_parameter(
                "B",
                shape=[input_dim, self.output_dim // 2],
                init=hk.initializers.RandomNormal(stddev=self.sigma)
            )
        else:
            # Use fixed random frequencies (via hk.get_state for consistency)
            B = hk.get_parameter(
                "B",
                shape=[input_dim, self.output_dim // 2],
                init=hk.initializers.RandomNormal(stddev=self.sigma)
            )
        
        # Project and compute Fourier features
        x_proj = 2.0 * jnp.pi * x @ B
        
        return jnp.concatenate([jnp.cos(x_proj), jnp.sin(x_proj)], axis=-1)


class SinusoidalPositionalEncoding(hk.Module):
    """
    Sinusoidal positional encoding (alternative to random Fourier features).
    
    Uses fixed frequencies at multiple scales.
    """
    
    def __init__(self,
                 n_frequencies: int = 10,
                 min_freq: float = 1.0,
                 max_freq: float = 100.0,
                 log_sampling: bool = True,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.n_frequencies = n_frequencies
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.log_sampling = log_sampling
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply positional encoding."""
        if self.log_sampling:
            freqs = jnp.logspace(
                jnp.log10(self.min_freq),
                jnp.log10(self.max_freq),
                self.n_frequencies
            )
        else:
            freqs = jnp.linspace(self.min_freq, self.max_freq, self.n_frequencies)
        
        # Encode each input dimension
        encoded = [x]
        for freq in freqs:
            encoded.append(jnp.sin(2 * jnp.pi * freq * x))
            encoded.append(jnp.cos(2 * jnp.pi * freq * x))
        
        return jnp.concatenate(encoded, axis=-1)


class WavePINN(hk.Module):
    """
    Main Physics-Informed Neural Network for wavefield u(x, y, [z], t).
    
    Architecture:
    - Fourier feature encoding (optional)
    - Deep MLP with residual connections
    - Output: scalar wavefield value
    """
    
    def __init__(self,
                 hidden_dims: List[int] = [256, 256, 256, 256],
                 use_fourier: bool = True,
                 fourier_dim: int = 256,
                 fourier_sigma: float = 10.0,
                 activation: str = "tanh",
                 use_residual: bool = True,
                 name: Optional[str] = None):
        """
        Initialize WavePINN.
        
        Args:
            hidden_dims: List of hidden layer dimensions
            use_fourier: Whether to use Fourier feature encoding
            fourier_dim: Dimension of Fourier features
            fourier_sigma: Bandwidth of Fourier features
            activation: Activation function ("tanh", "swish", "gelu")
            use_residual: Whether to use residual connections
            name: Module name
        """
        super().__init__(name=name)
        self.hidden_dims = hidden_dims
        self.use_fourier = use_fourier
        self.fourier_dim = fourier_dim
        self.fourier_sigma = fourier_sigma
        self.use_residual = use_residual
        
        # Select activation function
        self.activation_fn = self._get_activation(activation)
    
    def _get_activation(self, name: str) -> Callable:
        """Get activation function by name."""
        activations = {
            "tanh": jax.nn.tanh,
            "swish": jax.nn.swish,
            "gelu": jax.nn.gelu,
            "relu": jax.nn.relu,
            "softplus": jax.nn.softplus,
            "sin": jnp.sin,  # SIREN-style
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return activations[name]
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input coordinates (batch, ndim+1) where last dim is time
            
        Returns:
            Wavefield values (batch, 1)
        """
        # Optional Fourier feature encoding
        if self.use_fourier:
            encoder = FourierFeatureEncoder(
                output_dim=self.fourier_dim,
                sigma=self.fourier_sigma
            )
            h = encoder(x)
        else:
            h = x
        
        # MLP with residual connections
        for i, dim in enumerate(self.hidden_dims):
            h_prev = h
            h = hk.Linear(dim, name=f"layer_{i}")(h)
            h = self.activation_fn(h)
            
            # Residual connection (if dimensions match and not first layer)
            if self.use_residual and i > 0 and h.shape[-1] == h_prev.shape[-1]:
                h = h + h_prev
        
        # Output layer: scalar wavefield
        u = hk.Linear(1, name="output")(h)
        
        return u


class MediaNIF(hk.Module):
    """
    Neural Implicit Field for wave speed c(x, y, [z]).
    
    Learns the heterogeneous velocity model as a continuous function.
    Output is constrained to be positive via softplus activation.
    """
    
    def __init__(self,
                 hidden_dims: List[int] = [128, 128],
                 activation: str = "softplus",
                 c_min: float = 1.0,
                 c_max: float = 5.0,
                 use_fourier: bool = True,
                 fourier_dim: int = 128,
                 fourier_sigma: float = 5.0,
                 name: Optional[str] = None):
        """
        Initialize MediaNIF.
        
        Args:
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            c_min: Minimum wave speed (output constraint)
            c_max: Maximum wave speed (output constraint)
            use_fourier: Whether to use Fourier encoding
            fourier_dim: Dimension of Fourier features
            fourier_sigma: Bandwidth of Fourier features
            name: Module name
        """
        super().__init__(name=name)
        self.hidden_dims = hidden_dims
        self.c_min = c_min
        self.c_max = c_max
        self.use_fourier = use_fourier
        self.fourier_dim = fourier_dim
        self.fourier_sigma = fourier_sigma
        self.activation_fn = self._get_activation(activation)
    
    def _get_activation(self, name: str) -> Callable:
        """Get activation function by name."""
        activations = {
            "tanh": jax.nn.tanh,
            "swish": jax.nn.swish,
            "gelu": jax.nn.gelu,
            "relu": jax.nn.relu,
            "softplus": jax.nn.softplus,
        }
        return activations.get(name, jax.nn.softplus)
    
    def __call__(self, x_spatial: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass.
        
        Args:
            x_spatial: Spatial coordinates only (batch, ndim), no time
            
        Returns:
            Wave speed values (batch, 1), constrained to [c_min, c_max]
        """
        # Optional Fourier encoding
        if self.use_fourier:
            encoder = FourierFeatureEncoder(
                output_dim=self.fourier_dim,
                sigma=self.fourier_sigma,
                name="media_fourier"
            )
            h = encoder(x_spatial)
        else:
            h = x_spatial
        
        # MLP
        for i, dim in enumerate(self.hidden_dims):
            h = hk.Linear(dim, name=f"media_layer_{i}")(h)
            h = self.activation_fn(h)
        
        # Output layer with sigmoid to constrain to [c_min, c_max]
        raw_output = hk.Linear(1, name="media_output")(h)
        
        # Constrain output to [c_min, c_max] via scaled sigmoid
        c = self.c_min + (self.c_max - self.c_min) * jax.nn.sigmoid(raw_output)
        
        return c


class CombinedWavePINNNIF(hk.Module):
    """
    Combined model with both wavefield network and media network.
    
    This module provides a unified interface for joint forward/inverse problems.
    """
    
    def __init__(self,
                 wave_config: dict,
                 media_config: dict,
                 name: Optional[str] = None):
        """
        Initialize combined model.
        
        Args:
            wave_config: Configuration dict for WavePINN
            media_config: Configuration dict for MediaNIF
            name: Module name
        """
        super().__init__(name=name)
        self.wave_config = wave_config
        self.media_config = media_config
    
    def __call__(self, 
                 x: jnp.ndarray, 
                 return_media: bool = False) -> Tuple[jnp.ndarray, ...]:
        """
        Forward pass.
        
        Args:
            x: Full coordinates (batch, ndim+1) including time
            return_media: Whether to also return wave speed
            
        Returns:
            u: Wavefield (batch, 1)
            c: Wave speed (batch, 1) if return_media=True
        """
        # Wavefield network
        wave_net = WavePINN(**self.wave_config)
        u = wave_net(x)
        
        if return_media:
            # Media network (spatial coordinates only)
            media_net = MediaNIF(**self.media_config)
            x_spatial = x[..., :-1]  # Remove time dimension
            c = media_net(x_spatial)
            return u, c
        
        return u


def create_model(config: Optional[dict] = None) -> hk.Transformed:
    """
    Factory function to create the Haiku-transformed model.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Haiku transformed model with init and apply functions
    """
    if config is None:
        config = {
            'wave': {
                'hidden_dims': [256, 256, 256, 256],
                'use_fourier': True,
                'fourier_dim': 256,
                'fourier_sigma': 10.0,
                'activation': 'tanh',
                'use_residual': True,
            },
            'media': {
                'hidden_dims': [128, 128],
                'activation': 'softplus',
                'c_min': 1.0,
                'c_max': 5.0,
                'use_fourier': True,
                'fourier_dim': 128,
                'fourier_sigma': 5.0,
            }
        }
    
    def forward_fn(x: jnp.ndarray, return_media: bool = False):
        """Forward function for Haiku transform."""
        wave_net = WavePINN(
            hidden_dims=config['wave']['hidden_dims'],
            use_fourier=config['wave']['use_fourier'],
            fourier_dim=config['wave']['fourier_dim'],
            fourier_sigma=config['wave']['fourier_sigma'],
            activation=config['wave']['activation'],
            use_residual=config['wave']['use_residual'],
        )
        u = wave_net(x)
        
        if return_media:
            media_net = MediaNIF(
                hidden_dims=config['media']['hidden_dims'],
                activation=config['media']['activation'],
                c_min=config['media']['c_min'],
                c_max=config['media']['c_max'],
                use_fourier=config['media']['use_fourier'],
                fourier_dim=config['media']['fourier_dim'],
                fourier_sigma=config['media']['fourier_sigma'],
            )
            x_spatial = x[..., :-1]
            c = media_net(x_spatial)
            return u, c
        
        return u
    
    return hk.transform(forward_fn)


def create_wave_only_model(config: Optional[dict] = None) -> hk.Transformed:
    """Create a model with only the wavefield network (for forward problems)."""
    if config is None:
        config = {
            'hidden_dims': [256, 256, 256, 256],
            'use_fourier': True,
            'fourier_dim': 256,
            'fourier_sigma': 10.0,
            'activation': 'tanh',
            'use_residual': True,
        }
    
    def forward_fn(x: jnp.ndarray):
        wave_net = WavePINN(**config)
        return wave_net(x)
    
    return hk.transform(forward_fn)


def create_media_only_model(config: Optional[dict] = None) -> hk.Transformed:
    """Create a model with only the media network (for pure inverse problems)."""
    if config is None:
        config = {
            'hidden_dims': [128, 128],
            'activation': 'softplus',
            'c_min': 1.0,
            'c_max': 5.0,
            'use_fourier': True,
            'fourier_dim': 128,
            'fourier_sigma': 5.0,
        }
    
    def forward_fn(x_spatial: jnp.ndarray):
        media_net = MediaNIF(**config)
        return media_net(x_spatial)
    
    return hk.transform(forward_fn)


def count_parameters(params: dict) -> int:
    """Count total number of trainable parameters."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


if __name__ == "__main__":
    # Quick test
    print("Testing WavePINN-NIF model...")
    
    # Create model
    model = create_model()
    
    # Initialize
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((1, 3))  # (x, y, t) for 2D
    params = model.init(rng, dummy_input)
    
    print(f"✓ Model initialized")
    print(f"  Total parameters: {count_parameters(params):,}")
    
    # Forward pass (wavefield only)
    u = model.apply(params, rng, dummy_input)
    print(f"✓ Forward pass (u only): {u.shape}")
    
    # Forward pass (wavefield + media)
    u, c = model.apply(params, rng, dummy_input, return_media=True)
    print(f"✓ Forward pass (u, c): u={u.shape}, c={c.shape}")
    
    # Test batch processing
    batch_input = jnp.ones((100, 3))
    u_batch = model.apply(params, rng, batch_input)
    print(f"✓ Batch forward pass: {u_batch.shape}")
    
    # Test 3D case
    dummy_3d = jnp.zeros((1, 4))  # (x, y, z, t)
    params_3d = model.init(rng, dummy_3d)
    u_3d = model.apply(params_3d, rng, dummy_3d)
    print(f"✓ 3D forward pass: {u_3d.shape}")
    
    print("\n✅ All model tests passed!")
