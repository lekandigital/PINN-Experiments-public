"""
Inverse Problem Module for WavePINN-NIF-Scalar.

This module enables inverse tasks such as:
- Estimating slowness/velocity maps from boundary measurements
- Joint optimization of wavefield and medium parameters
- Slowness tomography using physics-informed constraints

The inverse problem is formulated as:
    min_{u, m} L_data(u, u_obs) + λ_pde * L_pde(u, m) + λ_reg * R(m)

where:
    u = wavefield (from neural network u_θ)
    m = slowness field (from neural network m_φ)
    u_obs = observed boundary traces
    L_pde = physics constraint (wave equation residual)
    R(m) = regularization on slowness (smoothness, bounds)
"""

from typing import Callable, Dict, List, Optional, Tuple, Union, NamedTuple
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import haiku as hk
import optax

from .model import WavePINN, WavePINNConfig, get_activation
from .utils import (
    init_fourier_basis, make_fourier_features,
    save_checkpoint, MetricsLogger, count_params
)

# Type aliases
Array = jax.Array
PRNGKey = jax.Array
Params = hk.Params


# =============================================================================
# Slowness Network
# =============================================================================

def slowness_mlp(
    hidden_dims: List[int] = [64, 64, 32],
    activation: str = "tanh",
    output_activation: str = "softplus",
    name: str = "slowness_mlp"
) -> Callable:
    """
    Create MLP for slowness field prediction.
    
    Maps spatial coordinates (x, z) to slowness m(x, z) = 1/c(x, z).
    
    The output activation ensures positive slowness values.
    
    Args:
        hidden_dims: Hidden layer sizes.
        activation: Hidden layer activation.
        output_activation: Output activation ("softplus" ensures positivity).
        name: Module name.
        
    Returns:
        Haiku forward function.
    """
    act_fn = get_activation(activation)
    
    # Output activation to ensure positive slowness
    if output_activation == "softplus":
        out_act = jax.nn.softplus
    elif output_activation == "exp":
        out_act = jnp.exp
    elif output_activation == "sigmoid":
        out_act = jax.nn.sigmoid
    else:
        out_act = lambda x: x  # Identity
    
    def forward(coords: Array) -> Array:
        """
        Forward pass: (x, z) -> m(x, z).
        
        Args:
            coords: (batch, 2) or (batch, 3) spatial coordinates.
            
        Returns:
            (batch,) slowness values.
        """
        x = coords
        
        for i, dim in enumerate(hidden_dims):
            x = hk.Linear(dim, name=f"{name}_linear_{i}")(x)
            x = act_fn(x)
        
        # Output layer
        x = hk.Linear(1, name=f"{name}_output")(x)
        
        # Positive slowness via softplus + offset
        # This maps output to range [min_slowness, ∞)
        min_slowness = 1e-4  # ~10 km/s max velocity
        x = out_act(x) + min_slowness
        
        return x.squeeze(-1)
    
    return forward


def create_slowness_model(
    hidden_dims: List[int] = [64, 64, 32],
    activation: str = "tanh",
    use_fourier: bool = True,
    num_fourier_features: int = 32,
    fourier_scale: float = 5.0,
    seed: int = 42
):
    """
    Create slowness model with optional Fourier features.
    
    Args:
        hidden_dims: Hidden layer sizes.
        activation: Activation function.
        use_fourier: Whether to use Fourier feature encoding.
        num_fourier_features: Number of Fourier features.
        fourier_scale: Fourier feature scale.
        seed: Random seed.
        
    Returns:
        Tuple of (haiku_model, fourier_basis).
    """
    # Fourier basis for spatial coordinates
    if use_fourier:
        key = jr.PRNGKey(seed)
        spatial_dim = 2  # (x, z) or (x, y) for 2D
        fourier_B = init_fourier_basis(key, spatial_dim, num_fourier_features, fourier_scale)
    else:
        fourier_B = None
    
    def model_fn(coords: Array, B: Optional[Array] = None) -> Array:
        """Forward pass with optional Fourier encoding."""
        features = coords[:, :2]  # Take spatial coords only (x, z)
        
        if use_fourier and B is not None:
            features = make_fourier_features(B, features, include_input=True)
        
        mlp = slowness_mlp(hidden_dims, activation)
        return mlp(features)
    
    return hk.transform(model_fn), fourier_B


# =============================================================================
# Inverse Problem Configuration
# =============================================================================

class InverseConfig(NamedTuple):
    """Configuration for inverse problem."""
    # Slowness network
    slowness_hidden_dims: List[int] = [64, 64, 32]
    slowness_activation: str = "tanh"
    use_fourier_slowness: bool = True
    
    # Loss weights
    lambda_data: float = 100.0  # Data misfit weight
    lambda_pde: float = 1.0     # PDE constraint weight
    lambda_bc: float = 10.0     # Boundary condition weight
    lambda_ic: float = 10.0     # Initial condition weight
    lambda_smooth: float = 0.01 # Smoothness regularization
    
    # Slowness bounds (for regularization)
    min_velocity: float = 1000.0  # m/s
    max_velocity: float = 5000.0  # m/s
    
    # Training
    learning_rate: float = 1e-3
    n_iterations: int = 5000
    log_every: int = 100


# =============================================================================
# Inverse Problem Solver
# =============================================================================

class InverseProblem:
    """
    Inverse problem solver for slowness reconstruction.
    
    Jointly optimizes:
    1. Wavefield network u_θ(x, z, t)
    2. Slowness network m_φ(x, z)
    
    using boundary observations and physics constraints.
    """
    
    def __init__(
        self,
        wavefield_pinn: WavePINN,
        config: InverseConfig = InverseConfig(),
        seed: int = 42
    ):
        """
        Initialize inverse problem solver.
        
        Args:
            wavefield_pinn: Pre-configured WavePINN for wavefield.
            config: Inverse problem configuration.
            seed: Random seed.
        """
        self.wave_pinn = wavefield_pinn
        self.config = config
        
        # Create slowness network
        self.slowness_model, self.slowness_fourier_B = create_slowness_model(
            hidden_dims=config.slowness_hidden_dims,
            activation=config.slowness_activation,
            use_fourier=config.use_fourier_slowness,
            seed=seed
        )
    
    def init_params(self, key: PRNGKey) -> Tuple[Params, Params]:
        """
        Initialize both wavefield and slowness network parameters.
        
        Args:
            key: Random key.
            
        Returns:
            Tuple of (wave_params, slowness_params).
        """
        k1, k2 = jr.split(key)
        
        wave_params = self.wave_pinn.init_params(k1)
        
        # Initialize slowness network
        dummy_coords = jnp.ones((1, 2))
        slowness_params = self.slowness_model.init(k2, dummy_coords, self.slowness_fourier_B)
        
        return wave_params, slowness_params
    
    def predict_slowness(
        self,
        slowness_params: Params,
        coords: Array
    ) -> Array:
        """
        Predict slowness at given spatial coordinates.
        
        Args:
            slowness_params: Slowness network parameters.
            coords: (N, 2+) coordinates with x, z in first columns.
            
        Returns:
            (N,) slowness values in s/m.
        """
        return self.slowness_model.apply(
            slowness_params, None, coords[:, :2], self.slowness_fourier_B
        )
    
    def predict_velocity(
        self,
        slowness_params: Params,
        coords: Array
    ) -> Array:
        """Predict velocity (1/slowness) at coordinates."""
        slowness = self.predict_slowness(slowness_params, coords)
        return 1.0 / slowness
    
    def data_loss(
        self,
        wave_params: Params,
        observed_data: Dict[str, Array]
    ) -> Array:
        """
        Compute data misfit loss at observation points.
        
        Args:
            wave_params: Wavefield network parameters.
            observed_data: Dict with 'coords' (N, 3) and 'values' (N,).
            
        Returns:
            Scalar MSE loss.
        """
        coords = observed_data['coords']
        u_obs = observed_data['values']
        
        u_pred = self.wave_pinn.forward(wave_params, coords)
        
        return jnp.mean((u_pred - u_obs) ** 2)
    
    def smoothness_regularization(
        self,
        slowness_params: Params,
        coords: Array
    ) -> Array:
        """
        Compute smoothness regularization on slowness field.
        
        Penalizes spatial gradients: ||∇m||²
        
        Args:
            slowness_params: Slowness parameters.
            coords: Spatial coordinates.
            
        Returns:
            Scalar regularization term.
        """
        def m_at_point(x, z):
            coord = jnp.array([[x, z]])
            return self.slowness_model.apply(
                slowness_params, None, coord, self.slowness_fourier_B
            )[0]
        
        def grad_m_single(x, z):
            dm_dx = jax.grad(m_at_point, argnums=0)(x, z)
            dm_dz = jax.grad(m_at_point, argnums=1)(x, z)
            return dm_dx ** 2 + dm_dz ** 2
        
        # Vectorize over coordinates
        grad_norms = jax.vmap(grad_m_single)(coords[:, 0], coords[:, 1])
        
        return jnp.mean(grad_norms)
    
    def bounds_regularization(
        self,
        slowness_params: Params,
        coords: Array
    ) -> Array:
        """
        Regularization to keep velocity within physical bounds.
        
        Penalizes velocities outside [min_velocity, max_velocity].
        """
        cfg = self.config
        velocity = self.predict_velocity(slowness_params, coords)
        
        # Penalty for velocities below minimum
        below_min = jnp.maximum(cfg.min_velocity - velocity, 0) ** 2
        
        # Penalty for velocities above maximum
        above_max = jnp.maximum(velocity - cfg.max_velocity, 0) ** 2
        
        return jnp.mean(below_min + above_max)
    
    def total_loss(
        self,
        wave_params: Params,
        slowness_params: Params,
        batch: Dict[str, Array],
        observed_data: Dict[str, Array],
        return_components: bool = False
    ) -> Union[Array, Tuple[Array, Dict[str, Array]]]:
        """
        Compute total inverse problem loss.
        
        L = λ_data * L_data + λ_pde * L_pde + λ_bc * L_bc + 
            λ_ic * L_ic + λ_smooth * L_smooth
        
        Args:
            wave_params: Wavefield network parameters.
            slowness_params: Slowness network parameters.
            batch: PDE collocation batch (interior, boundary, initial coords).
            observed_data: Observed wavefield data.
            return_components: Whether to return individual loss components.
            
        Returns:
            Total loss, or (total_loss, components_dict).
        """
        cfg = self.config
        
        # Get velocity from slowness network at interior points
        velocity = self.predict_velocity(slowness_params, batch['interior'])
        
        # Create batch with predicted velocity
        pde_batch = {
            'interior': batch['interior'],
            'boundary': batch['boundary'],
            'initial': batch['initial'],
            'velocity': velocity
        }
        
        # PDE loss (wavefield must satisfy wave equation with predicted velocity)
        _, wave_components = self.wave_pinn.total_loss(
            wave_params, pde_batch, return_components=True
        )
        
        # Data misfit loss
        loss_data = self.data_loss(wave_params, observed_data)
        
        # Smoothness regularization
        loss_smooth = self.smoothness_regularization(slowness_params, batch['interior'])
        
        # Bounds regularization
        loss_bounds = self.bounds_regularization(slowness_params, batch['interior'])
        
        # Weighted total
        total = (
            cfg.lambda_data * loss_data +
            cfg.lambda_pde * wave_components['pde'] +
            cfg.lambda_bc * wave_components['bc'] +
            cfg.lambda_ic * wave_components['ic_u'] +
            cfg.lambda_smooth * loss_smooth +
            0.1 * loss_bounds  # Small weight for bounds
        )
        
        if return_components:
            components = {
                'data': loss_data,
                'pde': wave_components['pde'],
                'bc': wave_components['bc'],
                'ic': wave_components['ic_u'],
                'smooth': loss_smooth,
                'bounds': loss_bounds,
                'total': total
            }
            return total, components
        
        return total
    
    def invert(
        self,
        batch: Dict[str, Array],
        observed_data: Dict[str, Array],
        key: Optional[PRNGKey] = None,
        log_dir: Optional[str] = None
    ) -> Tuple[Params, Params, List[Dict]]:
        """
        Run inverse optimization to recover slowness field.
        
        Args:
            batch: Collocation points (interior, boundary, initial).
            observed_data: Observed wavefield at boundary.
            key: Random key for initialization.
            log_dir: Optional directory for checkpoints.
            
        Returns:
            Tuple of (wave_params, slowness_params, history).
        """
        cfg = self.config
        
        if key is None:
            key = jr.PRNGKey(42)
        
        # Initialize parameters
        wave_params, slowness_params = self.init_params(key)
        
        print(f"Wavefield params: {count_params(wave_params):,}")
        print(f"Slowness params: {count_params(slowness_params):,}")
        
        # Create optimizer
        optimizer = optax.adam(cfg.learning_rate)
        
        # Combine params for joint optimization
        all_params = {'wave': wave_params, 'slowness': slowness_params}
        opt_state = optimizer.init(all_params)
        
        def loss_fn(params):
            return self.total_loss(
                params['wave'], params['slowness'],
                batch, observed_data, return_components=False
            )
        
        @jax.jit
        def train_step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss
        
        # Training loop
        history = []
        logger = MetricsLogger()
        
        print(f"\nStarting inverse optimization for {cfg.n_iterations} iterations...")
        
        for step in range(cfg.n_iterations):
            all_params, opt_state, loss = train_step(all_params, opt_state)
            
            if step % cfg.log_every == 0 or step == cfg.n_iterations - 1:
                # Get detailed losses
                _, components = self.total_loss(
                    all_params['wave'], all_params['slowness'],
                    batch, observed_data, return_components=True
                )
                
                metrics = {k: float(v) for k, v in components.items()}
                metrics['step'] = step
                
                logger.print_status(step, metrics)
                history.append(metrics)
        
        print(f"\nInverse optimization complete!")
        print(f"Final loss: {history[-1]['total']:.4e}")
        print(f"Data misfit: {history[-1]['data']:.4e}")
        
        return all_params['wave'], all_params['slowness'], history


# =============================================================================
# Utility Functions for Inverse Problems
# =============================================================================

def generate_synthetic_observations(
    pinn: WavePINN,
    params: Params,
    boundary_coords: Array,
    noise_level: float = 0.0,
    key: Optional[PRNGKey] = None
) -> Dict[str, Array]:
    """
    Generate synthetic boundary observations from a trained PINN.
    
    Useful for testing inverse module on known ground truth.
    
    Args:
        pinn: Trained WavePINN.
        params: Trained parameters.
        boundary_coords: (N, 3) boundary coordinates [x, z, t].
        noise_level: Relative noise level (0.0 = no noise).
        key: Random key for noise.
        
    Returns:
        Dict with 'coords' and 'values'.
    """
    u_clean = pinn.forward(params, boundary_coords)
    
    if noise_level > 0 and key is not None:
        noise = jr.normal(key, u_clean.shape) * noise_level * jnp.std(u_clean)
        u_obs = u_clean + noise
    else:
        u_obs = u_clean
    
    return {
        'coords': boundary_coords,
        'values': u_obs
    }


def evaluate_slowness_reconstruction(
    true_slowness: Array,
    pred_slowness: Array,
    spatial_coords: Array
) -> Dict[str, float]:
    """
    Evaluate slowness reconstruction quality.
    
    Args:
        true_slowness: Ground truth slowness grid.
        pred_slowness: Predicted slowness grid (same shape).
        spatial_coords: (N, 2) coordinates for the grids.
        
    Returns:
        Dictionary of metrics (MSE, MAE, relative error).
    """
    # Flatten if needed
    true_flat = true_slowness.ravel()
    pred_flat = pred_slowness.ravel()
    
    mse = jnp.mean((true_flat - pred_flat) ** 2)
    mae = jnp.mean(jnp.abs(true_flat - pred_flat))
    relative_error = jnp.mean(jnp.abs(true_flat - pred_flat) / (jnp.abs(true_flat) + 1e-8))
    
    # Convert to velocity and compute metrics
    true_vel = 1.0 / true_flat
    pred_vel = 1.0 / pred_flat
    vel_mse = jnp.mean((true_vel - pred_vel) ** 2)
    
    return {
        'slowness_mse': float(mse),
        'slowness_mae': float(mae),
        'relative_error': float(relative_error),
        'velocity_mse': float(vel_mse),
    }
