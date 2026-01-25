"""
Neural Implicit Field Model for WavePINN-NIF-Scalar.

This module implements the core PINN model including:
- Coordinate-based MLP with Fourier feature encoding
- Physics-informed loss functions (PDE, BC, IC)
- Proper autodiff for second-order derivatives (u_tt, ∇²u)
- Support for both 2D and 3D wave equations

The acoustic wave equation solved is:
    ∂²u/∂t² = c²(x) ∇²u + s(x, t)

where:
    u(x, t) = wavefield (pressure/displacement)
    c(x) = velocity field (m/s), or slowness m(x) = 1/c(x)
    s(x, t) = source term (Ricker wavelet)
"""

from typing import Callable, Dict, List, Optional, Tuple, Union, NamedTuple
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import haiku as hk

from .utils import init_fourier_basis, make_fourier_features

# Type aliases
Array = jax.Array
PRNGKey = jax.Array
Params = hk.Params


# =============================================================================
# Configuration
# =============================================================================

class WavePINNConfig(NamedTuple):
    """Configuration for WavePINN model."""
    hidden_dims: List[int] = [128, 128, 64]
    activation: str = "tanh"  # "tanh", "sin", "relu", "gelu"
    use_fourier_features: bool = True
    num_fourier_features: int = 64
    fourier_scale: float = 10.0
    output_dim: int = 1  # scalar wavefield
    # Loss weights
    lambda_pde: float = 1.0
    lambda_bc: float = 10.0
    lambda_ic: float = 10.0
    lambda_ic_dt: float = 1.0  # weight for du/dt at t=0
    # Domain info
    spatial_dim: int = 2  # 2D or 3D


# =============================================================================
# Activation Functions
# =============================================================================

def get_activation(name: str) -> Callable:
    """Get activation function by name."""
    activations = {
        'tanh': jnp.tanh,
        'sin': jnp.sin,
        'relu': jax.nn.relu,
        'gelu': jax.nn.gelu,
        'swish': jax.nn.swish,
        'softplus': jax.nn.softplus,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    return activations[name]


# =============================================================================
# Neural Network Architecture
# =============================================================================

def wavefield_mlp(
    hidden_dims: List[int],
    activation: str = "tanh",
    output_dim: int = 1,
    name: str = "wavefield_mlp"
) -> Callable:
    """
    Create a coordinate-based MLP for wavefield prediction.
    
    Maps (x, z, t) or encoded features to scalar wavefield u.
    
    Args:
        hidden_dims: List of hidden layer sizes.
        activation: Activation function name.
        output_dim: Output dimension (1 for scalar wavefield).
        name: Module name for Haiku.
        
    Returns:
        Haiku forward function.
    """
    act_fn = get_activation(activation)
    
    def forward(features: Array) -> Array:
        """
        Forward pass: features -> u(x, t).
        
        Args:
            features: (batch, n_features) input features.
            
        Returns:
            (batch, output_dim) wavefield values.
        """
        x = features
        
        for i, dim in enumerate(hidden_dims):
            x = hk.Linear(dim, name=f"{name}_linear_{i}")(x)
            x = act_fn(x)
        
        # Output layer (no activation)
        x = hk.Linear(output_dim, name=f"{name}_output")(x)
        
        return x
    
    return forward


def create_wavefield_model(config: WavePINNConfig):
    """
    Create the complete wavefield model with optional Fourier features.
    
    Args:
        config: Model configuration.
        
    Returns:
        Haiku transformed model.
    """
    def model_fn(coords: Array, fourier_B: Optional[Array] = None) -> Array:
        """
        Full forward pass: coords -> u(x, t).
        
        Args:
            coords: (batch, spatial_dim + 1) coordinates [x, z, t] or [x, y, z, t].
            fourier_B: Optional Fourier basis matrix for feature encoding.
            
        Returns:
            (batch,) or (batch, 1) wavefield predictions.
        """
        features = coords
        
        # Apply Fourier feature encoding if enabled
        if config.use_fourier_features and fourier_B is not None:
            features = make_fourier_features(fourier_B, coords, include_input=True)
        
        # MLP prediction
        mlp = wavefield_mlp(
            hidden_dims=config.hidden_dims,
            activation=config.activation,
            output_dim=config.output_dim
        )
        
        u = mlp(features)
        
        # Squeeze to (batch,) if output_dim=1
        if config.output_dim == 1:
            u = u.squeeze(-1)
        
        return u
    
    return hk.transform(model_fn)


# =============================================================================
# Physics-Informed Loss Functions
# =============================================================================

class WavePINN:
    """
    Physics-Informed Neural Network for 2D/3D acoustic wave equation.
    
    Solves: ∂²u/∂t² = c²(x) ∇²u + s(x, t)
    
    with Dirichlet BCs (u=0 on boundary) and zero initial conditions.
    """
    
    def __init__(self, config: WavePINNConfig, seed: int = 42):
        """
        Initialize WavePINN.
        
        Args:
            config: Model configuration.
            seed: Random seed for initialization.
        """
        self.config = config
        self.model = create_wavefield_model(config)
        
        # Initialize Fourier basis if needed
        key = jr.PRNGKey(seed)
        input_dim = config.spatial_dim + 1  # x, z, t (or x, y, z, t)
        
        if config.use_fourier_features:
            self.fourier_B = init_fourier_basis(
                key, input_dim, config.num_fourier_features, config.fourier_scale
            )
        else:
            self.fourier_B = None
    
    def init_params(self, key: PRNGKey) -> Params:
        """
        Initialize model parameters.
        
        Args:
            key: JAX random key.
            
        Returns:
            Initialized Haiku parameters.
        """
        dummy_coords = jnp.ones((1, self.config.spatial_dim + 1))
        return self.model.init(key, dummy_coords, self.fourier_B)
    
    def forward(self, params: Params, coords: Array) -> Array:
        """
        Forward pass: predict wavefield at coordinates.
        
        Args:
            params: Model parameters.
            coords: (batch, ndim) coordinates.
            
        Returns:
            (batch,) wavefield predictions.
        """
        return self.model.apply(params, None, coords, self.fourier_B)
    
    def _u_single(self, params: Params, x: float, z: float, t: float) -> float:
        """
        Evaluate u at a single point (for autodiff).
        
        This function takes scalar inputs for proper differentiation.
        """
        coords = jnp.array([[x, z, t]])
        return self.model.apply(params, None, coords, self.fourier_B)[0]
    
    def compute_pde_residual_2d(
        self,
        params: Params,
        coords: Array,
        velocity: Array,
        source: Optional[Array] = None
    ) -> Array:
        """
        Compute PDE residual: r = u_tt - c²(∇²u) - s.
        
        Uses JAX autodiff to compute second derivatives.
        
        Args:
            params: Model parameters.
            coords: (N, 3) coordinates [x, z, t].
            velocity: (N,) velocity at each point.
            source: (N,) optional source term.
            
        Returns:
            (N,) residual values.
        """
        # Define scalar function for single-point evaluation
        def u_scalar(x, z, t):
            coord = jnp.stack([x, z, t])
            return self.model.apply(params, None, coord[None, :], self.fourier_B)[0]
        
        # Compute second derivatives using nested grad
        def compute_derivs_single(x, z, t):
            # First derivatives
            u_x = jax.grad(u_scalar, argnums=0)(x, z, t)
            u_z = jax.grad(u_scalar, argnums=1)(x, z, t)
            u_t = jax.grad(u_scalar, argnums=2)(x, z, t)
            
            # Second derivatives
            u_xx = jax.grad(lambda xi: jax.grad(u_scalar, argnums=0)(xi, z, t))(x)
            u_zz = jax.grad(lambda zi: jax.grad(u_scalar, argnums=1)(x, zi, t))(z)
            u_tt = jax.grad(lambda ti: jax.grad(u_scalar, argnums=2)(x, z, ti))(t)
            
            return u_tt, u_xx, u_zz
        
        # Vectorize over batch
        u_tt, u_xx, u_zz = jax.vmap(
            compute_derivs_single
        )(coords[:, 0], coords[:, 1], coords[:, 2])
        
        # Laplacian
        laplacian = u_xx + u_zz
        
        # PDE: u_tt - c² * ∇²u - s = 0
        c2 = velocity ** 2
        residual = u_tt - c2 * laplacian
        
        if source is not None:
            residual = residual - source
        
        return residual
    
    def compute_pde_residual_2d_efficient(
        self,
        params: Params,
        coords: Array,
        velocity: Array,
        source: Optional[Array] = None
    ) -> Array:
        """
        Efficient PDE residual using forward-mode differentiation.
        
        This version uses jax.jacfwd for potentially better performance
        on GPUs with large batch sizes.
        """
        def u_fn(coords_single):
            """Evaluate u at single coordinate."""
            return self.model.apply(params, None, coords_single[None, :], self.fourier_B)[0]
        
        def compute_hessian_diag(coord):
            """Compute diagonal of Hessian (u_xx, u_zz, u_tt)."""
            # Gradient
            grad_u = jax.grad(u_fn)(coord)
            
            # Hessian diagonal via second grad
            def grad_component(i):
                return jax.grad(lambda c: jax.grad(u_fn)(c)[i])(coord)[i]
            
            u_xx = grad_component(0)
            u_zz = grad_component(1)
            u_tt = grad_component(2)
            
            return u_tt, u_xx + u_zz  # u_tt and laplacian
        
        # Vectorize
        u_tt, laplacian = jax.vmap(compute_hessian_diag)(coords)
        
        # PDE residual
        c2 = velocity ** 2
        residual = u_tt - c2 * laplacian
        
        if source is not None:
            residual = residual - source
        
        return residual
    
    def boundary_loss(
        self,
        params: Params,
        boundary_coords: Array,
        bc_values: Optional[Array] = None
    ) -> Array:
        """
        Compute boundary condition loss (Dirichlet: u = bc_values).
        
        Args:
            params: Model parameters.
            boundary_coords: (N, ndim) boundary point coordinates.
            bc_values: (N,) target values (default: 0).
            
        Returns:
            Scalar MSE loss.
        """
        u_pred = self.forward(params, boundary_coords)
        
        if bc_values is None:
            bc_values = jnp.zeros_like(u_pred)
        
        return jnp.mean((u_pred - bc_values) ** 2)
    
    def initial_loss(
        self,
        params: Params,
        initial_coords: Array,
        u0_values: Optional[Array] = None,
        include_velocity: bool = True
    ) -> Tuple[Array, Array]:
        """
        Compute initial condition loss: u(x, 0) = u0, ∂u/∂t(x, 0) = 0.
        
        Args:
            params: Model parameters.
            initial_coords: (N, ndim) coordinates at t=0.
            u0_values: (N,) initial displacement (default: 0).
            include_velocity: Whether to enforce ∂u/∂t = 0.
            
        Returns:
            Tuple of (u_loss, u_t_loss) scalar losses.
        """
        u_pred = self.forward(params, initial_coords)
        
        if u0_values is None:
            u0_values = jnp.zeros_like(u_pred)
        
        u_loss = jnp.mean((u_pred - u0_values) ** 2)
        
        if include_velocity:
            # Compute ∂u/∂t at initial time
            def u_scalar(x, z, t):
                coord = jnp.stack([x, z, t])
                return self.model.apply(params, None, coord[None, :], self.fourier_B)[0]
            
            def compute_u_t(x, z, t):
                return jax.grad(u_scalar, argnums=2)(x, z, t)
            
            u_t = jax.vmap(compute_u_t)(
                initial_coords[:, 0],
                initial_coords[:, 1],
                initial_coords[:, 2]
            )
            u_t_loss = jnp.mean(u_t ** 2)
        else:
            u_t_loss = jnp.array(0.0)
        
        return u_loss, u_t_loss
    
    def total_loss(
        self,
        params: Params,
        batch: Dict[str, Array],
        return_components: bool = False
    ) -> Union[Array, Tuple[Array, Dict[str, Array]]]:
        """
        Compute total physics-informed loss.
        
        L = λ_pde * L_pde + λ_bc * L_bc + λ_ic * L_ic + λ_ic_dt * L_ic_dt
        
        Args:
            params: Model parameters.
            batch: Dictionary containing:
                - 'interior': (N, 3) PDE collocation points
                - 'velocity': (N,) velocity at interior points
                - 'boundary': (M, 3) boundary points
                - 'initial': (K, 3) initial condition points
                - 'source': (N,) optional source term
            return_components: If True, also return individual losses.
            
        Returns:
            Total loss, or (total_loss, loss_dict) if return_components=True.
        """
        cfg = self.config
        
        # PDE residual loss
        pde_residual = self.compute_pde_residual_2d_efficient(
            params,
            batch['interior'],
            batch['velocity'],
            batch.get('source', None)
        )
        loss_pde = jnp.mean(pde_residual ** 2)
        
        # Boundary loss
        loss_bc = self.boundary_loss(params, batch['boundary'])
        
        # Initial condition loss
        loss_ic_u, loss_ic_dt = self.initial_loss(params, batch['initial'])
        
        # Total weighted loss
        total = (
            cfg.lambda_pde * loss_pde +
            cfg.lambda_bc * loss_bc +
            cfg.lambda_ic * loss_ic_u +
            cfg.lambda_ic_dt * loss_ic_dt
        )
        
        if return_components:
            components = {
                'pde': loss_pde,
                'bc': loss_bc,
                'ic_u': loss_ic_u,
                'ic_dt': loss_ic_dt,
                'total': total
            }
            return total, components
        
        return total


# =============================================================================
# Simplified Interface for Quick Testing
# =============================================================================

def create_simple_pinn(
    hidden_dims: List[int] = [128, 128, 64],
    use_fourier: bool = True,
    seed: int = 42
) -> WavePINN:
    """
    Create a WavePINN with default configuration.
    
    Args:
        hidden_dims: Hidden layer sizes.
        use_fourier: Whether to use Fourier features.
        seed: Random seed.
        
    Returns:
        Initialized WavePINN instance.
        
    Example:
        >>> pinn = create_simple_pinn()
        >>> params = pinn.init_params(jax.random.PRNGKey(0))
        >>> coords = jnp.array([[0.5, 0.5, 0.1]])
        >>> u = pinn.forward(params, coords)
    """
    config = WavePINNConfig(
        hidden_dims=hidden_dims,
        use_fourier_features=use_fourier,
    )
    return WavePINN(config, seed=seed)


# =============================================================================
# Gradient and Loss Computation Utilities
# =============================================================================

@partial(jax.jit, static_argnums=(0,))
def compute_loss_and_grad(
    pinn: WavePINN,
    params: Params,
    batch: Dict[str, Array]
) -> Tuple[Array, Params, Dict[str, Array]]:
    """
    JIT-compiled loss and gradient computation.
    
    Args:
        pinn: WavePINN instance.
        params: Model parameters.
        batch: Training batch.
        
    Returns:
        Tuple of (loss, gradients, loss_components).
    """
    def loss_fn(p):
        return pinn.total_loss(p, batch, return_components=False)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    _, components = pinn.total_loss(params, batch, return_components=True)
    
    return loss, grads, components


# =============================================================================
# Model Prediction Utilities
# =============================================================================

def predict_on_grid(
    pinn: WavePINN,
    params: Params,
    nx: int,
    nz: int,
    t: float,
    x_range: Tuple[float, float] = (0.0, 1.0),
    z_range: Tuple[float, float] = (0.0, 1.0)
) -> Array:
    """
    Predict wavefield on a regular spatial grid at time t.
    
    Args:
        pinn: WavePINN instance.
        params: Model parameters.
        nx, nz: Grid resolution.
        t: Time instant.
        x_range, z_range: Spatial domain.
        
    Returns:
        (nx, nz) array of wavefield values.
    """
    x = jnp.linspace(x_range[0], x_range[1], nx)
    z = jnp.linspace(z_range[0], z_range[1], nz)
    xx, zz = jnp.meshgrid(x, z, indexing='ij')
    
    coords = jnp.stack([
        xx.ravel(),
        zz.ravel(),
        jnp.full(nx * nz, t)
    ], axis=-1)
    
    u_flat = pinn.forward(params, coords)
    return u_flat.reshape(nx, nz)


def predict_time_series(
    pinn: WavePINN,
    params: Params,
    location: Tuple[float, float],
    times: Array
) -> Array:
    """
    Predict wavefield time series at a fixed location.
    
    Args:
        pinn: WavePINN instance.
        params: Model parameters.
        location: (x, z) spatial location.
        times: (T,) time array.
        
    Returns:
        (T,) wavefield values over time.
    """
    x, z = location
    coords = jnp.stack([
        jnp.full(len(times), x),
        jnp.full(len(times), z),
        times
    ], axis=-1)
    
    return pinn.forward(params, coords)
