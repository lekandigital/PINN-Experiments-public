"""
Physics-informed loss functions for the acoustic wave equation.

Implements PDE residuals, boundary conditions, and initial conditions
using JAX automatic differentiation.

Governing equation:
    ∂²u/∂t² - c²(x)∇²u = 0

where u(x, t) is the wavefield and c(x) is the spatially-varying wave speed.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Dict, Tuple, Optional
from functools import partial


class WavePDELoss:
    """
    Physics-informed loss for the acoustic wave equation.
    
    Computes:
    - PDE residual: ∂²u/∂t² - c²(∂²u/∂x² + ∂²u/∂y² + ...)
    - Boundary condition loss (Dirichlet, Neumann, or Absorbing)
    - Initial condition loss (u(t=0) and ∂u/∂t(t=0))
    """
    
    def __init__(self,
                 model_apply: Callable,
                 lambda_pde: float = 1.0,
                 lambda_bc: float = 10.0,
                 lambda_ic: float = 10.0,
                 lambda_data: float = 1.0,
                 bc_type: str = "dirichlet",
                 ndim: int = 2):
        """
        Initialize the physics loss calculator.
        
        Args:
            model_apply: Model's apply function (params, rng, x, return_media) -> (u, c)
            lambda_pde: Weight for PDE residual loss
            lambda_bc: Weight for boundary condition loss
            lambda_ic: Weight for initial condition loss
            lambda_data: Weight for data fitting loss (if available)
            bc_type: Type of boundary condition ("dirichlet", "neumann", "absorbing")
            ndim: Number of spatial dimensions (2 or 3)
        """
        self.model_apply = model_apply
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic
        self.lambda_data = lambda_data
        self.bc_type = bc_type
        self.ndim = ndim
    
    def _compute_derivatives(self, 
                            params: Dict,
                            rng: jax.Array,
                            x: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Compute all required derivatives using JAX autodiff.
        
        Args:
            params: Model parameters
            rng: Random key
            x: Coordinates (batch, ndim+1)
            
        Returns:
            Dictionary with u, c, and all required derivatives
        """
        # Define scalar functions for differentiation
        def u_scalar(x_single):
            """Get scalar u value at single point."""
            u, c = self.model_apply(params, rng, x_single[None, :], return_media=True)
            return u[0, 0], c[0, 0]
        
        def u_only(x_single):
            """Get only u value."""
            u, _ = u_scalar(x_single)
            return u
        
        def c_only(x_single):
            """Get only c value."""
            _, c = u_scalar(x_single)
            return c
        
        # First derivatives of u
        grad_u_fn = jax.grad(u_only)
        
        # Hessian (second derivatives) of u
        hessian_u_fn = jax.hessian(u_only)
        
        # Vectorize over batch
        def compute_single(x_single):
            u_val = u_only(x_single)
            c_val = c_only(x_single)
            
            grad_u = grad_u_fn(x_single)
            hess_u = hessian_u_fn(x_single)
            
            return {
                'u': u_val,
                'c': c_val,
                'grad_u': grad_u,
                'hess_u': hess_u,
            }
        
        # Vectorize
        batch_compute = jax.vmap(compute_single)
        return batch_compute(x)
    
    def pde_residual(self,
                     params: Dict,
                     rng: jax.Array,
                     x_interior: jnp.ndarray) -> jnp.ndarray:
        """
        Compute PDE residual: ∂²u/∂t² - c²∇²u at collocation points.
        
        Args:
            params: Model parameters
            rng: Random key
            x_interior: Interior points (batch, ndim+1)
            
        Returns:
            PDE residual values (batch,)
        """
        def u_and_c_at_point(x_single):
            """Get u and c at a single point."""
            u, c = self.model_apply(params, rng, x_single[None, :], return_media=True)
            return u[0, 0], c[0, 0]
        
        def u_at_point(x_single):
            u, _ = u_and_c_at_point(x_single)
            return u
        
        def compute_residual_single(x_single):
            """Compute residual for a single point."""
            # Get wave speed
            _, c = u_and_c_at_point(x_single)
            
            # First derivatives
            grad_u = jax.grad(u_at_point)(x_single)
            
            # Second derivatives (Hessian diagonal)
            hess_u = jax.hessian(u_at_point)(x_single)
            
            # Extract needed components
            # For 2D: x_single = [x, y, t], so indices are [0, 1, 2]
            # For 3D: x_single = [x, y, z, t], so indices are [0, 1, 2, 3]
            
            u_tt = hess_u[-1, -1]  # Time second derivative
            
            # Spatial Laplacian (sum of second spatial derivatives)
            laplacian = 0.0
            for i in range(self.ndim):
                laplacian = laplacian + hess_u[i, i]
            
            # PDE residual: u_tt - c^2 * laplacian
            residual = u_tt - c**2 * laplacian
            
            return residual
        
        # Vectorize over batch
        residuals = jax.vmap(compute_residual_single)(x_interior)
        return residuals
    
    def boundary_loss_dirichlet(self,
                                params: Dict,
                                rng: jax.Array,
                                x_boundary: jnp.ndarray,
                                u_target: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Dirichlet boundary condition: u = g on boundary.
        
        Args:
            params: Model parameters
            rng: Random key
            x_boundary: Boundary points (batch, ndim+1)
            u_target: Target values (batch,), defaults to 0
            
        Returns:
            Boundary loss (scalar)
        """
        if u_target is None:
            u_target = jnp.zeros(len(x_boundary))
        
        # Evaluate u at boundary
        u_pred = self.model_apply(params, rng, x_boundary, return_media=False)
        u_pred = u_pred.squeeze()
        
        return jnp.mean((u_pred - u_target)**2)
    
    def boundary_loss_neumann(self,
                              params: Dict,
                              rng: jax.Array,
                              x_boundary: jnp.ndarray,
                              normals: jnp.ndarray,
                              flux_target: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Neumann boundary condition: ∂u/∂n = g on boundary.
        
        Args:
            params: Model parameters
            rng: Random key
            x_boundary: Boundary points (batch, ndim+1)
            normals: Outward normal vectors (batch, ndim)
            flux_target: Target flux values (batch,), defaults to 0
            
        Returns:
            Neumann loss (scalar)
        """
        if flux_target is None:
            flux_target = jnp.zeros(len(x_boundary))
        
        def u_at_point(x_single):
            u = self.model_apply(params, rng, x_single[None, :], return_media=False)
            return u[0, 0]
        
        def normal_derivative(x_single, normal):
            grad_u = jax.grad(u_at_point)(x_single)
            # Normal derivative: ∇u · n (only spatial components)
            return jnp.dot(grad_u[:self.ndim], normal)
        
        # Compute normal derivatives
        du_dn = jax.vmap(normal_derivative)(x_boundary, normals)
        
        return jnp.mean((du_dn - flux_target)**2)
    
    def boundary_loss_absorbing(self,
                                params: Dict,
                                rng: jax.Array,
                                x_boundary: jnp.ndarray,
                                normals: jnp.ndarray) -> jnp.ndarray:
        """
        Absorbing (Sommerfeld) boundary condition: ∂u/∂t + c·∂u/∂n = 0.
        
        This condition minimizes reflections from the boundary.
        
        Args:
            params: Model parameters
            rng: Random key
            x_boundary: Boundary points (batch, ndim+1)
            normals: Outward normal vectors (batch, ndim)
            
        Returns:
            Absorbing BC loss (scalar)
        """
        def u_and_c_at_point(x_single):
            u, c = self.model_apply(params, rng, x_single[None, :], return_media=True)
            return u[0, 0], c[0, 0]
        
        def u_at_point(x_single):
            u, _ = u_and_c_at_point(x_single)
            return u
        
        def absorbing_residual(x_single, normal):
            # Get wave speed
            _, c = u_and_c_at_point(x_single)
            
            # Gradient of u
            grad_u = jax.grad(u_at_point)(x_single)
            
            u_t = grad_u[-1]  # Time derivative
            u_n = jnp.dot(grad_u[:self.ndim], normal)  # Normal derivative
            
            # Absorbing condition: u_t + c * u_n = 0
            return u_t + c * u_n
        
        residuals = jax.vmap(absorbing_residual)(x_boundary, normals)
        return jnp.mean(residuals**2)
    
    def initial_condition_loss(self,
                               params: Dict,
                               rng: jax.Array,
                               x_initial: jnp.ndarray,
                               u0_target: jnp.ndarray,
                               v0_target: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Initial condition loss: u(x, t=0) = u0, ∂u/∂t(x, t=0) = v0.
        
        Args:
            params: Model parameters
            rng: Random key
            x_initial: Initial points at t=0 (batch, ndim+1)
            u0_target: Target initial displacement (batch,)
            v0_target: Target initial velocity (batch,)
            
        Returns:
            Tuple of (loss_u0, loss_v0)
        """
        def u_at_point(x_single):
            u = self.model_apply(params, rng, x_single[None, :], return_media=False)
            return u[0, 0]
        
        def u_and_ut(x_single):
            u_val = u_at_point(x_single)
            grad_u = jax.grad(u_at_point)(x_single)
            u_t = grad_u[-1]  # Time derivative
            return u_val, u_t
        
        # Compute u and u_t at all initial points
        u_vals, ut_vals = jax.vmap(u_and_ut)(x_initial)
        
        loss_u0 = jnp.mean((u_vals - u0_target)**2)
        loss_v0 = jnp.mean((ut_vals - v0_target)**2)
        
        return loss_u0, loss_v0
    
    def data_loss(self,
                  params: Dict,
                  rng: jax.Array,
                  x_data: jnp.ndarray,
                  u_data: jnp.ndarray) -> jnp.ndarray:
        """
        Data fitting loss for observed wavefield values.
        
        Args:
            params: Model parameters
            rng: Random key
            x_data: Observation points (batch, ndim+1)
            u_data: Observed values (batch,)
            
        Returns:
            Data loss (scalar)
        """
        u_pred = self.model_apply(params, rng, x_data, return_media=False)
        u_pred = u_pred.squeeze()
        
        return jnp.mean((u_pred - u_data)**2)
    
    def total_loss(self,
                   params: Dict,
                   rng: jax.Array,
                   data_batch: Dict) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Compute total physics-informed loss.
        
        Args:
            params: Model parameters
            rng: Random key
            data_batch: Dictionary with keys:
                - 'interior': Interior collocation points
                - 'boundary': Boundary points (dict with face keys or single array)
                - 'initial': Initial condition points
                - 'u0_target': Target u at t=0 (optional)
                - 'v0_target': Target ∂u/∂t at t=0 (optional)
                - 'normals': Normal vectors for boundary (optional)
                
        Returns:
            Tuple of (total_loss, loss_dict with individual components)
        """
        losses = {}
        
        # 1. PDE residual loss
        x_interior = data_batch['interior']
        residuals = self.pde_residual(params, rng, x_interior)
        losses['loss_pde'] = jnp.mean(residuals**2)
        
        # 2. Boundary condition loss
        boundary_data = data_batch.get('boundary', {})
        if isinstance(boundary_data, dict):
            # Multiple boundary faces
            bc_losses = []
            for face_name, x_face in boundary_data.items():
                if len(x_face) > 0:
                    bc_loss = self.boundary_loss_dirichlet(params, rng, x_face)
                    bc_losses.append(bc_loss)
            losses['loss_bc'] = jnp.mean(jnp.array(bc_losses)) if bc_losses else jnp.array(0.0)
        else:
            # Single boundary array
            if len(boundary_data) > 0:
                losses['loss_bc'] = self.boundary_loss_dirichlet(params, rng, boundary_data)
            else:
                losses['loss_bc'] = jnp.array(0.0)
        
        # 3. Initial condition loss
        x_initial = data_batch.get('initial', jnp.zeros((0, self.ndim + 1)))
        u0_target = data_batch.get('u0_target', jnp.zeros(len(x_initial)))
        v0_target = data_batch.get('v0_target', jnp.zeros(len(x_initial)))
        
        if len(x_initial) > 0:
            loss_u0, loss_v0 = self.initial_condition_loss(
                params, rng, x_initial, u0_target, v0_target
            )
            losses['loss_ic_u'] = loss_u0
            losses['loss_ic_v'] = loss_v0
            losses['loss_ic'] = loss_u0 + loss_v0
        else:
            losses['loss_ic'] = jnp.array(0.0)
            losses['loss_ic_u'] = jnp.array(0.0)
            losses['loss_ic_v'] = jnp.array(0.0)
        
        # 4. Data loss (if observations available)
        if 'x_data' in data_batch and 'u_data' in data_batch:
            losses['loss_data'] = self.data_loss(
                params, rng, data_batch['x_data'], data_batch['u_data']
            )
        else:
            losses['loss_data'] = jnp.array(0.0)
        
        # Compute weighted total
        total = (
            self.lambda_pde * losses['loss_pde'] +
            self.lambda_bc * losses['loss_bc'] +
            self.lambda_ic * losses['loss_ic'] +
            self.lambda_data * losses['loss_data']
        )
        losses['loss_total'] = total
        
        return total, losses


def create_loss_fn(model_apply: Callable,
                   config: Dict) -> Callable:
    """
    Factory function to create a JIT-compiled loss function.
    
    Args:
        model_apply: Model's apply function
        config: Configuration dictionary with loss weights
        
    Returns:
        JIT-compiled loss function
    """
    loss_computer = WavePDELoss(
        model_apply=model_apply,
        lambda_pde=config.get('lambda_pde', 1.0),
        lambda_bc=config.get('lambda_bc', 10.0),
        lambda_ic=config.get('lambda_ic', 10.0),
        lambda_data=config.get('lambda_data', 1.0),
        bc_type=config.get('bc_type', 'dirichlet'),
        ndim=config.get('ndim', 2),
    )
    
    @jax.jit
    def loss_fn(params, rng, data_batch):
        return loss_computer.total_loss(params, rng, data_batch)
    
    return loss_fn


class AdaptiveLossWeights:
    """
    Adaptive loss weight balancing based on gradient magnitudes.
    
    Implements the "GradNorm" approach to balance multi-task losses.
    """
    
    def __init__(self,
                 n_losses: int = 4,
                 alpha: float = 0.12,
                 initial_weights: Optional[jnp.ndarray] = None):
        """
        Initialize adaptive weights.
        
        Args:
            n_losses: Number of loss terms
            alpha: Balancing hyperparameter
            initial_weights: Initial weights (defaults to uniform)
        """
        self.n_losses = n_losses
        self.alpha = alpha
        if initial_weights is None:
            self.weights = jnp.ones(n_losses) / n_losses
        else:
            self.weights = initial_weights
        
        self.loss_history = []
    
    def update_weights(self,
                       grads_per_loss: list,
                       current_losses: jnp.ndarray) -> jnp.ndarray:
        """
        Update weights based on gradient magnitudes.
        
        Args:
            grads_per_loss: List of gradient trees, one per loss term
            current_losses: Current loss values (n_losses,)
            
        Returns:
            Updated weights
        """
        # Compute gradient norms
        grad_norms = []
        for grads in grads_per_loss:
            norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
            grad_norms.append(norm)
        grad_norms = jnp.array(grad_norms)
        
        # Compute relative training rates
        if len(self.loss_history) > 0:
            loss_ratios = current_losses / (self.loss_history[-1] + 1e-8)
        else:
            loss_ratios = jnp.ones(self.n_losses)
        
        self.loss_history.append(current_losses)
        
        # Target gradient norm (mean)
        mean_grad_norm = jnp.mean(grad_norms * self.weights)
        
        # GradNorm update
        target_norms = mean_grad_norm * (loss_ratios ** self.alpha)
        
        # Update weights to match target
        new_weights = self.weights * (target_norms / (grad_norms + 1e-8))
        new_weights = new_weights / jnp.sum(new_weights)  # Normalize
        
        self.weights = new_weights
        return new_weights


if __name__ == "__main__":
    # Quick test
    print("Testing physics loss computation...")
    
    import sys
    sys.path.insert(0, '.')
    from model import create_model
    
    # Create model
    model = create_model()
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((1, 3))
    params = model.init(rng, dummy_input)
    
    # Create loss computer
    loss_computer = WavePDELoss(
        model_apply=model.apply,
        lambda_pde=1.0,
        lambda_bc=10.0,
        lambda_ic=10.0,
        ndim=2
    )
    
    # Create test data
    n_test = 10
    x_interior = jax.random.uniform(rng, (n_test, 3))
    x_boundary = jax.random.uniform(rng, (n_test, 3))
    x_initial = jnp.concatenate([
        jax.random.uniform(rng, (n_test, 2)),
        jnp.zeros((n_test, 1))
    ], axis=1)
    
    data_batch = {
        'interior': x_interior,
        'boundary': x_boundary,
        'initial': x_initial,
        'u0_target': jnp.zeros(n_test),
        'v0_target': jnp.zeros(n_test),
    }
    
    print("\n[1/3] Testing PDE residual computation...")
    residuals = loss_computer.pde_residual(params, rng, x_interior[:3])
    print(f"  ✓ Residuals shape: {residuals.shape}")
    print(f"  ✓ Residuals range: [{residuals.min():.6f}, {residuals.max():.6f}]")
    
    print("\n[2/3] Testing total loss computation...")
    total_loss, loss_dict = loss_computer.total_loss(params, rng, data_batch)
    print(f"  ✓ Total loss: {total_loss:.6f}")
    for key, val in loss_dict.items():
        print(f"    {key}: {val:.6f}")
    
    print("\n[3/3] Testing gradient computation...")
    def loss_for_grad(p):
        total, _ = loss_computer.total_loss(p, rng, data_batch)
        return total
    
    grads = jax.grad(loss_for_grad)(params)
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
    print(f"  ✓ Gradient norm: {grad_norm:.6f}")
    
    print("\n✅ All physics loss tests passed!")
