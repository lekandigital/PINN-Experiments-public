"""
SurfPINN Physics-Informed Loss Functions
=========================================
Implements physics constraints for free-surface flow:
- Navier-Stokes residuals (continuity, momentum)
- Incompressibility constraint (∇·u = 0)
- Kinematic free-surface condition
- Mean curvature smoothing for surface realism

All losses computed via JAX automatic differentiation or finite differences.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Dict, NamedTuple
from functools import partial


class PhysicsConfig(NamedTuple):
    """Configuration for physics loss weights."""
    weight_data: float = 1.0       # Data fitting loss
    weight_continuity: float = 0.1  # ∇·u = 0
    weight_momentum: float = 0.1    # Navier-Stokes momentum
    weight_kinematic: float = 0.1   # Free surface kinematic condition
    weight_curvature: float = 0.01  # Mean curvature smoothing
    gravity: float = 9.81           # Gravitational acceleration
    viscosity: float = 1e-6         # Kinematic viscosity (water)
    density: float = 1000.0         # Fluid density (water)


# ==============================================================================
# Finite Difference Operators
# ==============================================================================

def gradient_x(f: jnp.ndarray, dx: float = 1.0) -> jnp.ndarray:
    """
    Compute ∂f/∂x using central differences.
    
    Args:
        f: Field of shape (..., Nx, Ny) or (..., Nx, Ny, 1)
        dx: Grid spacing in x direction
    
    Returns:
        Gradient ∂f/∂x with same shape as input
    """
    # Squeeze trailing dimension if present
    squeeze = f.ndim > 2 and f.shape[-1] == 1
    if squeeze:
        f = jnp.squeeze(f, axis=-1)
    
    # Central difference: (f[i+1] - f[i-1]) / (2*dx)
    grad = jnp.zeros_like(f)
    grad = grad.at[..., 1:-1, :].set(
        (f[..., 2:, :] - f[..., :-2, :]) / (2 * dx)
    )
    # Forward/backward difference at boundaries
    grad = grad.at[..., 0, :].set((f[..., 1, :] - f[..., 0, :]) / dx)
    grad = grad.at[..., -1, :].set((f[..., -1, :] - f[..., -2, :]) / dx)
    
    if squeeze:
        grad = grad[..., None]
    return grad


def gradient_y(f: jnp.ndarray, dy: float = 1.0) -> jnp.ndarray:
    """
    Compute ∂f/∂y using central differences.
    
    Args:
        f: Field of shape (..., Nx, Ny) or (..., Nx, Ny, 1)
        dy: Grid spacing in y direction
    
    Returns:
        Gradient ∂f/∂y with same shape as input
    """
    squeeze = f.ndim > 2 and f.shape[-1] == 1
    if squeeze:
        f = jnp.squeeze(f, axis=-1)
    
    grad = jnp.zeros_like(f)
    grad = grad.at[..., :, 1:-1].set(
        (f[..., :, 2:] - f[..., :, :-2]) / (2 * dy)
    )
    grad = grad.at[..., :, 0].set((f[..., :, 1] - f[..., :, 0]) / dy)
    grad = grad.at[..., :, -1].set((f[..., :, -1] - f[..., :, -2]) / dy)
    
    if squeeze:
        grad = grad[..., None]
    return grad


def laplacian_2d(f: jnp.ndarray, dx: float = 1.0, dy: float = 1.0) -> jnp.ndarray:
    """
    Compute Laplacian ∇²f = ∂²f/∂x² + ∂²f/∂y² using finite differences.
    
    Args:
        f: Field of shape (..., Nx, Ny) or (..., Nx, Ny, 1)
        dx, dy: Grid spacing
    
    Returns:
        Laplacian with same shape as input
    """
    squeeze = f.ndim > 2 and f.shape[-1] == 1
    if squeeze:
        f = jnp.squeeze(f, axis=-1)
    
    lap = jnp.zeros_like(f)
    
    # Interior points: standard 5-point stencil
    d2f_dx2 = (f[..., 2:, 1:-1] - 2*f[..., 1:-1, 1:-1] + f[..., :-2, 1:-1]) / (dx**2)
    d2f_dy2 = (f[..., 1:-1, 2:] - 2*f[..., 1:-1, 1:-1] + f[..., 1:-1, :-2]) / (dy**2)
    
    lap = lap.at[..., 1:-1, 1:-1].set(d2f_dx2 + d2f_dy2)
    
    if squeeze:
        lap = lap[..., None]
    return lap


def divergence_2d(u: jnp.ndarray, v: jnp.ndarray, dx: float = 1.0, dy: float = 1.0) -> jnp.ndarray:
    """
    Compute divergence ∇·u = ∂u/∂x + ∂v/∂y.
    
    Args:
        u: x-velocity component
        v: y-velocity component
        dx, dy: Grid spacing
    
    Returns:
        Divergence field
    """
    du_dx = gradient_x(u, dx)
    dv_dy = gradient_y(v, dy)
    return du_dx + dv_dy


# ==============================================================================
# Physics Loss Components
# ==============================================================================

def continuity_loss(
    u: jnp.ndarray,
    v: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0
) -> jnp.ndarray:
    """
    Incompressibility constraint: ∇·u = 0.
    
    For incompressible flow, the divergence of velocity must vanish.
    
    Args:
        u: x-velocity field (batch, Nx, Ny) or (batch, Nx, Ny, 1)
        v: y-velocity field (batch, Nx, Ny) or (batch, Nx, Ny, 1)
        dx, dy: Grid spacing
    
    Returns:
        Scalar loss value
    """
    div = divergence_2d(u, v, dx, dy)
    return jnp.mean(div ** 2)


def momentum_loss_shallow_water(
    h: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    dh_dt: jnp.ndarray,
    du_dt: jnp.ndarray,
    dv_dt: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    g: float = 9.81,
    nu: float = 1e-6
) -> jnp.ndarray:
    """
    Shallow water equations momentum residual.
    
    Momentum equations:
        ∂u/∂t + u·∇u = -g·∂h/∂x + ν·∇²u
        ∂v/∂t + u·∇v = -g·∂h/∂y + ν·∇²v
    
    Continuity:
        ∂h/∂t + ∇·(h·u) = 0
    
    Args:
        h: Height field
        u, v: Velocity components
        dh_dt, du_dt, dv_dt: Time derivatives
        dx, dy: Grid spacing
        g: Gravity
        nu: Viscosity
    
    Returns:
        Scalar momentum loss
    """
    # Spatial gradients
    dh_dx = gradient_x(h, dx)
    dh_dy = gradient_y(h, dy)
    du_dx = gradient_x(u, dx)
    du_dy = gradient_y(u, dy)
    dv_dx = gradient_x(v, dx)
    dv_dy = gradient_y(v, dy)
    
    # Laplacians for viscosity
    lap_u = laplacian_2d(u, dx, dy)
    lap_v = laplacian_2d(v, dx, dy)
    
    # Convective terms
    u_dot_grad_u = u * du_dx + v * du_dy
    u_dot_grad_v = u * dv_dx + v * dv_dy
    
    # Momentum residuals
    res_u = du_dt + u_dot_grad_u + g * dh_dx - nu * lap_u
    res_v = dv_dt + u_dot_grad_v + g * dh_dy - nu * lap_v
    
    # Height (mass) conservation residual
    d_hu_dx = gradient_x(h * u, dx)
    d_hv_dy = gradient_y(h * v, dy)
    res_h = dh_dt + d_hu_dx + d_hv_dy
    
    return jnp.mean(res_u**2 + res_v**2 + res_h**2)


def kinematic_surface_loss(
    h: jnp.ndarray,
    w: jnp.ndarray,
    dh_dt: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0
) -> jnp.ndarray:
    """
    Kinematic free-surface boundary condition.
    
    The free surface is a material surface:
        ∂h/∂t + u·∂h/∂x + v·∂h/∂y = w
    
    where w is the vertical velocity component at the surface.
    
    Args:
        h: Height field
        w: Vertical velocity at surface
        dh_dt: Time derivative of height
        u, v: Horizontal velocity components
        dx, dy: Grid spacing
    
    Returns:
        Scalar kinematic condition loss
    """
    dh_dx = gradient_x(h, dx)
    dh_dy = gradient_y(h, dy)
    
    # Kinematic condition residual
    residual = dh_dt + u * dh_dx + v * dh_dy - w
    
    return jnp.mean(residual ** 2)


def mean_curvature_loss(height: jnp.ndarray, dx: float = 1.0, dy: float = 1.0) -> jnp.ndarray:
    """
    Mean curvature smoothing loss for surface realism.
    
    For a height field h(x,y), the mean curvature is:
        κ = ∇·(∇h / √(1 + |∇h|²))
    
    Expanded:
        κ = [(1 + h_y²)h_xx - 2h_x·h_y·h_xy + (1 + h_x²)h_yy] / (1 + h_x² + h_y²)^(3/2)
    
    Penalizing κ² encourages smooth surfaces without artificial wrinkles.
    
    Args:
        height: Height field of shape (..., Nx, Ny, 1) or (..., Nx, Ny)
        dx, dy: Grid spacing
    
    Returns:
        Scalar curvature loss
    """
    # Squeeze trailing dimension
    h = jnp.squeeze(height, axis=-1) if height.shape[-1] == 1 else height
    
    # First derivatives (central differences)
    h_x = jnp.zeros_like(h)
    h_y = jnp.zeros_like(h)
    
    h_x = h_x.at[..., 1:-1, :].set((h[..., 2:, :] - h[..., :-2, :]) / (2 * dx))
    h_y = h_y.at[..., :, 1:-1].set((h[..., :, 2:] - h[..., :, :-2]) / (2 * dy))
    
    # Second derivatives
    h_xx = jnp.zeros_like(h)
    h_yy = jnp.zeros_like(h)
    h_xy = jnp.zeros_like(h)
    
    h_xx = h_xx.at[..., 1:-1, :].set(
        (h[..., 2:, :] - 2*h[..., 1:-1, :] + h[..., :-2, :]) / (dx**2)
    )
    h_yy = h_yy.at[..., :, 1:-1].set(
        (h[..., :, 2:] - 2*h[..., :, 1:-1] + h[..., :, :-2]) / (dy**2)
    )
    
    # Mixed derivative (interior only)
    h_xy = h_xy.at[..., 1:-1, 1:-1].set(
        (h[..., 2:, 2:] - h[..., 2:, :-2] - h[..., :-2, 2:] + h[..., :-2, :-2]) / (4 * dx * dy)
    )
    
    # Mean curvature formula
    grad_sq = 1.0 + h_x**2 + h_y**2
    numerator = (1 + h_y**2) * h_xx - 2 * h_x * h_y * h_xy + (1 + h_x**2) * h_yy
    denominator = jnp.power(grad_sq, 1.5) + 1e-8  # Avoid division by zero
    
    kappa = numerator / denominator
    
    # Only compute loss in interior where derivatives are valid
    kappa_interior = kappa[..., 1:-1, 1:-1]
    
    return jnp.mean(kappa_interior ** 2)


# ==============================================================================
# Combined Loss Functions
# ==============================================================================

def data_loss(
    height_pred: jnp.ndarray,
    height_true: jnp.ndarray,
    velocity_pred: jnp.ndarray,
    velocity_true: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Data fitting loss (MSE).
    
    Args:
        height_pred: Predicted height field
        height_true: Ground truth height
        velocity_pred: Predicted velocities
        velocity_true: Ground truth velocities
    
    Returns:
        Tuple of (height_loss, velocity_loss)
    """
    height_loss = jnp.mean((height_pred - height_true) ** 2)
    velocity_loss = jnp.mean((velocity_pred - velocity_true) ** 2)
    return height_loss, velocity_loss


def total_physics_loss(
    height_pred: jnp.ndarray,
    velocity_pred: jnp.ndarray,
    height_true: jnp.ndarray,
    velocity_true: jnp.ndarray,
    config: PhysicsConfig = PhysicsConfig(),
    dx: float = 1.0,
    dy: float = 1.0,
    dt: float = 1.0
) -> Dict[str, jnp.ndarray]:
    """
    Compute total physics-informed loss.
    
    Combines:
    - Data loss (MSE on height and velocity)
    - Continuity loss (incompressibility)
    - Curvature smoothing loss
    
    Args:
        height_pred: Predicted height (batch, Nx, Ny, 1)
        velocity_pred: Predicted velocity (batch, N_particles, 3) or (batch, Nx, Ny, 2)
        height_true: Ground truth height
        velocity_true: Ground truth velocity
        config: Physics configuration
        dx, dy, dt: Grid/time spacing
    
    Returns:
        Dictionary of loss components and total loss
    """
    losses = {}
    
    # Data fitting loss
    h_loss, v_loss = data_loss(height_pred, height_true, velocity_pred, velocity_true)
    losses['height_mse'] = h_loss
    losses['velocity_mse'] = v_loss
    losses['data'] = config.weight_data * (h_loss + v_loss)
    
    # Curvature smoothing
    curv_loss = mean_curvature_loss(height_pred, dx, dy)
    losses['curvature'] = config.weight_curvature * curv_loss
    
    # Continuity (if velocity is grid-based)
    if velocity_pred.ndim >= 3 and velocity_pred.shape[-1] >= 2:
        u = velocity_pred[..., 0]
        v = velocity_pred[..., 1]
        cont_loss = continuity_loss(u, v, dx, dy)
        losses['continuity'] = config.weight_continuity * cont_loss
    else:
        losses['continuity'] = jnp.array(0.0)
    
    # Total loss
    losses['total'] = losses['data'] + losses['curvature'] + losses['continuity']
    
    return losses


# ==============================================================================
# Lagrangian Physics
# ==============================================================================

def lagrangian_acceleration_loss(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    predicted_velocities: jnp.ndarray,
    dt: float = 1.0,
    g: float = 9.81
) -> jnp.ndarray:
    """
    Physics loss for Lagrangian particles.
    
    Particles should follow Newton's laws:
        dv/dt = -g·ẑ + (pressure/viscous forces)
    
    For free-surface particles, primarily gravity acts.
    
    Args:
        positions: Particle positions over time (batch, N, T, 3)
        velocities: Particle velocities over time (batch, N, T, 3)
        predicted_velocities: Model predictions (batch, N, 3)
        dt: Time step
        g: Gravity
    
    Returns:
        Physics consistency loss
    """
    # Simple consistency: predicted velocity should match trajectory derivative
    if positions.ndim >= 3 and positions.shape[-2] > 1:
        # Finite difference velocity from positions
        dx_dt = (positions[..., 1:, :] - positions[..., :-1, :]) / dt
        
        # Compare with predicted
        vel_error = jnp.mean((predicted_velocities - dx_dt[..., -1, :]) ** 2)
        
        # Gravity term for z-acceleration
        if velocities.ndim >= 3 and velocities.shape[-2] > 1:
            dv_dt = (velocities[..., 1:, :] - velocities[..., :-1, :]) / dt
            gravity_residual = dv_dt[..., -1, 2] + g  # a_z should be -g
            gravity_loss = jnp.mean(gravity_residual ** 2)
        else:
            gravity_loss = jnp.array(0.0)
        
        return vel_error + 0.1 * gravity_loss
    
    return jnp.array(0.0)


# ==============================================================================
# Metrics
# ==============================================================================

def compute_psnr(pred: jnp.ndarray, true: jnp.ndarray) -> jnp.ndarray:
    """
    Peak Signal-to-Noise Ratio for height field accuracy.
    
    PSNR = 20 * log10(MAX / sqrt(MSE))
    
    Args:
        pred: Predicted field
        true: Ground truth field
    
    Returns:
        PSNR in dB
    """
    mse = jnp.mean((pred - true) ** 2)
    max_val = jnp.max(true) - jnp.min(true)
    psnr = 20 * jnp.log10(max_val / (jnp.sqrt(mse) + 1e-8))
    return psnr


def compute_rmse(pred: jnp.ndarray, true: jnp.ndarray) -> jnp.ndarray:
    """Root Mean Square Error."""
    return jnp.sqrt(jnp.mean((pred - true) ** 2))


def compute_relative_l2(pred: jnp.ndarray, true: jnp.ndarray) -> jnp.ndarray:
    """Relative L2 error: ||pred - true||_2 / ||true||_2"""
    return jnp.linalg.norm(pred - true) / (jnp.linalg.norm(true) + 1e-8)
