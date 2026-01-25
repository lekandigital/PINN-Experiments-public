"""
Synthetic Data Generation for WavePINN-NIF-Scalar.

This module generates training and validation data including:
- 2D/3D heterogeneous slowness (1/velocity) maps with random anomalies
- Ricker wavelet temporal sources
- Collocation points for PDE, boundary, and initial conditions
- Export functionality to HDF5/NPZ formats

The data generation uses JAX for on-device random number generation,
enabling efficient GPU-accelerated sampling.
"""

from typing import Dict, List, Optional, Tuple, Union
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from pathlib import Path

from .utils import save_to_hdf5, create_grid_2d, create_grid_3d

# Type aliases
Array = jax.Array
PRNGKey = jax.Array


# =============================================================================
# Slowness Map Generation
# =============================================================================

def generate_slowness_map_2d(
    nx: int,
    nz: int,
    base_velocity: float = 2000.0,
    n_anomalies: int = 5,
    anomaly_strength: float = 0.3,
    min_sigma: float = 0.05,
    max_sigma: float = 0.2,
    seed: int = 42,
    x_range: Tuple[float, float] = (0.0, 1.0),
    z_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[Array, Array, Array]:
    """
    Generate 2D slowness field (1/velocity) with random Gaussian anomalies.
    
    Creates a heterogeneous velocity model by adding Gaussian perturbations
    to a constant background velocity. Slowness = 1/velocity in s/m.
    
    Args:
        nx: Number of grid points in x-direction.
        nz: Number of grid points in z-direction.
        base_velocity: Background velocity in m/s (default 2000 m/s).
        n_anomalies: Number of random Gaussian anomalies to add.
        anomaly_strength: Relative strength of anomalies (0.3 = ±30% variation).
        min_sigma: Minimum anomaly width (normalized, 0-1).
        max_sigma: Maximum anomaly width (normalized, 0-1).
        seed: Random seed for reproducibility.
        x_range: Domain extent in x (default [0, 1]).
        z_range: Domain extent in z (default [0, 1]).
        
    Returns:
        Tuple of:
        - slowness: 2D array of shape (nx, nz) in s/m
        - velocity: 2D array of shape (nx, nz) in m/s
        - coords: Flat coordinates array of shape (nx*nz, 2)
        
    Example:
        >>> slowness, velocity, coords = generate_slowness_map_2d(100, 100, seed=42)
        >>> print(f"Velocity range: [{velocity.min():.0f}, {velocity.max():.0f}] m/s")
    """
    key = jr.PRNGKey(seed)
    
    # Create spatial grid
    x = jnp.linspace(x_range[0], x_range[1], nx)
    z = jnp.linspace(z_range[0], z_range[1], nz)
    xx, zz = jnp.meshgrid(x, z, indexing='ij')
    
    # Initialize with base velocity
    velocity = jnp.ones((nx, nz)) * base_velocity
    
    # Add Gaussian anomalies
    for i in range(n_anomalies):
        key, k1, k2, k3, k4 = jr.split(key, 5)
        
        # Random anomaly center (avoid edges)
        cx = jr.uniform(k1, (), minval=0.15, maxval=0.85)
        cz = jr.uniform(k2, (), minval=0.15, maxval=0.85)
        
        # Map to domain coordinates
        cx = x_range[0] + cx * (x_range[1] - x_range[0])
        cz = z_range[0] + cz * (z_range[1] - z_range[0])
        
        # Random amplitude (positive or negative perturbation)
        amp = jr.uniform(k3, (), minval=-anomaly_strength, maxval=anomaly_strength)
        
        # Random width
        sigma = jr.uniform(k4, (), minval=min_sigma, maxval=max_sigma)
        sigma_scaled = sigma * min(x_range[1] - x_range[0], z_range[1] - z_range[0])
        
        # Gaussian perturbation
        anomaly = amp * jnp.exp(
            -((xx - cx)**2 + (zz - cz)**2) / (2 * sigma_scaled**2)
        )
        
        velocity = velocity * (1 + anomaly)
    
    # Ensure positive velocities
    velocity = jnp.maximum(velocity, base_velocity * 0.3)
    
    # Convert to slowness
    slowness = 1.0 / velocity
    
    # Flat coordinates
    coords = jnp.stack([xx.ravel(), zz.ravel()], axis=-1)
    
    return slowness, velocity, coords


def generate_slowness_map_3d(
    nx: int,
    ny: int,
    nz: int,
    base_velocity: float = 2000.0,
    n_anomalies: int = 8,
    anomaly_strength: float = 0.3,
    seed: int = 42,
    x_range: Tuple[float, float] = (0.0, 1.0),
    y_range: Tuple[float, float] = (0.0, 1.0),
    z_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[Array, Array, Array]:
    """
    Generate 3D slowness field with random Gaussian anomalies.
    
    Args:
        nx, ny, nz: Grid resolution in each dimension.
        base_velocity: Background velocity in m/s.
        n_anomalies: Number of random anomalies.
        anomaly_strength: Relative perturbation strength.
        seed: Random seed.
        x_range, y_range, z_range: Domain extents.
        
    Returns:
        Tuple of (slowness, velocity, flat_coords).
    """
    key = jr.PRNGKey(seed)
    
    x = jnp.linspace(x_range[0], x_range[1], nx)
    y = jnp.linspace(y_range[0], y_range[1], ny)
    z = jnp.linspace(z_range[0], z_range[1], nz)
    xx, yy, zz = jnp.meshgrid(x, y, z, indexing='ij')
    
    velocity = jnp.ones((nx, ny, nz)) * base_velocity
    
    for i in range(n_anomalies):
        key, k1, k2, k3, k4, k5 = jr.split(key, 6)
        
        cx = jr.uniform(k1, (), minval=0.15, maxval=0.85)
        cy = jr.uniform(k2, (), minval=0.15, maxval=0.85)
        cz = jr.uniform(k3, (), minval=0.15, maxval=0.85)
        
        cx = x_range[0] + cx * (x_range[1] - x_range[0])
        cy = y_range[0] + cy * (y_range[1] - y_range[0])
        cz = z_range[0] + cz * (z_range[1] - z_range[0])
        
        amp = jr.uniform(k4, (), minval=-anomaly_strength, maxval=anomaly_strength)
        sigma = jr.uniform(k5, (), minval=0.05, maxval=0.15)
        
        anomaly = amp * jnp.exp(
            -((xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2) / (2 * sigma**2)
        )
        velocity = velocity * (1 + anomaly)
    
    velocity = jnp.maximum(velocity, base_velocity * 0.3)
    slowness = 1.0 / velocity
    coords = jnp.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    
    return slowness, velocity, coords


def generate_layered_velocity_2d(
    nx: int,
    nz: int,
    layer_velocities: List[float],
    layer_depths: List[float],
    x_range: Tuple[float, float] = (0.0, 1.0),
    z_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[Array, Array, Array]:
    """
    Generate 2D layered velocity model (e.g., for sedimentary structures).
    
    Args:
        nx, nz: Grid resolution.
        layer_velocities: List of velocities from top to bottom (m/s).
        layer_depths: List of layer bottom depths (normalized 0-1).
        x_range, z_range: Domain extents.
        
    Returns:
        Tuple of (slowness, velocity, coords).
        
    Example:
        >>> slowness, velocity, _ = generate_layered_velocity_2d(
        ...     100, 100,
        ...     layer_velocities=[1500, 2000, 2500, 3000],
        ...     layer_depths=[0.2, 0.4, 0.7, 1.0]
        ... )
    """
    x = jnp.linspace(x_range[0], x_range[1], nx)
    z = jnp.linspace(z_range[0], z_range[1], nz)
    _, zz = jnp.meshgrid(x, z, indexing='ij')
    
    # Normalize z to [0, 1] for depth comparison
    z_norm = (zz - z_range[0]) / (z_range[1] - z_range[0])
    
    velocity = jnp.ones((nx, nz)) * layer_velocities[-1]
    
    # Assign velocities from bottom to top
    for vel, depth in zip(reversed(layer_velocities[:-1]), reversed(layer_depths[:-1])):
        velocity = jnp.where(z_norm < depth, vel, velocity)
    
    slowness = 1.0 / velocity
    xx, zz = jnp.meshgrid(x, z, indexing='ij')
    coords = jnp.stack([xx.ravel(), zz.ravel()], axis=-1)
    
    return slowness, velocity, coords


# =============================================================================
# Ricker Wavelet Source
# =============================================================================

def ricker_wavelet(
    t: Array,
    f0: float,
    t0: Optional[float] = None,
    amplitude: float = 1.0
) -> Array:
    """
    Compute Ricker (Mexican-hat) wavelet.
    
    The Ricker wavelet is the second derivative of a Gaussian, commonly used
    as a seismic source function. It has zero DC component and a well-defined
    dominant frequency.
    
    Formula: A * (1 - 2π²f₀²(t-t₀)²) * exp(-π²f₀²(t-t₀)²)
    
    Args:
        t: Time array in seconds.
        f0: Dominant (peak) frequency in Hz.
        t0: Time shift / delay in seconds (default: 1.5/f0 for zero-phase start).
        amplitude: Peak amplitude scaling.
        
    Returns:
        Wavelet values at each time point.
        
    Example:
        >>> t = jnp.linspace(0, 0.5, 500)
        >>> wavelet = ricker_wavelet(t, f0=25.0)
        >>> print(f"Peak at t={t[jnp.argmax(jnp.abs(wavelet))]:.3f}s")
    """
    if t0 is None:
        t0 = 1.5 / f0  # Standard delay to capture the onset
    
    tau = t - t0
    a = (jnp.pi * f0 * tau) ** 2
    wavelet = amplitude * (1 - 2 * a) * jnp.exp(-a)
    
    return wavelet


def ricker_derivative(
    t: Array,
    f0: float,
    t0: Optional[float] = None,
    amplitude: float = 1.0
) -> Array:
    """
    Compute time derivative of Ricker wavelet.
    
    Useful for initial velocity conditions: ∂u/∂t(x, 0).
    
    Args:
        t: Time array in seconds.
        f0: Dominant frequency in Hz.
        t0: Time shift (default: 1.5/f0).
        amplitude: Amplitude scaling.
        
    Returns:
        Derivative values at each time point.
    """
    if t0 is None:
        t0 = 1.5 / f0
    
    tau = t - t0
    pi_f0 = jnp.pi * f0
    a = (pi_f0 * tau) ** 2
    
    # d/dt of Ricker
    derivative = amplitude * (-4 * pi_f0**2 * tau * (1 - a)) * jnp.exp(-a)
    
    return derivative


# =============================================================================
# Collocation Point Sampling
# =============================================================================

@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def sample_collocation_points_2d(
    key: PRNGKey,
    n_interior: int,
    n_boundary: int,
    n_initial: int,
    domain_bounds: Tuple[float, float, float, float, float, float]
) -> Dict[str, Array]:
    """
    Sample collocation points for 2D+time physics-informed training.
    
    Generates random points for:
    - Interior: PDE residual enforcement (x, z, t)
    - Boundary: BC enforcement on domain edges
    - Initial: IC enforcement at t=0
    
    Args:
        key: JAX random key.
        n_interior: Number of interior (PDE) points.
        n_boundary: Number of boundary points (total, split among 4 edges).
        n_initial: Number of initial condition points at t=0.
        domain_bounds: Tuple (x_min, x_max, z_min, z_max, t_min, t_max).
        
    Returns:
        Dictionary with keys:
        - 'interior': (n_interior, 3) array [x, z, t]
        - 'boundary': (n_boundary, 3) array [x, z, t] on domain edges
        - 'initial': (n_initial, 3) array [x, z, 0]
        - 'boundary_type': (n_boundary,) int array indicating edge (0-3)
    """
    x_min, x_max, z_min, z_max, t_min, t_max = domain_bounds
    
    k1, k2, k3, k4 = jr.split(key, 4)
    
    # -------------------------------------------------------------------------
    # Interior points: uniform random in (x, z, t) domain
    # -------------------------------------------------------------------------
    x_int = jr.uniform(k1, (n_interior, 1), minval=x_min, maxval=x_max)
    z_int = jr.uniform(k1, (n_interior, 1), minval=z_min, maxval=z_max)
    t_int = jr.uniform(k1, (n_interior, 1), minval=t_min, maxval=t_max)
    interior = jnp.concatenate([x_int, z_int, t_int], axis=-1)
    
    # -------------------------------------------------------------------------
    # Boundary points: on 4 edges of spatial domain, random in t
    # Edge 0: x = x_min (left)
    # Edge 1: x = x_max (right)
    # Edge 2: z = z_min (bottom)
    # Edge 3: z = z_max (top)
    # -------------------------------------------------------------------------
    n_per_edge = n_boundary // 4
    boundary_parts = []
    edge_types = []
    
    # Left edge (x = x_min)
    k2, k_edge = jr.split(k2)
    z_left = jr.uniform(k_edge, (n_per_edge,), minval=z_min, maxval=z_max)
    t_left = jr.uniform(k_edge, (n_per_edge,), minval=t_min, maxval=t_max)
    left = jnp.stack([jnp.full(n_per_edge, x_min), z_left, t_left], axis=-1)
    boundary_parts.append(left)
    edge_types.append(jnp.zeros(n_per_edge, dtype=jnp.int32))
    
    # Right edge (x = x_max)
    k2, k_edge = jr.split(k2)
    z_right = jr.uniform(k_edge, (n_per_edge,), minval=z_min, maxval=z_max)
    t_right = jr.uniform(k_edge, (n_per_edge,), minval=t_min, maxval=t_max)
    right = jnp.stack([jnp.full(n_per_edge, x_max), z_right, t_right], axis=-1)
    boundary_parts.append(right)
    edge_types.append(jnp.ones(n_per_edge, dtype=jnp.int32))
    
    # Bottom edge (z = z_min)
    k2, k_edge = jr.split(k2)
    x_bot = jr.uniform(k_edge, (n_per_edge,), minval=x_min, maxval=x_max)
    t_bot = jr.uniform(k_edge, (n_per_edge,), minval=t_min, maxval=t_max)
    bottom = jnp.stack([x_bot, jnp.full(n_per_edge, z_min), t_bot], axis=-1)
    boundary_parts.append(bottom)
    edge_types.append(jnp.full(n_per_edge, 2, dtype=jnp.int32))
    
    # Top edge (z = z_max)
    k2, k_edge = jr.split(k2)
    x_top = jr.uniform(k_edge, (n_per_edge,), minval=x_min, maxval=x_max)
    t_top = jr.uniform(k_edge, (n_per_edge,), minval=t_min, maxval=t_max)
    top = jnp.stack([x_top, jnp.full(n_per_edge, z_max), t_top], axis=-1)
    boundary_parts.append(top)
    edge_types.append(jnp.full(n_per_edge, 3, dtype=jnp.int32))
    
    boundary = jnp.concatenate(boundary_parts, axis=0)
    boundary_type = jnp.concatenate(edge_types, axis=0)
    
    # -------------------------------------------------------------------------
    # Initial points: t = t_min (usually 0)
    # -------------------------------------------------------------------------
    x_init = jr.uniform(k3, (n_initial,), minval=x_min, maxval=x_max)
    z_init = jr.uniform(k3, (n_initial,), minval=z_min, maxval=z_max)
    t_init = jnp.full(n_initial, t_min)
    initial = jnp.stack([x_init, z_init, t_init], axis=-1)
    
    return {
        'interior': interior,
        'boundary': boundary,
        'initial': initial,
        'boundary_type': boundary_type
    }


def sample_collocation_points_3d(
    key: PRNGKey,
    n_interior: int,
    n_boundary: int,
    n_initial: int,
    domain_bounds: Dict[str, Tuple[float, float]]
) -> Dict[str, Array]:
    """
    Sample collocation points for 3D+time physics-informed training.
    
    Args:
        key: JAX random key.
        n_interior: Number of interior points.
        n_boundary: Number of boundary points (on 6 faces).
        n_initial: Number of initial condition points.
        domain_bounds: Dict with 'x', 'y', 'z', 't' keys and (min, max) tuples.
        
    Returns:
        Dictionary with 'interior', 'boundary', 'initial' arrays.
    """
    x_min, x_max = domain_bounds['x']
    y_min, y_max = domain_bounds['y']
    z_min, z_max = domain_bounds['z']
    t_min, t_max = domain_bounds['t']
    
    k1, k2, k3 = jr.split(key, 3)
    
    # Interior points
    interior = jr.uniform(k1, (n_interior, 4), minval=0, maxval=1)
    interior = interior * jnp.array([
        x_max - x_min, y_max - y_min, z_max - z_min, t_max - t_min
    ])
    interior = interior + jnp.array([x_min, y_min, z_min, t_min])
    
    # Boundary points (6 faces, simplified)
    n_per_face = n_boundary // 6
    boundary_parts = []
    
    # X-faces
    for x_val in [x_min, x_max]:
        k2, k_face = jr.split(k2)
        face = jr.uniform(k_face, (n_per_face, 4))
        face = face.at[:, 0].set((x_val - x_min) / (x_max - x_min + 1e-8))
        face = face * jnp.array([x_max - x_min, y_max - y_min, z_max - z_min, t_max - t_min])
        face = face + jnp.array([x_min, y_min, z_min, t_min])
        boundary_parts.append(face)
    
    # Y-faces
    for y_val in [y_min, y_max]:
        k2, k_face = jr.split(k2)
        face = jr.uniform(k_face, (n_per_face, 4))
        face = face.at[:, 1].set((y_val - y_min) / (y_max - y_min + 1e-8))
        face = face * jnp.array([x_max - x_min, y_max - y_min, z_max - z_min, t_max - t_min])
        face = face + jnp.array([x_min, y_min, z_min, t_min])
        boundary_parts.append(face)
    
    # Z-faces
    for z_val in [z_min, z_max]:
        k2, k_face = jr.split(k2)
        face = jr.uniform(k_face, (n_per_face, 4))
        face = face.at[:, 2].set((z_val - z_min) / (z_max - z_min + 1e-8))
        face = face * jnp.array([x_max - x_min, y_max - y_min, z_max - z_min, t_max - t_min])
        face = face + jnp.array([x_min, y_min, z_min, t_min])
        boundary_parts.append(face)
    
    boundary = jnp.concatenate(boundary_parts, axis=0)
    
    # Initial points at t=0
    initial = jr.uniform(k3, (n_initial, 4))
    initial = initial.at[:, 3].set(0.0)  # t = t_min
    initial = initial * jnp.array([x_max - x_min, y_max - y_min, z_max - z_min, 0.0])
    initial = initial + jnp.array([x_min, y_min, z_min, t_min])
    
    return {
        'interior': interior,
        'boundary': boundary,
        'initial': initial
    }


# =============================================================================
# Source Term Generation
# =============================================================================

def generate_point_source_2d(
    coords: Array,
    source_location: Tuple[float, float],
    source_width: float = 0.02
) -> Array:
    """
    Generate spatial Gaussian source term for point source.
    
    Creates a smooth approximation of a delta function at the source location.
    
    Args:
        coords: (N, 2) array of (x, z) coordinates.
        source_location: (x_s, z_s) source position.
        source_width: Width of Gaussian (controls smoothness).
        
    Returns:
        (N,) array of source weights at each coordinate.
    """
    x_s, z_s = source_location
    x = coords[:, 0]
    z = coords[:, 1]
    
    r2 = (x - x_s)**2 + (z - z_s)**2
    source = jnp.exp(-r2 / (2 * source_width**2))
    
    # Normalize
    source = source / (2 * jnp.pi * source_width**2)
    
    return source


def compute_source_term(
    coords_xzt: Array,
    source_location: Tuple[float, float],
    f0: float,
    source_width: float = 0.02
) -> Array:
    """
    Compute full source term s(x, z, t) = spatial_source(x, z) * ricker(t).
    
    Args:
        coords_xzt: (N, 3) array of (x, z, t) coordinates.
        source_location: (x_s, z_s) source position.
        f0: Dominant frequency of Ricker wavelet.
        source_width: Spatial width of source.
        
    Returns:
        (N,) array of source values at each space-time point.
    """
    spatial = generate_point_source_2d(coords_xzt[:, :2], source_location, source_width)
    temporal = ricker_wavelet(coords_xzt[:, 2], f0)
    
    return spatial * temporal


# =============================================================================
# Velocity Field Interpolation
# =============================================================================

def interpolate_velocity_at_coords(
    velocity_grid: Array,
    coords: Array,
    x_range: Tuple[float, float] = (0.0, 1.0),
    z_range: Tuple[float, float] = (0.0, 1.0)
) -> Array:
    """
    Bilinear interpolation of velocity field at arbitrary coordinates.
    
    Args:
        velocity_grid: (nx, nz) velocity array.
        coords: (N, 2+) coordinates with x, z in first two columns.
        x_range, z_range: Domain bounds.
        
    Returns:
        (N,) interpolated velocity values.
    """
    nx, nz = velocity_grid.shape
    
    # Normalize coordinates to grid indices
    x = coords[:, 0]
    z = coords[:, 1]
    
    x_idx = (x - x_range[0]) / (x_range[1] - x_range[0]) * (nx - 1)
    z_idx = (z - z_range[0]) / (z_range[1] - z_range[0]) * (nz - 1)
    
    # Clip to valid range
    x_idx = jnp.clip(x_idx, 0, nx - 1.001)
    z_idx = jnp.clip(z_idx, 0, nz - 1.001)
    
    # Integer and fractional parts
    x0 = jnp.floor(x_idx).astype(jnp.int32)
    z0 = jnp.floor(z_idx).astype(jnp.int32)
    x1 = x0 + 1
    z1 = z0 + 1
    
    # Clip upper indices
    x1 = jnp.minimum(x1, nx - 1)
    z1 = jnp.minimum(z1, nz - 1)
    
    # Interpolation weights
    wx = x_idx - x0
    wz = z_idx - z0
    
    # Bilinear interpolation
    v00 = velocity_grid[x0, z0]
    v01 = velocity_grid[x0, z1]
    v10 = velocity_grid[x1, z0]
    v11 = velocity_grid[x1, z1]
    
    v = (v00 * (1 - wx) * (1 - wz) +
         v01 * (1 - wx) * wz +
         v10 * wx * (1 - wz) +
         v11 * wx * wz)
    
    return v


# =============================================================================
# Data Export
# =============================================================================

def export_training_data(
    filepath: Union[str, Path],
    slowness_map: Array,
    velocity_map: Array,
    collocation_points: Dict[str, Array],
    source_params: Dict,
    domain_bounds: Dict[str, Tuple[float, float]],
) -> None:
    """
    Export generated training data to HDF5 file.
    
    Args:
        filepath: Output file path (.h5 or .hdf5).
        slowness_map: (nx, nz) or (nx, ny, nz) slowness array.
        velocity_map: Corresponding velocity array.
        collocation_points: Dict with 'interior', 'boundary', 'initial' arrays.
        source_params: Dict with 'location', 'f0', 'width' etc.
        domain_bounds: Dict with coordinate bounds.
    """
    data = {
        'slowness': np.asarray(slowness_map),
        'velocity': np.asarray(velocity_map),
        'interior_coords': np.asarray(collocation_points['interior']),
        'boundary_coords': np.asarray(collocation_points['boundary']),
        'initial_coords': np.asarray(collocation_points['initial']),
    }
    
    if 'boundary_type' in collocation_points:
        data['boundary_type'] = np.asarray(collocation_points['boundary_type'])
    
    attrs = {
        'source_x': source_params.get('location', (0.5, 0.5))[0],
        'source_z': source_params.get('location', (0.5, 0.5))[1],
        'source_f0': source_params.get('f0', 25.0),
        'x_min': domain_bounds.get('x', (0, 1))[0],
        'x_max': domain_bounds.get('x', (0, 1))[1],
        'z_min': domain_bounds.get('z', (0, 1))[0],
        'z_max': domain_bounds.get('z', (0, 1))[1],
        't_min': domain_bounds.get('t', (0, 1))[0],
        't_max': domain_bounds.get('t', (0, 1))[1],
    }
    
    save_to_hdf5(filepath, data, attrs)


# =============================================================================
# Convenience Function: Generate Complete Training Set
# =============================================================================

def generate_training_dataset(
    seed: int = 42,
    nx: int = 100,
    nz: int = 100,
    n_interior: int = 10000,
    n_boundary: int = 2000,
    n_initial: int = 1000,
    domain_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    source_location: Tuple[float, float] = (0.5, 0.1),
    f0: float = 25.0,
    base_velocity: float = 2000.0,
    n_anomalies: int = 5,
) -> Dict:
    """
    Generate complete training dataset with slowness map and collocation points.
    
    Args:
        seed: Random seed.
        nx, nz: Grid resolution for slowness map.
        n_interior: Number of PDE collocation points.
        n_boundary: Number of BC points.
        n_initial: Number of IC points.
        domain_bounds: Coordinate bounds (default: x,z,t in [0,1]).
        source_location: Source position (x_s, z_s).
        f0: Source dominant frequency.
        base_velocity: Background velocity.
        n_anomalies: Number of velocity anomalies.
        
    Returns:
        Dictionary with all training data and metadata.
    """
    if domain_bounds is None:
        domain_bounds = {
            'x': (0.0, 1.0),
            'z': (0.0, 1.0),
            't': (0.0, 0.5)
        }
    
    key = jr.PRNGKey(seed)
    k1, k2 = jr.split(key)
    
    # Generate velocity model
    slowness, velocity, spatial_coords = generate_slowness_map_2d(
        nx=nx, nz=nz,
        base_velocity=base_velocity,
        n_anomalies=n_anomalies,
        seed=seed,
        x_range=domain_bounds['x'],
        z_range=domain_bounds['z']
    )
    
    # Sample collocation points
    bounds_tuple = (
        domain_bounds['x'][0], domain_bounds['x'][1],
        domain_bounds['z'][0], domain_bounds['z'][1],
        domain_bounds['t'][0], domain_bounds['t'][1]
    )
    
    collocation = sample_collocation_points_2d(
        k2,
        n_interior=n_interior,
        n_boundary=n_boundary,
        n_initial=n_initial,
        domain_bounds=bounds_tuple
    )
    
    # Interpolate velocity at interior points
    velocity_at_interior = interpolate_velocity_at_coords(
        velocity,
        collocation['interior'],
        x_range=domain_bounds['x'],
        z_range=domain_bounds['z']
    )
    
    return {
        'slowness': slowness,
        'velocity': velocity,
        'spatial_coords': spatial_coords,
        'collocation': collocation,
        'velocity_at_interior': velocity_at_interior,
        'source_params': {
            'location': source_location,
            'f0': f0,
            'width': 0.02
        },
        'domain_bounds': domain_bounds
    }
