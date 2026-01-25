"""
Shallow Water Equations (SWE) Data Generator on Sphere

Generates synthetic data for the shallow water equations on a unit sphere:
- Height field h with Legendre polynomial perturbation
- Velocity field (u, v) from solid-body rotation
- Coriolis parameter f = 2Ω sin(φ)

Output saved to HDF5 format for training.

References:
- Williamson et al., "A standard test set for numerical approximations 
  to the shallow water equations in spherical geometry"
"""

import numpy as np
import h5py
from typing import Tuple, Optional
from scipy.special import sph_harm


def sample_sphere_points(n_points: int, method: str = 'fibonacci') -> np.ndarray:
    """
    Sample points uniformly on unit sphere.
    
    Args:
        n_points: number of points to sample
        method: sampling method ('fibonacci', 'random', 'grid')
        
    Returns:
        points: [n_points, 3] Cartesian coordinates on unit sphere
    """
    if method == 'fibonacci':
        # Fibonacci spiral for quasi-uniform distribution
        indices = np.arange(n_points, dtype=float) + 0.5
        
        phi = np.arccos(1 - 2 * indices / n_points)
        theta = np.pi * (1 + 5**0.5) * indices
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        points = np.stack([x, y, z], axis=-1)
        
    elif method == 'random':
        # Random uniform sampling
        theta = 2 * np.pi * np.random.rand(n_points)
        phi = np.arccos(2 * np.random.rand(n_points) - 1)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        points = np.stack([x, y, z], axis=-1)
        
    elif method == 'grid':
        # Latitude-longitude grid (has polar clustering)
        n_lat = int(np.sqrt(n_points / 2))
        n_lon = 2 * n_lat
        
        lats = np.linspace(-np.pi/2, np.pi/2, n_lat)
        lons = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
        
        lat_grid, lon_grid = np.meshgrid(lats, lons)
        lat_flat = lat_grid.flatten()
        lon_flat = lon_grid.flatten()
        
        x = np.cos(lat_flat) * np.cos(lon_flat)
        y = np.cos(lat_flat) * np.sin(lon_flat)
        z = np.sin(lat_flat)
        
        points = np.stack([x, y, z], axis=-1)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return points


def cartesian_to_spherical(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates to spherical (latitude, longitude).
    
    Args:
        points: [N, 3] Cartesian coordinates
        
    Returns:
        lat: [N] latitude in radians (-π/2 to π/2)
        lon: [N] longitude in radians (0 to 2π)
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    lat = np.arcsin(np.clip(z, -1, 1))
    lon = np.arctan2(y, x) % (2 * np.pi)
    
    return lat, lon


def compute_sphere_normals(points: np.ndarray) -> np.ndarray:
    """
    Compute outward unit normals on unit sphere.
    For unit sphere, normal = position.
    
    Args:
        points: [N, 3] points on unit sphere
        
    Returns:
        normals: [N, 3] unit normal vectors
    """
    # Normalize just in case points aren't exactly on sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / (norms + 1e-10)


def legendre_p2(x: np.ndarray) -> np.ndarray:
    """
    Legendre polynomial P_2(x) = (3x² - 1) / 2
    
    Args:
        x: input values
        
    Returns:
        P_2(x)
    """
    return 0.5 * (3 * x**2 - 1)


def generate_sphere_swe_data(
    n_points: int = 5000,
    H0: float = 1000.0,  # Mean height (meters)
    h_amp: float = 100.0,  # Height perturbation amplitude
    U0: float = 10.0,  # Solid body rotation velocity (m/s)
    omega: float = 7.292e-5,  # Earth's rotation rate (rad/s)
    save_path: Optional[str] = None
) -> dict:
    """
    Generate shallow water equations data on unit sphere.
    
    The solution is a steady-state solid-body rotation with
    geostrophically balanced height field.
    
    Initial conditions:
    - Height: h = H0 + h_amp * P_2(sin(lat))
    - Velocity: Solid body rotation (u_lat = U0 * cos(lat), u_lon = 0)
    
    Args:
        n_points: number of collocation points
        H0: mean height in meters
        h_amp: height perturbation amplitude
        U0: rotation velocity at equator
        omega: planetary rotation rate
        save_path: if provided, save to HDF5 file
        
    Returns:
        dict with:
            - collocation_points: [N, 3]
            - initial_condition: [N, 4] (h, vx, vy, vz)
            - coriolis: [N] Coriolis parameter
            - normals: [N, 3]
    """
    # Sample points on sphere
    points = sample_sphere_points(n_points, method='fibonacci')
    
    # Get spherical coordinates
    lat, lon = cartesian_to_spherical(points)
    
    # Compute normals (same as points for unit sphere)
    normals = compute_sphere_normals(points)
    
    # ========================
    # Height field
    # ========================
    # h = H0 + h_amp * P_2(sin(lat))
    sin_lat = np.sin(lat)
    h = H0 + h_amp * legendre_p2(sin_lat)
    
    # ========================
    # Velocity field (solid body rotation)
    # ========================
    # In spherical: u_lat = U0 * cos(lat), u_lon = 0
    # Convert to Cartesian velocity
    
    # Rotation axis is z-axis
    # v = ω × r where ω = (0, 0, U0) in angular velocity sense
    # For solid body rotation: v_tangent = U0 * cos(lat) * e_lon
    
    cos_lat = np.cos(lat)
    
    # Longitudinal unit vector in Cartesian: e_lon = (-sin(lon), cos(lon), 0)
    e_lon = np.zeros_like(points)
    e_lon[:, 0] = -np.sin(lon)
    e_lon[:, 1] = np.cos(lon)
    e_lon[:, 2] = 0.0
    
    # Velocity vector
    v_mag = U0 * cos_lat
    velocity = v_mag[:, np.newaxis] * e_lon
    
    # ========================
    # Coriolis parameter
    # ========================
    # f = 2Ω sin(lat)
    coriolis = 2 * omega * sin_lat
    
    # ========================
    # Package data
    # ========================
    # Initial condition: [h, vx, vy, vz]
    initial_condition = np.zeros((n_points, 4))
    initial_condition[:, 0] = h
    initial_condition[:, 1:4] = velocity
    
    result = {
        'collocation_points': points.astype(np.float32),
        'initial_condition': initial_condition.astype(np.float32),
        'solution': initial_condition.astype(np.float32),  # Steady state
        'coriolis': coriolis.astype(np.float32),
        'normals': normals.astype(np.float32),
        'lat': lat.astype(np.float32),
        'lon': lon.astype(np.float32),
        'params': {
            'H0': H0,
            'h_amp': h_amp,
            'U0': U0,
            'omega': omega,
            'n_points': n_points
        }
    }
    
    # Save to HDF5 if requested
    if save_path is not None:
        with h5py.File(save_path, 'w') as f:
            for key in ['collocation_points', 'initial_condition', 'solution',
                       'coriolis', 'normals', 'lat', 'lon']:
                f.create_dataset(key, data=result[key])
            
            # Save parameters as attributes
            params = f.create_group('params')
            for k, v in result['params'].items():
                params.attrs[k] = v
        
        print(f"Saved SWE data to {save_path}")
    
    return result


def compute_swe_residual(
    h: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    coriolis: np.ndarray,
    g: float = 9.81
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute shallow water equation residuals (for verification).
    
    SWE in spherical coordinates:
    ∂h/∂t + ∇·(hu) = 0
    ∂u/∂t + u·∇u + fk×u + g∇h = 0
    
    For steady state, all time derivatives = 0.
    
    Returns residuals for continuity and momentum equations.
    """
    # This would require finite differences on sphere
    # Placeholder for full implementation
    raise NotImplementedError("Full SWE residual computation requires spherical FD stencils")


if __name__ == '__main__':
    # Generate sample data
    data = generate_sphere_swe_data(
        n_points=5000,
        save_path='sphere_swe_data.h5'
    )
    print(f"Generated {data['collocation_points'].shape[0]} points")
    print(f"Height range: [{data['initial_condition'][:, 0].min():.1f}, "
          f"{data['initial_condition'][:, 0].max():.1f}]")
