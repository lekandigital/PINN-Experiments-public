"""
Reaction-Diffusion on Torus

Simulates Gray-Scott reaction-diffusion model on a torus surface.
Produces Turing-like patterns (spots, stripes) depending on parameters.

Gray-Scott equations:
    ∂U/∂t = D_u Δ_S U - UV² + F(1-U)
    ∂V/∂t = D_v Δ_S V + UV² - (F+k)V

where Δ_S is the Laplace-Beltrami operator on the torus.

Parameters (feed, kill) control pattern type:
- (0.055, 0.062): spots
- (0.03, 0.062): worms/stripes
- (0.025, 0.06): maze patterns

Output saved to HDF5 format for training.
"""

import numpy as np
import h5py
from typing import Tuple, Optional
from scipy.ndimage import laplace


def generate_torus_points(
    n_theta: int = 100,
    n_phi: int = 60,
    R: float = 1.0,  # Major radius
    r: float = 0.4   # Minor radius
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate points on a torus surface.
    
    Parametrization:
        x = (R + r cos(φ)) cos(θ)
        y = (R + r cos(φ)) sin(θ)
        z = r sin(φ)
    
    Args:
        n_theta: number of points around major circumference
        n_phi: number of points around minor circumference
        R: major radius (center of tube to center of torus)
        r: minor radius (tube radius)
        
    Returns:
        points: [N, 3] Cartesian coordinates
        normals: [N, 3] outward unit normals
    """
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    theta_flat = theta_grid.flatten()
    phi_flat = phi_grid.flatten()
    
    # Torus parametrization
    x = (R + r * np.cos(phi_flat)) * np.cos(theta_flat)
    y = (R + r * np.cos(phi_flat)) * np.sin(theta_flat)
    z = r * np.sin(phi_flat)
    
    points = np.stack([x, y, z], axis=-1)
    
    # Compute normals (gradient of implicit function)
    # For torus, normal points away from tube center
    # n = (cos(φ)cos(θ), cos(φ)sin(θ), sin(φ))
    nx = np.cos(phi_flat) * np.cos(theta_flat)
    ny = np.cos(phi_flat) * np.sin(theta_flat)
    nz = np.sin(phi_flat)
    
    normals = np.stack([nx, ny, nz], axis=-1)
    
    return points, normals, theta_flat, phi_flat


def torus_laplacian(
    field: np.ndarray,
    n_theta: int,
    n_phi: int,
    R: float,
    r: float
) -> np.ndarray:
    """
    Compute Laplace-Beltrami operator on torus using finite differences.
    
    The torus metric tensor gives:
        Δ_S f = (1/r²) ∂²f/∂φ² + (1/(R + r cos(φ))²) ∂²f/∂θ² 
                - (sin(φ))/(r(R + r cos(φ))) ∂f/∂φ
    
    Args:
        field: [n_phi, n_theta] scalar field
        n_theta, n_phi: grid dimensions
        R, r: torus radii
        
    Returns:
        laplacian: [n_phi, n_theta] Laplacian of field
    """
    dtheta = 2 * np.pi / n_theta
    dphi = 2 * np.pi / n_phi
    
    # Create phi grid for metric terms
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    # Metric coefficients
    h_theta = R + r * np.cos(phi)  # [n_phi]
    
    # Second derivative in theta (periodic)
    d2_theta = (np.roll(field, -1, axis=1) - 2 * field + np.roll(field, 1, axis=1)) / dtheta**2
    
    # Second derivative in phi (periodic)
    d2_phi = (np.roll(field, -1, axis=0) - 2 * field + np.roll(field, 1, axis=0)) / dphi**2
    
    # First derivative in phi (for connection term)
    d_phi = (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2 * dphi)
    
    # Laplacian components
    lap = np.zeros_like(field)
    
    for i in range(n_phi):
        h = h_theta[i]
        sin_phi = np.sin(phi[i])
        
        # Δf = (1/r²) ∂²f/∂φ² + (1/h²) ∂²f/∂θ² - (sin(φ)/(rh)) ∂f/∂φ
        lap[i, :] = (1 / r**2) * d2_phi[i, :] + \
                    (1 / h**2) * d2_theta[i, :] - \
                    (sin_phi / (r * h)) * d_phi[i, :]
    
    return lap


def simulate_gray_scott(
    n_theta: int = 100,
    n_phi: int = 60,
    R: float = 1.0,
    r: float = 0.4,
    feed: float = 0.055,
    kill: float = 0.062,
    D_u: float = 0.2,
    D_v: float = 0.1,
    dt: float = 0.1,
    n_steps: int = 200,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate Gray-Scott reaction-diffusion on torus.
    
    Args:
        n_theta, n_phi: grid dimensions
        R, r: torus radii
        feed, kill: Gray-Scott parameters
        D_u, D_v: diffusion coefficients
        dt: time step
        n_steps: number of simulation steps
        seed: random seed for initial perturbation
        
    Returns:
        U: [n_phi, n_theta] final U concentration
        V: [n_phi, n_theta] final V concentration
    """
    np.random.seed(seed)
    
    # Initialize
    U = np.ones((n_phi, n_theta))
    V = np.zeros((n_phi, n_theta))
    
    # Seed with random perturbation in center region
    center_phi = n_phi // 2
    center_theta = n_theta // 2
    size = min(n_phi, n_theta) // 5
    
    phi_lo, phi_hi = center_phi - size, center_phi + size
    theta_lo, theta_hi = center_theta - size, center_theta + size
    
    U[phi_lo:phi_hi, theta_lo:theta_hi] = 0.5 + 0.1 * np.random.rand(2*size, 2*size)
    V[phi_lo:phi_hi, theta_lo:theta_hi] = 0.25 + 0.1 * np.random.rand(2*size, 2*size)
    
    # Time stepping
    for step in range(n_steps):
        # Compute Laplacians
        lap_U = torus_laplacian(U, n_theta, n_phi, R, r)
        lap_V = torus_laplacian(V, n_theta, n_phi, R, r)
        
        # Reaction terms
        UVV = U * V * V
        
        # Gray-Scott update
        dU = D_u * lap_U - UVV + feed * (1 - U)
        dV = D_v * lap_V + UVV - (feed + kill) * V
        
        U += dt * dU
        V += dt * dV
        
        # Clamp values
        U = np.clip(U, 0, 1)
        V = np.clip(V, 0, 1)
        
        if step % 50 == 0:
            print(f"Step {step}/{n_steps}, U: [{U.min():.3f}, {U.max():.3f}], "
                  f"V: [{V.min():.3f}, {V.max():.3f}]")
    
    return U, V


def generate_torus_reaction_diffusion_data(
    n_theta: int = 100,
    n_phi: int = 60,
    R: float = 1.0,
    r: float = 0.4,
    feed: float = 0.055,
    kill: float = 0.062,
    D_u: float = 0.2,
    D_v: float = 0.1,
    dt: float = 0.1,
    n_steps: int = 200,
    save_path: Optional[str] = None
) -> dict:
    """
    Generate reaction-diffusion data on torus.
    
    Args:
        n_theta, n_phi: grid dimensions
        R, r: torus radii
        feed, kill: Gray-Scott parameters
        D_u, D_v: diffusion coefficients
        dt: time step
        n_steps: simulation steps
        save_path: if provided, save to HDF5
        
    Returns:
        dict with geometry and concentration data
    """
    # Generate torus geometry
    points, normals, theta, phi = generate_torus_points(n_theta, n_phi, R, r)
    
    # Simulate Gray-Scott
    print(f"Simulating Gray-Scott on torus ({n_theta}x{n_phi})...")
    U, V = simulate_gray_scott(
        n_theta, n_phi, R, r,
        feed, kill, D_u, D_v, dt, n_steps
    )
    
    # Flatten concentration fields to match point ordering
    U_flat = U.flatten()
    V_flat = V.flatten()
    
    # Stack as [N, 2] concentration field
    concentration = np.stack([U_flat, V_flat], axis=-1)
    
    result = {
        'collocation_points': points.astype(np.float32),
        'normals': normals.astype(np.float32),
        'theta': theta.astype(np.float32),
        'phi': phi.astype(np.float32),
        'solution_conc': concentration.astype(np.float32),
        'U_grid': U.astype(np.float32),
        'V_grid': V.astype(np.float32),
        'params': {
            'n_theta': n_theta,
            'n_phi': n_phi,
            'R': R,
            'r': r,
            'feed': feed,
            'kill': kill,
            'D_u': D_u,
            'D_v': D_v,
            'dt': dt,
            'n_steps': n_steps
        }
    }
    
    # Save to HDF5
    if save_path is not None:
        with h5py.File(save_path, 'w') as f:
            for key in ['collocation_points', 'normals', 'theta', 'phi',
                       'solution_conc', 'U_grid', 'V_grid']:
                f.create_dataset(key, data=result[key])
            
            params = f.create_group('params')
            for k, v in result['params'].items():
                params.attrs[k] = v
        
        print(f"Saved reaction-diffusion data to {save_path}")
    
    return result


if __name__ == '__main__':
    # Generate sample data
    data = generate_torus_reaction_diffusion_data(
        n_theta=100,
        n_phi=60,
        n_steps=200,
        save_path='torus_reacdiff_data.h5'
    )
    print(f"Generated {data['collocation_points'].shape[0]} points")
    print(f"Concentration range U: [{data['solution_conc'][:, 0].min():.3f}, "
          f"{data['solution_conc'][:, 0].max():.3f}]")
    print(f"Concentration range V: [{data['solution_conc'][:, 1].min():.3f}, "
          f"{data['solution_conc'][:, 1].max():.3f}]")
