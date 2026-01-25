"""
SurfPINN Synthetic Data Generator
=================================
Generates synthetic dam-break scenarios for training and testing.
Produces both Eulerian height fields and Lagrangian particle trajectories.

Physics: Shallow water equations approximation for dam break
- Initial height: h₀ = 0.8 for x < 0.3, else h₀ = 0.2
- Domain: [0,1] × [0,1] spatial, t ∈ [0,1] temporal
"""

import numpy as np
import h5py
from typing import Tuple, Dict
from pathlib import Path


def shallow_water_dam_break(
    x: np.ndarray, 
    t: float, 
    h_left: float = 0.8, 
    h_right: float = 0.2,
    x_dam: float = 0.3,
    g: float = 9.81
) -> np.ndarray:
    """
    Analytical approximation of dam break using shallow water equations.
    
    The Ritter solution for dam break on dry bed:
    - Rarefaction wave propagates leftward at speed -√(gh₀)
    - Shock front propagates rightward
    
    Args:
        x: Spatial coordinates (1D array)
        t: Time value
        h_left: Initial water height on left (dam side)
        h_right: Initial water height on right
        x_dam: Position of dam
        g: Gravitational acceleration
    
    Returns:
        Height field h(x, t)
    """
    # Wave speeds
    c0 = np.sqrt(g * h_left)  # Initial wave speed
    
    # Avoid division by zero at t=0
    t = max(t, 1e-6)
    
    # Simplified dam break profile (smoothed Ritter solution)
    h = np.zeros_like(x)
    
    for i, xi in enumerate(x):
        # Position relative to dam
        x_rel = (xi - x_dam) / (c0 * t + 1e-6)
        
        if xi < x_dam - c0 * t:
            # Undisturbed region (left of rarefaction)
            h[i] = h_left
        elif xi > x_dam + 2 * c0 * t:
            # Undisturbed dry/shallow region (right)
            h[i] = h_right
        else:
            # Rarefaction fan region
            # Smooth transition using tanh
            transition = 0.5 * (1 - np.tanh(3 * x_rel))
            h[i] = h_right + (h_left - h_right) * transition
    
    return h


def generate_velocity_field(
    height: np.ndarray,
    dx: float,
    g: float = 9.81
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute velocity field from height using shallow water momentum.
    
    u ≈ √(g * h) in flow direction (simplified)
    Velocity is derived from height gradient: u ~ -∂h/∂x
    
    Args:
        height: 2D height field (Nx, Ny)
        dx: Grid spacing
        g: Gravitational acceleration
    
    Returns:
        (u, v): Velocity components in x and y directions
    """
    # Compute height gradients
    dh_dx = np.gradient(height, dx, axis=1)
    dh_dy = np.gradient(height, dx, axis=0)
    
    # Velocity proportional to negative gradient (flow downhill)
    # Scale by wave speed for physical consistency
    c = np.sqrt(g * np.maximum(height, 0.01))
    
    u = -c * dh_dx / (np.abs(dh_dx).max() + 1e-6)
    v = -c * dh_dy / (np.abs(dh_dy).max() + 1e-6)
    
    return u, v


def generate_eulerian_data(
    nx: int = 64,
    ny: int = 64,
    nt: int = 16,
    domain: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    t_max: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Generate Eulerian height field data for dam break scenario.
    
    Args:
        nx, ny: Grid resolution in x and y
        nt: Number of time steps
        domain: (xmin, xmax, ymin, ymax)
        t_max: Maximum simulation time
    
    Returns:
        Dictionary with:
        - 'grid_coords': (nx, ny, nt, 3) spacetime coordinates
        - 'height': (nx, ny, nt, 1) water height
        - 'velocity': (nx, ny, nt, 2) velocity field (u, v)
    """
    xmin, xmax, ymin, ymax = domain
    
    # Create spatial grid
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    t = np.linspace(0, t_max, nt)
    
    dx = x[1] - x[0]
    
    # Initialize output arrays
    grid_coords = np.zeros((nx, ny, nt, 3), dtype=np.float32)
    height = np.zeros((nx, ny, nt, 1), dtype=np.float32)
    velocity = np.zeros((nx, ny, nt, 2), dtype=np.float32)
    
    # Generate data for each time step
    for ti, t_val in enumerate(t):
        # Height profile (extends 1D solution to 2D with slight y-variation)
        for yi, y_val in enumerate(y):
            # Add small y-dependency for 2D structure
            y_factor = 1.0 + 0.05 * np.sin(2 * np.pi * y_val)
            h_1d = shallow_water_dam_break(x, t_val)
            height[:, yi, ti, 0] = h_1d * y_factor
        
        # Compute velocity field from height
        u, v = generate_velocity_field(height[:, :, ti, 0], dx)
        velocity[:, :, ti, 0] = u
        velocity[:, :, ti, 1] = v
        
        # Store coordinates
        X, Y = np.meshgrid(x, y, indexing='ij')
        grid_coords[:, :, ti, 0] = X
        grid_coords[:, :, ti, 1] = Y
        grid_coords[:, :, ti, 2] = t_val
    
    return {
        'grid_coords': grid_coords,
        'height': height,
        'velocity': velocity
    }


def generate_lagrangian_data(
    n_particles: int = 1000,
    nt: int = 16,
    domain: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    t_max: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Generate Lagrangian particle trajectory data.
    
    Particles are initialized on the free surface and advected
    according to the shallow water velocity field.
    
    Args:
        n_particles: Number of surface particles to track
        nt: Number of time steps
        domain: (xmin, xmax, ymin, ymax)
        t_max: Maximum simulation time
    
    Returns:
        Dictionary with:
        - 'positions': (n_particles, nt, 3) particle positions (x, y, z=height)
        - 'velocities': (n_particles, nt, 3) particle velocities
    """
    xmin, xmax, ymin, ymax = domain
    dt = t_max / (nt - 1)
    g = 9.81
    
    # Initialize particles uniformly across domain
    np.random.seed(42)  # Reproducibility
    x = np.random.uniform(xmin + 0.05, xmax - 0.05, n_particles)
    y = np.random.uniform(ymin + 0.05, ymax - 0.05, n_particles)
    
    positions = np.zeros((n_particles, nt, 3), dtype=np.float32)
    velocities = np.zeros((n_particles, nt, 3), dtype=np.float32)
    
    for ti in range(nt):
        t_val = ti * dt
        
        # Compute height at particle positions
        h = shallow_water_dam_break(x, t_val)
        
        # Store positions (z = height on surface)
        positions[:, ti, 0] = x
        positions[:, ti, 1] = y
        positions[:, ti, 2] = h
        
        # Compute velocities from local height gradient
        # Use finite difference approximation
        eps = 0.01
        h_plus = shallow_water_dam_break(x + eps, t_val)
        h_minus = shallow_water_dam_break(x - eps, t_val)
        dh_dx = (h_plus - h_minus) / (2 * eps)
        
        # Velocity components
        c = np.sqrt(g * np.maximum(h, 0.01))
        u = -c * np.sign(dh_dx) * np.minimum(np.abs(dh_dx), 1.0)
        v = np.zeros_like(u)  # Primarily 1D flow
        w = np.zeros_like(u)  # Vertical velocity (surface following)
        
        velocities[:, ti, 0] = u
        velocities[:, ti, 1] = v
        velocities[:, ti, 2] = w
        
        # Advect particles for next time step (simple Euler)
        if ti < nt - 1:
            x = np.clip(x + u * dt, xmin + 0.01, xmax - 0.01)
            y = np.clip(y + v * dt, ymin + 0.01, ymax - 0.01)
    
    return {
        'positions': positions,
        'velocities': velocities
    }


def generate_dataset(
    output_path: str = "data/synthetic_dam_break.h5",
    nx: int = 64,
    ny: int = 64,
    nt: int = 16,
    n_particles: int = 1000
) -> str:
    """
    Generate complete synthetic dataset and save to HDF5.
    
    Args:
        output_path: Path to save HDF5 file
        nx, ny: Eulerian grid resolution
        nt: Number of time steps
        n_particles: Number of Lagrangian particles
    
    Returns:
        Path to saved file
    """
    print(f"Generating synthetic dam break dataset...")
    print(f"  Eulerian grid: {nx}×{ny}×{nt}")
    print(f"  Lagrangian particles: {n_particles}")
    
    # Generate data
    eulerian = generate_eulerian_data(nx=nx, ny=ny, nt=nt)
    lagrangian = generate_lagrangian_data(n_particles=n_particles, nt=nt)
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to HDF5
    with h5py.File(output_path, 'w') as f:
        # Eulerian group
        eul = f.create_group('eulerian')
        eul.create_dataset('grid_coords', data=eulerian['grid_coords'], 
                          compression='gzip', compression_opts=4)
        eul.create_dataset('height', data=eulerian['height'],
                          compression='gzip', compression_opts=4)
        eul.create_dataset('velocity', data=eulerian['velocity'],
                          compression='gzip', compression_opts=4)
        
        # Lagrangian group
        lag = f.create_group('lagrangian')
        lag.create_dataset('positions', data=lagrangian['positions'],
                          compression='gzip', compression_opts=4)
        lag.create_dataset('velocities', data=lagrangian['velocities'],
                          compression='gzip', compression_opts=4)
        
        # Metadata
        f.attrs['nx'] = nx
        f.attrs['ny'] = ny
        f.attrs['nt'] = nt
        f.attrs['n_particles'] = n_particles
        f.attrs['domain'] = [0.0, 1.0, 0.0, 1.0]
        f.attrs['t_max'] = 1.0
        f.attrs['description'] = 'Synthetic dam break for SurfPINN training'
    
    print(f"  Saved to: {output_path}")
    return output_path


def load_dataset(path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load dataset from HDF5 file.
    
    Args:
        path: Path to HDF5 file
    
    Returns:
        Dictionary with 'eulerian' and 'lagrangian' data
    """
    with h5py.File(path, 'r') as f:
        data = {
            'eulerian': {
                'grid_coords': f['eulerian/grid_coords'][...],
                'height': f['eulerian/height'][...],
                'velocity': f['eulerian/velocity'][...]
            },
            'lagrangian': {
                'positions': f['lagrangian/positions'][...],
                'velocities': f['lagrangian/velocities'][...]
            },
            'metadata': dict(f.attrs)
        }
    return data


if __name__ == "__main__":
    # Generate default dataset when run as script
    generate_dataset()
