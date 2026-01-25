"""
Synthetic Data Generation for Cell-Path PINNs

Provides Boids-inspired synthetic trajectory generators for testing and 
data augmentation.
"""

import numpy as np
from typing import Tuple, Optional


def generate_synthetic_trajectory(
    n_steps: int = 500,
    dt: float = 0.1,
    start_pos: Optional[Tuple[float, float]] = None,
    target_pos: Tuple[float, float] = (5.0, 5.0),
    noise_scale: float = 0.1,
    attraction_strength: float = 0.05,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a single microbe trajectory with chemotactic behavior.
    
    Simulates a microbe moving towards a nutrient source (target) with
    some random perturbations (Boids-inspired).
    
    Args:
        n_steps: Number of time steps
        dt: Time step size
        start_pos: Starting (x, y) position. If None, random in [0, 10]
        target_pos: Target nutrient source position
        noise_scale: Scale of random velocity perturbations
        attraction_strength: Strength of attraction to target
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (t, x, y) arrays, each of shape (n_steps,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize position
    if start_pos is None:
        x = np.random.uniform(0, 10)
        y = np.random.uniform(0, 10)
    else:
        x, y = start_pos
    
    # Initialize velocity (small random)
    vx = np.random.uniform(-0.5, 0.5)
    vy = np.random.uniform(-0.5, 0.5)
    
    # Target position
    tx, ty = target_pos
    
    # Storage
    t_arr = np.zeros(n_steps)
    x_arr = np.zeros(n_steps)
    y_arr = np.zeros(n_steps)
    
    for i in range(n_steps):
        t_arr[i] = i * dt
        x_arr[i] = x
        y_arr[i] = y
        
        # Direction to target (chemotactic gradient)
        dx = tx - x
        dy = ty - y
        dist = np.sqrt(dx**2 + dy**2) + 1e-8
        
        # Normalize direction
        dx_norm = dx / dist
        dy_norm = dy / dist
        
        # Update velocity with attraction and noise
        vx += attraction_strength * dx_norm + noise_scale * np.random.randn()
        vy += attraction_strength * dy_norm + noise_scale * np.random.randn()
        
        # Damping to prevent runaway velocities
        speed = np.sqrt(vx**2 + vy**2)
        max_speed = 2.0
        if speed > max_speed:
            vx = vx / speed * max_speed
            vy = vy / speed * max_speed
        
        # Update position
        x += vx * dt
        y += vy * dt
    
    return t_arr, x_arr, y_arr


def generate_circular_trajectory(
    n_steps: int = 100,
    radius: float = 1.0,
    center: Tuple[float, float] = (0.0, 0.0),
    omega: float = 1.0,
    noise_scale: float = 0.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a circular trajectory for testing.
    
    Perfect for validating the constant-speed (geodesic) constraint,
    as circular motion has constant angular velocity.
    
    Args:
        n_steps: Number of time points
        radius: Circle radius
        center: Center (cx, cy) of the circle
        omega: Angular velocity (radians per unit time)
        noise_scale: Gaussian noise to add to positions
        seed: Random seed
        
    Returns:
        Tuple of (t, x, y) arrays, each of shape (n_steps,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    cx, cy = center
    
    # Time from 0 to 2π/omega (one full circle)
    t = np.linspace(0, 2 * np.pi / omega, n_steps)
    
    # Circular trajectory
    x = cx + radius * np.cos(omega * t)
    y = cy + radius * np.sin(omega * t)
    
    # Add noise if specified
    if noise_scale > 0:
        x += noise_scale * np.random.randn(n_steps)
        y += noise_scale * np.random.randn(n_steps)
    
    return t, x, y


def generate_spiral_trajectory(
    n_steps: int = 200,
    r_start: float = 5.0,
    r_end: float = 0.5,
    center: Tuple[float, float] = (5.0, 5.0),
    n_rotations: float = 3.0,
    noise_scale: float = 0.05,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a spiral trajectory (microbe approaching nutrient source).
    
    Models a microbe spiraling inward towards a nutrient concentration,
    which is a common pattern in chemotactic behavior.
    
    Args:
        n_steps: Number of time points
        r_start: Starting radius
        r_end: Ending radius
        center: Center of spiral (nutrient source)
        n_rotations: Number of full rotations
        noise_scale: Gaussian noise
        seed: Random seed
        
    Returns:
        Tuple of (t, x, y) arrays
    """
    if seed is not None:
        np.random.seed(seed)
    
    cx, cy = center
    
    # Time and angle
    t = np.linspace(0, n_rotations * 2 * np.pi, n_steps)
    theta = t  # Angle increases linearly
    
    # Radius decreases linearly (spiral inward)
    r = np.linspace(r_start, r_end, n_steps)
    
    # Convert to Cartesian
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    
    # Normalize time to [0, 1] range for training stability
    t = t / t.max()
    
    # Add noise
    if noise_scale > 0:
        x += noise_scale * np.random.randn(n_steps)
        y += noise_scale * np.random.randn(n_steps)
    
    return t, x, y


def generate_multi_trajectory(
    n_trajectories: int = 3,
    n_steps: int = 200,
    seed: Optional[int] = None
) -> list:
    """
    Generate multiple synthetic trajectories with different patterns.
    
    Useful for training with diverse data.
    
    Args:
        n_trajectories: Number of trajectories to generate
        n_steps: Points per trajectory
        seed: Random seed
        
    Returns:
        List of (t, x, y) tuples
    """
    if seed is not None:
        np.random.seed(seed)
    
    trajectories = []
    
    for i in range(n_trajectories):
        # Vary the pattern
        pattern = i % 3
        
        if pattern == 0:
            # Chemotactic drift
            traj = generate_synthetic_trajectory(
                n_steps=n_steps,
                start_pos=(np.random.uniform(0, 3), np.random.uniform(0, 3)),
                target_pos=(np.random.uniform(7, 10), np.random.uniform(7, 10)),
                seed=None  # Already seeded above
            )
        elif pattern == 1:
            # Circular
            traj = generate_circular_trajectory(
                n_steps=n_steps,
                radius=np.random.uniform(1, 3),
                center=(5, 5),
                noise_scale=0.02
            )
        else:
            # Spiral
            traj = generate_spiral_trajectory(
                n_steps=n_steps,
                r_start=np.random.uniform(3, 5),
                r_end=np.random.uniform(0.2, 0.5),
                noise_scale=0.03
            )
        
        trajectories.append(traj)
    
    return trajectories


def normalize_trajectory(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Normalize trajectory data to [0, 1] range for training stability.
    
    Args:
        t, x, y: Raw trajectory arrays
        
    Returns:
        Normalized (t, x, y) and dict with normalization parameters
    """
    t_min, t_max = t.min(), t.max()
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Avoid division by zero
    t_range = t_max - t_min if t_max > t_min else 1.0
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0
    
    t_norm = (t - t_min) / t_range
    x_norm = (x - x_min) / x_range
    y_norm = (y - y_min) / y_range
    
    params = {
        't_min': t_min, 't_max': t_max, 't_range': t_range,
        'x_min': x_min, 'x_max': x_max, 'x_range': x_range,
        'y_min': y_min, 'y_max': y_max, 'y_range': y_range,
    }
    
    return t_norm, x_norm, y_norm, params


def denormalize_trajectory(
    t_norm: np.ndarray,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    params: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reverse normalization to get original scale.
    
    Args:
        t_norm, x_norm, y_norm: Normalized arrays
        params: Dict from normalize_trajectory
        
    Returns:
        Original-scale (t, x, y)
    """
    t = t_norm * params['t_range'] + params['t_min']
    x = x_norm * params['x_range'] + params['x_min']
    y = y_norm * params['y_range'] + params['y_min']
    
    return t, x, y
