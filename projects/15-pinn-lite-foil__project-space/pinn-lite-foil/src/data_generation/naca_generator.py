"""
NACA 4-Digit Airfoil Generator
Generates parametric NACA airfoil geometries for CFD simulation.

Usage:
    python naca_generator.py --n_samples 50 --output data/raw/
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import json


def naca4_thickness(t: float, x: np.ndarray) -> np.ndarray:
    """
    Calculate NACA 4-digit thickness distribution.
    
    Args:
        t: Maximum thickness as fraction of chord (e.g., 0.12 for 12%)
        x: Chordwise positions (0 to 1)
    
    Returns:
        Half-thickness at each x position
    """
    return 5 * t * (
        0.2969 * np.sqrt(x) 
        - 0.1260 * x 
        - 0.3516 * x**2 
        + 0.2843 * x**3 
        - 0.1015 * x**4  # Closed trailing edge
    )


def naca4_camber(m: float, p: float, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate NACA 4-digit camber line and gradient.
    
    Args:
        m: Maximum camber as fraction of chord
        p: Location of maximum camber as fraction of chord
        x: Chordwise positions (0 to 1)
    
    Returns:
        Tuple of (camber height, camber gradient) arrays
    """
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    
    if m == 0 or p == 0:
        return yc, dyc
    
    # Forward of max camber
    mask_front = x < p
    yc[mask_front] = (m / p**2) * (2 * p * x[mask_front] - x[mask_front]**2)
    dyc[mask_front] = (2 * m / p**2) * (p - x[mask_front])
    
    # Aft of max camber
    mask_aft = x >= p
    yc[mask_aft] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x[mask_aft] - x[mask_aft]**2)
    dyc[mask_aft] = (2 * m / (1 - p)**2) * (p - x[mask_aft])
    
    return yc, dyc


def generate_naca4(
    m: float, 
    p: float, 
    t: float, 
    n_points: int = 300,
    cosine_spacing: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate NACA 4-digit airfoil coordinates.
    
    Args:
        m: Maximum camber (0-0.09)
        p: Camber position (0.1-0.9)
        t: Thickness ratio (0.01-0.40)
        n_points: Number of points per surface
        cosine_spacing: Use cosine spacing for better LE resolution
    
    Returns:
        Tuple of (x, y) coordinate arrays for complete airfoil
    """
    # Generate chordwise distribution
    if cosine_spacing:
        beta = np.linspace(0, np.pi, n_points)
        x = 0.5 * (1 - np.cos(beta))
    else:
        x = np.linspace(0, 1, n_points)
    
    # Calculate thickness and camber
    yt = naca4_thickness(t, x)
    yc, dyc = naca4_camber(m, p, x)
    
    # Calculate surface normals
    theta = np.arctan(dyc)
    
    # Upper surface
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    
    # Lower surface
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    
    # Combine surfaces (TE -> LE on upper, LE -> TE on lower)
    x_coords = np.concatenate([xu[::-1], xl[1:]])
    y_coords = np.concatenate([yu[::-1], yl[1:]])
    
    return x_coords, y_coords


def naca_designation(m: float, p: float, t: float) -> str:
    """Convert NACA parameters to 4-digit designation string."""
    m_digit = int(round(m * 100))
    p_digit = int(round(p * 10))
    t_digits = int(round(t * 100))
    return f"NACA{m_digit}{p_digit}{t_digits:02d}"


def generate_parametric_sweep(
    m_range: Tuple[float, float] = (0.01, 0.04),
    p_range: Tuple[float, float] = (0.2, 0.6),
    t_range: Tuple[float, float] = (0.08, 0.18),
    n_samples: int = 50,
    seed: int = 42
) -> List[Dict]:
    """
    Generate parametric sweep of NACA airfoil parameters.
    
    Args:
        m_range: (min, max) for maximum camber
        p_range: (min, max) for camber position
        t_range: (min, max) for thickness
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
    
    Returns:
        List of parameter dictionaries
    """
    np.random.seed(seed)
    
    samples = []
    for i in range(n_samples):
        m = np.random.uniform(*m_range)
        p = np.random.uniform(*p_range)
        t = np.random.uniform(*t_range)
        
        samples.append({
            'id': i,
            'm': float(m),
            'p': float(p),
            't': float(t),
            'designation': naca_designation(m, p, t)
        })
    
    return samples


def save_airfoil_dat(
    x: np.ndarray, 
    y: np.ndarray, 
    filepath: Path, 
    designation: str
) -> None:
    """Save airfoil coordinates in standard .dat format."""
    with open(filepath, 'w') as f:
        f.write(f"{designation}\n")
        for xi, yi in zip(x, y):
            f.write(f"  {xi:10.6f}  {yi:10.6f}\n")


def save_airfoil_csv(
    x: np.ndarray, 
    y: np.ndarray, 
    filepath: Path
) -> None:
    """Save airfoil coordinates in CSV format."""
    data = np.column_stack([x, y])
    np.savetxt(filepath, data, delimiter=',', header='x,y', comments='')


def main():
    parser = argparse.ArgumentParser(
        description='Generate NACA 4-digit airfoil geometries'
    )
    parser.add_argument(
        '--n_samples', type=int, default=50,
        help='Number of airfoil geometries to generate'
    )
    parser.add_argument(
        '--n_points', type=int, default=300,
        help='Number of points per airfoil surface'
    )
    parser.add_argument(
        '--output', type=str, default='data/raw/',
        help='Output directory for airfoil files'
    )
    parser.add_argument(
        '--m_min', type=float, default=0.01,
        help='Minimum camber'
    )
    parser.add_argument(
        '--m_max', type=float, default=0.04,
        help='Maximum camber'
    )
    parser.add_argument(
        '--p_min', type=float, default=0.2,
        help='Minimum camber position'
    )
    parser.add_argument(
        '--p_max', type=float, default=0.6,
        help='Maximum camber position'
    )
    parser.add_argument(
        '--t_min', type=float, default=0.08,
        help='Minimum thickness'
    )
    parser.add_argument(
        '--t_max', type=float, default=0.18,
        help='Maximum thickness'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--format', type=str, choices=['dat', 'csv', 'both'], default='both',
        help='Output format'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate parametric sweep
    print(f"Generating {args.n_samples} NACA airfoil geometries...")
    samples = generate_parametric_sweep(
        m_range=(args.m_min, args.m_max),
        p_range=(args.p_min, args.p_max),
        t_range=(args.t_min, args.t_max),
        n_samples=args.n_samples,
        seed=args.seed
    )
    
    # Generate and save airfoils
    for sample in samples:
        x, y = generate_naca4(
            sample['m'], 
            sample['p'], 
            sample['t'],
            n_points=args.n_points
        )
        
        base_name = f"airfoil_{sample['id']:04d}_{sample['designation']}"
        
        if args.format in ['dat', 'both']:
            save_airfoil_dat(x, y, output_dir / f"{base_name}.dat", sample['designation'])
        
        if args.format in ['csv', 'both']:
            save_airfoil_csv(x, y, output_dir / f"{base_name}.csv")
        
        print(f"  Generated: {sample['designation']} (m={sample['m']:.3f}, p={sample['p']:.2f}, t={sample['t']:.3f})")
    
    # Save metadata
    metadata = {
        'n_samples': args.n_samples,
        'n_points': args.n_points,
        'parameters': {
            'm_range': [args.m_min, args.m_max],
            'p_range': [args.p_min, args.p_max],
            't_range': [args.t_min, args.t_max]
        },
        'seed': args.seed,
        'samples': samples
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Generated {args.n_samples} airfoils in {output_dir}")
    print(f"✓ Saved metadata to {output_dir / 'metadata.json'}")


if __name__ == '__main__':
    main()
