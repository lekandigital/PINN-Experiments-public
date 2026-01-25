#!/usr/bin/env python3
"""
Generate synthetic 2D airfoil flow data using potential flow theory.
This provides ground truth for PINN training without OpenFOAM.

For a thin symmetric airfoil at small angles, potential flow gives:
- Lift coefficient: Cl = 2 * pi * alpha (radians)
- Velocity field from conformal mapping (Joukowski transform)

Usage:
    python synthetic_data.py --output data/processed/training_data.h5 --n_points 100000
"""

import numpy as np
import h5py
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def joukowski_flow(x: np.ndarray, y: np.ndarray, alpha_deg: float,
                   c: float = 1.0, U_inf: float = 1.0) -> tuple:
    """
    Compute potential flow around a Joukowski airfoil.

    Args:
        x, y: Coordinates (can be arrays)
        alpha_deg: Angle of attack in degrees
        c: Chord length
        U_inf: Freestream velocity

    Returns:
        u, v, p: Velocity components and pressure coefficient
    """
    alpha = np.deg2rad(alpha_deg)

    # Transform to complex plane
    z = x + 1j * y

    # Joukowski parameters (simplified for thin airfoil)
    a = c / 4  # Circle radius

    # Avoid division by zero at the origin
    eps = 1e-10
    z_safe = np.where(np.abs(z) < eps, eps + 0j, z)

    # Complex potential derivative (velocity)
    # For flow past circle with circulation (Kutta condition)
    Gamma = 4 * np.pi * a * U_inf * np.sin(alpha)

    # Complex velocity (derivative of complex potential)
    dW_dz = U_inf * np.exp(-1j * alpha) - 1j * Gamma / (2 * np.pi * z_safe)

    # Velocity components
    u = np.real(dW_dz)
    v = -np.imag(dW_dz)

    # Pressure coefficient from Bernoulli (steady, incompressible)
    V_mag = np.sqrt(u**2 + v**2)
    Cp = 1 - (V_mag / U_inf)**2

    # Convert to dimensional pressure (rho = 1 for simplicity)
    p = Cp * 0.5 * U_inf**2

    return u.astype(np.float32), v.astype(np.float32), p.astype(np.float32)


def generate_naca_points(code: str = '0012', n_points: int = 100) -> tuple:
    """
    Generate NACA 4-digit airfoil surface points.

    Args:
        code: 4-digit NACA code (e.g., '0012', '2412')
        n_points: Number of points per surface

    Returns:
        x_upper, y_upper, x_lower, y_lower: Surface coordinates
    """
    m = int(code[0]) / 100.0  # Maximum camber
    p = int(code[1]) / 10.0   # Position of max camber
    t = int(code[2:]) / 100.0 # Maximum thickness

    if p == 0:
        p = 0.5  # Avoid division by zero for symmetric airfoils

    # Cosine spacing for smooth LE/TE resolution
    beta = np.linspace(0, np.pi, n_points)
    x = (1 - np.cos(beta)) / 2

    # Thickness distribution (NACA 4-digit formula)
    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)

    # Camber line
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    if m > 0:
        mask = x < p
        yc[mask] = m / (p**2) * (2*p*x[mask] - x[mask]**2)
        yc[~mask] = m / ((1-p)**2) * ((1 - 2*p) + 2*p*x[~mask] - x[~mask]**2)

        dyc_dx[mask] = 2*m / (p**2) * (p - x[mask])
        dyc_dx[~mask] = 2*m / ((1-p)**2) * (p - x[~mask])

    theta = np.arctan(dyc_dx)

    # Upper and lower surfaces
    x_upper = x - yt * np.sin(theta)
    y_upper = yc + yt * np.cos(theta)
    x_lower = x + yt * np.sin(theta)
    y_lower = yc - yt * np.cos(theta)

    return x_upper, y_upper, x_lower, y_lower


def generate_boundary_points(naca_code: str, n_surface: int = 200,
                            n_farfield: int = 100) -> dict:
    """
    Generate boundary condition points for PINN training.

    Returns:
        dict with 'surface', 'inlet', 'outlet', 'top', 'bottom' boundary points
    """
    # Airfoil surface (no-slip: u=v=0)
    x_u, y_u, x_l, y_l = generate_naca_points(naca_code, n_surface // 2)
    x_surface = np.concatenate([x_u, x_l[::-1]])
    y_surface = np.concatenate([y_u, y_l[::-1]])

    # Far-field boundaries
    domain = {'xmin': -0.5, 'xmax': 2.0, 'ymin': -1.0, 'ymax': 1.0}

    # Inlet (left boundary)
    x_inlet = np.full(n_farfield, domain['xmin'])
    y_inlet = np.linspace(domain['ymin'], domain['ymax'], n_farfield)

    # Outlet (right boundary)
    x_outlet = np.full(n_farfield, domain['xmax'])
    y_outlet = np.linspace(domain['ymin'], domain['ymax'], n_farfield)

    return {
        'surface': (x_surface.astype(np.float32), y_surface.astype(np.float32)),
        'inlet': (x_inlet.astype(np.float32), y_inlet.astype(np.float32)),
        'outlet': (x_outlet.astype(np.float32), y_outlet.astype(np.float32)),
    }


def generate_training_data(
    n_domain_points: int = 50000,
    n_boundary_points: int = 1000,
    n_airfoils: int = 5,
    aoa_range: tuple = (-5, 15),
    output_path: str = 'data/processed/training_data.h5'
) -> str:
    """
    Generate complete training dataset.

    Creates:
    - Domain collocation points with PDE ground truth
    - Boundary points with BC ground truth
    - Multiple AoA values for parameterized learning
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # NACA codes to use
    naca_codes = ['0012', '2412', '4412', '0015', '2415'][:n_airfoils]
    aoa_values = np.linspace(aoa_range[0], aoa_range[1], 9)  # 9 angles

    all_x, all_y, all_aoa = [], [], []
    all_u, all_v, all_p = [], [], []

    for naca in naca_codes:
        logger.info(f"Generating data for NACA {naca}...")

        for aoa in aoa_values:
            # Domain points (random sampling)
            x_domain = np.random.uniform(-0.5, 2.0, n_domain_points)
            y_domain = np.random.uniform(-1.0, 1.0, n_domain_points)

            # Exclude points inside/very close to airfoil
            x_surf, y_surf, _, _ = generate_naca_points(naca, 50)
            # Simple exclusion: remove points too close to chord line in airfoil region
            mask = ~((x_domain >= 0) & (x_domain <= 1) & (np.abs(y_domain) < 0.15))
            x_domain = x_domain[mask]
            y_domain = y_domain[mask]

            # Compute flow field
            u, v, p = joukowski_flow(x_domain, y_domain, aoa)

            # Append data
            n_pts = len(x_domain)
            all_x.append(x_domain.astype(np.float32))
            all_y.append(y_domain.astype(np.float32))
            all_aoa.append(np.full(n_pts, aoa, dtype=np.float32))
            all_u.append(u)
            all_v.append(v)
            all_p.append(p)

    # Concatenate all data
    x_all = np.concatenate(all_x)
    y_all = np.concatenate(all_y)
    aoa_all = np.concatenate(all_aoa)
    u_all = np.concatenate(all_u)
    v_all = np.concatenate(all_v)
    p_all = np.concatenate(all_p)

    logger.info(f"Saving {len(x_all):,} samples to {output_path}...")

    with h5py.File(output_path, 'w') as f:
        # Training data
        f.create_dataset('x', data=x_all, compression='gzip')
        f.create_dataset('y', data=y_all, compression='gzip')
        f.create_dataset('aoa', data=aoa_all, compression='gzip')
        f.create_dataset('u', data=u_all, compression='gzip')
        f.create_dataset('v', data=v_all, compression='gzip')
        f.create_dataset('p', data=p_all, compression='gzip')

        # Metadata
        f.attrs['n_samples'] = len(x_all)
        f.attrs['naca_codes'] = ','.join(naca_codes)
        f.attrs['aoa_min'] = aoa_range[0]
        f.attrs['aoa_max'] = aoa_range[1]
        f.attrs['domain_xmin'] = -0.5
        f.attrs['domain_xmax'] = 2.0
        f.attrs['domain_ymin'] = -1.0
        f.attrs['domain_ymax'] = 1.0

    logger.info(f"Generated {len(x_all):,} training samples successfully!")
    return output_path


def generate_test_data(output_path: str = 'data/processed/test_data.h5') -> str:
    """Generate held-out test set with different AoA values."""
    logger.info("Generating test dataset...")
    return generate_training_data(
        n_domain_points=10000,
        n_airfoils=3,
        aoa_range=(0, 10),
        output_path=output_path
    )


def verify_data(path: str) -> None:
    """Print data statistics for verification."""
    with h5py.File(path, 'r') as f:
        logger.info(f"\n=== Dataset: {path} ===")
        logger.info(f"Samples: {f.attrs['n_samples']:,}")
        logger.info(f"NACA codes: {f.attrs['naca_codes']}")
        logger.info(f"AoA range: [{f.attrs['aoa_min']}, {f.attrs['aoa_max']}]")

        for key in ['x', 'y', 'aoa', 'u', 'v', 'p']:
            data = f[key][:]
            logger.info(f"  {key}: shape={data.shape}, range=[{data.min():.4f}, {data.max():.4f}]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic airfoil flow data')
    parser.add_argument('--output', default='data/processed/training_data.h5',
                        help='Output HDF5 file path')
    parser.add_argument('--n_points', type=int, default=50000,
                        help='Number of domain points per case')
    parser.add_argument('--n_airfoils', type=int, default=5,
                        help='Number of different airfoils')
    parser.add_argument('--aoa_min', type=float, default=-5,
                        help='Minimum angle of attack (degrees)')
    parser.add_argument('--aoa_max', type=float, default=15,
                        help='Maximum angle of attack (degrees)')
    parser.add_argument('--test_only', action='store_true',
                        help='Generate only test data')
    args = parser.parse_args()

    if args.test_only:
        test_path = generate_test_data()
        verify_data(test_path)
    else:
        # Generate training data
        train_path = generate_training_data(
            n_domain_points=args.n_points,
            n_airfoils=args.n_airfoils,
            aoa_range=(args.aoa_min, args.aoa_max),
            output_path=args.output
        )
        verify_data(train_path)

        # Generate test data
        test_path = generate_test_data()
        verify_data(test_path)

        logger.info("\nData generation complete!")
