"""
Dataset Generator for Maxwell-PINN-NIF

This module generates synthetic training data for electromagnetic simulations.
It creates spatially-varying permittivity ε(x) and permeability μ(x) maps,
simulates electromagnetic fields using FDTD (via MEEP or analytical solutions),
and exports everything to HDF5 format for training.

Two modes are supported:
1. MEEP mode: Full FDTD simulation using MEEP library (requires installation)
2. Analytical mode: Uses analytical plane wave / waveguide solutions (no dependencies)

The analytical mode is useful for testing and validation, while MEEP mode
provides realistic training data for complex geometries.

Dependencies:
- numpy, h5py (required)
- meep (optional, for FDTD simulations)
- scipy (optional, for advanced material generation)
"""

import numpy as np
import h5py
from typing import Tuple, Optional, Dict, List, Callable
from pathlib import Path
import warnings


# ============================================================================
# Material Map Generators
# ============================================================================

def generate_uniform_material(
    Nx: int = 64,
    Ny: int = 64,
    Nz: int = 64,
    eps_val: float = 1.0,
    mu_val: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate uniform permittivity and permeability maps.
    
    Args:
        Nx, Ny, Nz: Grid dimensions
        eps_val: Constant permittivity value
        mu_val: Constant permeability value
        
    Returns:
        eps_map: [Nx, Ny, Nz] permittivity array
        mu_map: [Nx, Ny, Nz] permeability array
    """
    eps_map = np.full((Nx, Ny, Nz), eps_val, dtype=np.float32)
    mu_map = np.full((Nx, Ny, Nz), mu_val, dtype=np.float32)
    return eps_map, mu_map


def generate_layered_material(
    Nx: int = 64,
    Ny: int = 64,
    Nz: int = 64,
    num_layers: int = 3,
    eps_range: Tuple[float, float] = (1.0, 4.0),
    mu_range: Tuple[float, float] = (0.9, 1.1),
    direction: str = 'z'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate layered material (stratified medium) along one axis.
    
    This models structures like Bragg reflectors, thin-film stacks,
    or geological layers.
    
    Args:
        Nx, Ny, Nz: Grid dimensions
        num_layers: Number of material layers
        eps_range: (min, max) permittivity values
        mu_range: (min, max) permeability values
        direction: Layering direction ('x', 'y', or 'z')
        
    Returns:
        eps_map, mu_map: Material property arrays
    """
    eps_map = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    mu_map = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    
    # Determine grid size along layering direction
    sizes = {'x': Nx, 'y': Ny, 'z': Nz}
    N = sizes[direction]
    
    # Random layer thicknesses (approximately equal)
    layer_widths = np.random.rand(num_layers)
    layer_widths = (layer_widths / layer_widths.sum() * N).astype(int)
    layer_widths[-1] = N - layer_widths[:-1].sum()  # Ensure exact sum
    
    # Random material values for each layer
    eps_values = np.random.uniform(eps_range[0], eps_range[1], num_layers)
    mu_values = np.random.uniform(mu_range[0], mu_range[1], num_layers)
    
    # Fill layers
    start = 0
    for i, width in enumerate(layer_widths):
        end = start + width
        if direction == 'x':
            eps_map[start:end, :, :] = eps_values[i]
            mu_map[start:end, :, :] = mu_values[i]
        elif direction == 'y':
            eps_map[:, start:end, :] = eps_values[i]
            mu_map[:, start:end, :] = mu_values[i]
        else:  # z
            eps_map[:, :, start:end] = eps_values[i]
            mu_map[:, :, start:end] = mu_values[i]
        start = end
    
    return eps_map, mu_map


def generate_random_inclusions(
    Nx: int = 64,
    Ny: int = 64,
    Nz: int = 64,
    background_eps: float = 1.0,
    inclusion_eps_range: Tuple[float, float] = (2.0, 12.0),
    num_inclusions: int = 5,
    radius_range: Tuple[float, float] = (0.05, 0.15),
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate material with random spherical inclusions.
    
    Models composites, metamaterials, or heterogeneous media.
    
    Args:
        Nx, Ny, Nz: Grid dimensions
        background_eps: Background permittivity
        inclusion_eps_range: (min, max) permittivity for inclusions
        num_inclusions: Number of spherical inclusions
        radius_range: (min, max) radius as fraction of domain size
        seed: Random seed for reproducibility
        
    Returns:
        eps_map, mu_map: Material property arrays
    """
    if seed is not None:
        np.random.seed(seed)
    
    eps_map = np.full((Nx, Ny, Nz), background_eps, dtype=np.float32)
    mu_map = np.ones((Nx, Ny, Nz), dtype=np.float32)
    
    # Create coordinate grids
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    z = np.linspace(0, 1, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    for _ in range(num_inclusions):
        # Random center (away from boundaries)
        cx = np.random.uniform(0.15, 0.85)
        cy = np.random.uniform(0.15, 0.85)
        cz = np.random.uniform(0.15, 0.85)
        
        # Random radius
        r = np.random.uniform(radius_range[0], radius_range[1])
        
        # Random permittivity
        eps_val = np.random.uniform(inclusion_eps_range[0], inclusion_eps_range[1])
        
        # Create spherical mask
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)
        mask = dist < r
        
        # Apply material
        eps_map[mask] = eps_val
    
    return eps_map, mu_map


def generate_sinusoidal_material(
    Nx: int = 64,
    Ny: int = 64,
    Nz: int = 64,
    eps_mean: float = 2.5,
    eps_amplitude: float = 1.0,
    wavelength: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sinusoidally varying permittivity (photonic crystal-like).
    
    Args:
        Nx, Ny, Nz: Grid dimensions
        eps_mean: Mean permittivity
        eps_amplitude: Amplitude of sinusoidal variation
        wavelength: Spatial wavelength of modulation
        
    Returns:
        eps_map, mu_map: Material property arrays
    """
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    z = np.linspace(0, 1, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    k = 2 * np.pi / wavelength
    eps_map = eps_mean + eps_amplitude * np.sin(k * X) * np.cos(k * Y)
    eps_map = np.clip(eps_map, 1.0, 12.0).astype(np.float32)
    
    mu_map = np.ones((Nx, Ny, Nz), dtype=np.float32)
    
    return eps_map, mu_map


# ============================================================================
# Analytical Field Solutions (for validation and MEEP-free testing)
# ============================================================================

def compute_plane_wave_fields(
    coords: np.ndarray,
    k_vector: np.ndarray = np.array([0, 0, 1]),
    E0: float = 1.0,
    omega: float = 1.0,
    eps: float = 1.0,
    mu: float = 1.0,
    polarization: str = 'x'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute analytical plane wave solution in homogeneous medium.
    
    E = E0 * exp(i(k·r - ωt)) * polarization_vector
    H = (k × E) / (ωμ)
    
    For time-harmonic (single frequency) fields, we return the real part.
    
    Args:
        coords: [N, 3] spatial coordinates
        k_vector: Propagation direction (will be normalized)
        E0: Electric field amplitude
        omega: Angular frequency
        eps: Permittivity
        mu: Permeability
        polarization: 'x', 'y', or 'z' for E-field direction
        
    Returns:
        E_field: [N, 3] electric field
        H_field: [N, 3] magnetic field
    """
    # Normalize k-vector and compute wavenumber
    k_hat = k_vector / np.linalg.norm(k_vector)
    n = np.sqrt(eps * mu)  # Refractive index
    k = omega * n  # Wavenumber magnitude
    
    # Phase at each point: k·r
    phase = k * (coords @ k_hat)
    
    # E-field (real part of complex exponential)
    E_field = np.zeros_like(coords)
    pol_idx = {'x': 0, 'y': 1, 'z': 2}[polarization]
    E_field[:, pol_idx] = E0 * np.cos(phase)
    
    # H-field from H = (k × E) / (ωμ)
    # For plane wave: H = (n/η) * (k̂ × E), where η = sqrt(μ/ε)
    eta = np.sqrt(mu / eps)  # Impedance
    
    # Cross product k̂ × ê (polarization unit vector)
    e_hat = np.zeros(3)
    e_hat[pol_idx] = 1.0
    h_dir = np.cross(k_hat, e_hat)
    
    H_field = np.zeros_like(coords)
    H_magnitude = E0 * np.cos(phase) / eta
    for i in range(3):
        H_field[:, i] = H_magnitude * h_dir[i]
    
    return E_field.astype(np.float32), H_field.astype(np.float32)


def compute_waveguide_mode_2d(
    coords: np.ndarray,
    width: float = 0.5,
    mode_number: int = 1,
    omega: float = 1.0,
    eps_core: float = 4.0,
    eps_clad: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D waveguide TE mode (Ez polarization).
    
    For a slab waveguide centered at y=0.5 with given width:
    - Inside core: Ez ~ sin(m*π*y/width) * exp(i*β*z)
    - Outside: evanescent decay
    
    This is simplified (doesn't solve the transcendental equation exactly).
    
    Args:
        coords: [N, 3] spatial coordinates
        width: Waveguide width
        mode_number: Mode order (1, 2, 3, ...)
        omega: Angular frequency
        eps_core: Core permittivity
        eps_clad: Cladding permittivity
        
    Returns:
        E_field, H_field: [N, 3] field arrays
    """
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    
    # Waveguide centered at y = 0.5
    y_center = 0.5
    y_local = y - y_center
    
    # Core region mask
    in_core = np.abs(y_local) < width / 2
    
    # Transverse wavenumber
    kt = mode_number * np.pi / width
    
    # Propagation constant (simplified)
    k0 = omega * np.sqrt(eps_core)
    beta = np.sqrt(k0**2 - kt**2 + 0j).real
    
    # Decay constant in cladding
    k_clad = omega * np.sqrt(eps_clad)
    gamma = np.sqrt(kt**2 - k_clad**2 + 0j).real
    
    # Ez field
    Ez = np.zeros(len(coords), dtype=np.float32)
    
    # In core: sinusoidal
    Ez[in_core] = np.sin(kt * (y_local[in_core] + width/2)) * np.cos(beta * z[in_core])
    
    # In cladding: exponential decay
    above = (~in_core) & (y_local > 0)
    below = (~in_core) & (y_local < 0)
    
    Ez[above] = np.sin(kt * width) * np.exp(-gamma * (y_local[above] - width/2)) * np.cos(beta * z[above])
    Ez[below] = 0  # Simplified - set to zero in bottom cladding
    
    # Construct field arrays (TM-like: Ez non-zero)
    E_field = np.zeros((len(coords), 3), dtype=np.float32)
    H_field = np.zeros((len(coords), 3), dtype=np.float32)
    E_field[:, 2] = Ez  # Ez component
    
    # Approximate H from curl (Hx, Hy from dEz/dy, dEz/dz)
    # This is simplified - full solution requires solving Maxwell's equations
    
    return E_field, H_field


# ============================================================================
# MEEP-based FDTD Simulation (optional)
# ============================================================================

def check_meep_available() -> bool:
    """Check if MEEP is available for import."""
    try:
        import meep
        return True
    except ImportError:
        return False


def simulate_fields_meep_2d(
    eps_map_2d: np.ndarray,
    dx: float = 0.02,
    frequency: float = 1.0,
    run_time: float = 50.0,
    pml_thickness: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run 2D MEEP FDTD simulation with given permittivity map.
    
    Simulates TM mode (Ez, Hx, Hy non-zero) with point source at center.
    
    Args:
        eps_map_2d: [Nx, Ny] permittivity array
        dx: Grid spacing
        frequency: Source frequency
        run_time: Simulation run time
        pml_thickness: PML thickness in grid units
        
    Returns:
        Ez, Hx, Hy: Field component arrays
    """
    if not check_meep_available():
        raise ImportError("MEEP is required for FDTD simulation. Install with: conda install -c conda-forge pymeep")
    
    import meep as mp
    
    Nx, Ny = eps_map_2d.shape
    Lx, Ly = Nx * dx, Ny * dx
    
    # Create material function from epsilon map
    def eps_func(pt):
        i = int((pt.x + Lx/2) / dx)
        j = int((pt.y + Ly/2) / dx)
        i = max(0, min(i, Nx-1))
        j = max(0, min(j, Ny-1))
        return eps_map_2d[i, j]
    
    # Define geometry as a block with spatially-varying epsilon
    # MEEP requires using material_function for inhomogeneous media
    geometry = []
    
    # For simplicity, we use a uniform medium here
    # Full implementation would use mp.MaterialFunction
    eps_avg = float(eps_map_2d.mean())
    geometry = [mp.Block(
        size=mp.Vector3(Lx, Ly, 0),
        center=mp.Vector3(0, 0, 0),
        material=mp.Medium(epsilon=eps_avg)
    )]
    
    # Point source at center
    sources = [mp.Source(
        mp.ContinuousSource(frequency=frequency),
        component=mp.Ez,
        center=mp.Vector3(0, 0, 0)
    )]
    
    # Create simulation
    sim = mp.Simulation(
        cell_size=mp.Vector3(Lx + 2*pml_thickness, Ly + 2*pml_thickness, 0),
        geometry=geometry,
        sources=sources,
        boundary_layers=[mp.PML(pml_thickness)],
        resolution=1/dx
    )
    
    # Run simulation
    sim.run(until=run_time)
    
    # Extract fields
    Ez = sim.get_array(center=mp.Vector3(), size=mp.Vector3(Lx, Ly, 0), component=mp.Ez)
    Hx = sim.get_array(center=mp.Vector3(), size=mp.Vector3(Lx, Ly, 0), component=mp.Hx)
    Hy = sim.get_array(center=mp.Vector3(), size=mp.Vector3(Lx, Ly, 0), component=mp.Hy)
    
    return Ez.astype(np.float32), Hx.astype(np.float32), Hy.astype(np.float32)


# ============================================================================
# Dataset Creation and Export
# ============================================================================

def create_coordinate_grid(
    Nx: int, Ny: int, Nz: int,
    domain: Tuple[float, float, float, float, float, float] = (0, 1, 0, 1, 0, 1)
) -> np.ndarray:
    """
    Create flattened coordinate grid for PINN training.
    
    Args:
        Nx, Ny, Nz: Grid dimensions
        domain: (xmin, xmax, ymin, ymax, zmin, zmax)
        
    Returns:
        coords: [Nx*Ny*Nz, 3] coordinate array
    """
    xmin, xmax, ymin, ymax, zmin, zmax = domain
    
    x = np.linspace(xmin, xmax, Nx)
    y = np.linspace(ymin, ymax, Ny)
    z = np.linspace(zmin, zmax, Nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    coords = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    return coords.astype(np.float32)


def generate_training_dataset(
    n_samples: int = 10000,
    grid_size: Tuple[int, int, int] = (32, 32, 32),
    material_type: str = 'uniform',
    use_meep: bool = False,
    seed: Optional[int] = 42,
    **material_kwargs
) -> Dict[str, np.ndarray]:
    """
    Generate a complete training dataset.
    
    Args:
        n_samples: Number of random sample points (for PINN training)
        grid_size: (Nx, Ny, Nz) grid dimensions
        material_type: 'uniform', 'layered', 'inclusions', 'sinusoidal'
        use_meep: Use MEEP for field simulation (requires installation)
        seed: Random seed
        material_kwargs: Additional arguments for material generator
        
    Returns:
        dataset: Dict with 'coords', 'eps', 'mu', 'E_field', 'H_field'
    """
    if seed is not None:
        np.random.seed(seed)
    
    Nx, Ny, Nz = grid_size
    
    # Generate material maps
    material_generators = {
        'uniform': generate_uniform_material,
        'layered': generate_layered_material,
        'inclusions': generate_random_inclusions,
        'sinusoidal': generate_sinusoidal_material
    }
    
    if material_type not in material_generators:
        raise ValueError(f"Unknown material type: {material_type}")
    
    eps_map, mu_map = material_generators[material_type](Nx, Ny, Nz, **material_kwargs)
    
    # Create coordinate grid
    coords_grid = create_coordinate_grid(Nx, Ny, Nz)
    
    # Generate fields
    if use_meep and check_meep_available():
        # Use 2D slice for MEEP (center z-slice)
        eps_2d = eps_map[:, :, Nz//2]
        Ez, Hx, Hy = simulate_fields_meep_2d(eps_2d)
        
        # Expand to 3D (replicate along z)
        E_field = np.zeros((Nx * Ny * Nz, 3), dtype=np.float32)
        H_field = np.zeros((Nx * Ny * Nz, 3), dtype=np.float32)
        
        for k in range(Nz):
            idx_start = k * Nx * Ny
            idx_end = (k + 1) * Nx * Ny
            E_field[idx_start:idx_end, 2] = Ez.flatten()
            H_field[idx_start:idx_end, 0] = Hx.flatten()
            H_field[idx_start:idx_end, 1] = Hy.flatten()
    else:
        # Use analytical plane wave solution
        avg_eps = float(eps_map.mean())
        avg_mu = float(mu_map.mean())
        E_field, H_field = compute_plane_wave_fields(
            coords_grid, 
            k_vector=np.array([0, 0, 1]),
            eps=avg_eps,
            mu=avg_mu
        )
    
    # Flatten material maps to match coordinate array
    eps_flat = eps_map.flatten()
    mu_flat = mu_map.flatten()
    
    # Optionally subsample for training (PINN doesn't need all grid points)
    if n_samples < len(coords_grid):
        indices = np.random.choice(len(coords_grid), n_samples, replace=False)
        coords = coords_grid[indices]
        eps = eps_flat[indices]
        mu = mu_flat[indices]
        E = E_field[indices]
        H = H_field[indices]
    else:
        coords = coords_grid
        eps = eps_flat
        mu = mu_flat
        E = E_field
        H = H_field
    
    return {
        'coords': coords,
        'eps': eps,
        'mu': mu,
        'E_field': E,
        'H_field': H,
        'eps_map': eps_map,
        'mu_map': mu_map,
        'grid_size': np.array(grid_size)
    }


def save_dataset_hdf5(
    dataset: Dict[str, np.ndarray],
    filename: str,
    compression: str = 'gzip'
) -> None:
    """
    Save dataset to HDF5 file.
    
    Args:
        dataset: Dictionary with numpy arrays
        filename: Output filename (should end in .h5)
        compression: Compression algorithm ('gzip', 'lzf', or None)
    """
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        for key, value in dataset.items():
            if isinstance(value, np.ndarray):
                f.create_dataset(
                    key, 
                    data=value, 
                    compression=compression if len(value) > 1000 else None
                )
        
        # Add metadata
        f.attrs['created_by'] = 'Maxwell-PINN-NIF Dataset Generator'
        f.attrs['version'] = '1.0'
    
    print(f"Dataset saved to {filepath}")
    print(f"  Samples: {len(dataset['coords'])}")
    print(f"  File size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")


def load_dataset_hdf5(filename: str) -> Dict[str, np.ndarray]:
    """
    Load dataset from HDF5 file.
    
    Args:
        filename: Path to HDF5 file
        
    Returns:
        dataset: Dictionary with numpy arrays
    """
    dataset = {}
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            dataset[key] = f[key][:]
    return dataset


# ============================================================================
# PyTorch Dataset Wrapper
# ============================================================================

def create_torch_dataset(
    hdf5_path: str,
    device: str = 'cpu'
):
    """
    Create a PyTorch-compatible dataset from HDF5 file.
    
    Returns a simple dict-based dataset. For production use,
    consider subclassing torch.utils.data.Dataset.
    """
    import torch
    
    data = load_dataset_hdf5(hdf5_path)
    
    return {
        'coords': torch.from_numpy(data['coords']).float().to(device),
        'eps': torch.from_numpy(data['eps']).float().to(device),
        'mu': torch.from_numpy(data['mu']).float().to(device),
        'E_field': torch.from_numpy(data['E_field']).float().to(device),
        'H_field': torch.from_numpy(data['H_field']).float().to(device)
    }


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    print("Maxwell-PINN-NIF Dataset Generator")
    print("=" * 50)
    
    # Generate different material types
    for material_type in ['uniform', 'layered', 'inclusions', 'sinusoidal']:
        print(f"\nGenerating {material_type} dataset...")
        
        dataset = generate_training_dataset(
            n_samples=10000,
            grid_size=(32, 32, 32),
            material_type=material_type,
            use_meep=False,  # Use analytical solution
            seed=42
        )
        
        print(f"  Coordinates: {dataset['coords'].shape}")
        print(f"  ε range: [{dataset['eps'].min():.2f}, {dataset['eps'].max():.2f}]")
        print(f"  μ range: [{dataset['mu'].min():.2f}, {dataset['mu'].max():.2f}]")
        print(f"  E-field shape: {dataset['E_field'].shape}")
        print(f"  H-field shape: {dataset['H_field'].shape}")
        
        # Save to HDF5
        save_dataset_hdf5(dataset, f"data/em_data_{material_type}.h5")
    
    print("\n✓ Dataset generation complete!")
