"""
NIF-Cloth4D: Synthetic Data Generator

This module generates synthetic SDF data for testing the NIF-Cloth4D network
when Blender simulation data is not available. It creates analytically-defined
deforming surfaces (e.g., sine-wave cloth) with known SDF values.
"""

import numpy as np
import h5py
import os
from typing import Tuple, Optional
from pathlib import Path


def create_wavy_cloth_sdf(
    grid_size: int = 64,
    t: float = 0.0,
    amplitude: float = 0.3,
    frequency: float = 2.0,
    cloth_thickness: float = 0.02,
    bounds: Tuple[float, float] = (-1.0, 1.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create SDF for a wavy cloth surface (z = A*sin(f*x + phase) + A*sin(f*y + phase)).
    
    The cloth is modeled as a thin shell with a small thickness parameter.
    The SDF is approximately the distance to the surface.
    
    Args:
        grid_size: Resolution of the SDF volume (grid_size^3)
        t: Time parameter (controls wave phase)
        amplitude: Wave amplitude
        frequency: Wave frequency (higher = more wrinkles)
        cloth_thickness: Half-thickness of the cloth (for SDF)
        bounds: Min/max coordinate bounds
    
    Returns:
        sdf_volume: SDF values as (grid_size, grid_size, grid_size) array
        grid_coords: Coordinate arrays for each axis
    """
    # Create 3D coordinate grid
    coords = np.linspace(bounds[0], bounds[1], grid_size)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    
    # Define wavy cloth surface: z = A*sin(f*x + t) + A*sin(f*y + t)
    # Add time-dependent phase for animation
    phase = t * 2 * np.pi
    surface_z = amplitude * np.sin(frequency * X + phase) + \
                amplitude * np.sin(frequency * Y + phase * 0.7)
    
    # Add slight draping effect (lower in center)
    drape = 0.1 * (X**2 + Y**2)
    surface_z = surface_z - drape
    
    # Compute approximate SDF: distance to the surface
    # For a thin surface, SDF ≈ |z - surface_z| - thickness/2
    distance_to_surface = np.abs(Z - surface_z)
    sdf = distance_to_surface - cloth_thickness
    
    return sdf.astype(np.float32), coords.astype(np.float32)


def create_sphere_sdf(
    grid_size: int = 64,
    t: float = 0.0,
    radius: float = 0.5,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    bounds: Tuple[float, float] = (-1.0, 1.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create SDF for a sphere (useful for simple testing).
    
    Args:
        grid_size: Resolution of the SDF volume
        t: Time parameter (can modulate radius for animation)
        radius: Base sphere radius
        center: Sphere center (x, y, z)
        bounds: Coordinate bounds
    
    Returns:
        sdf_volume: SDF values
        grid_coords: Coordinate arrays
    """
    coords = np.linspace(bounds[0], bounds[1], grid_size)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    
    # Time-varying radius (pulsing sphere)
    r = radius * (1.0 + 0.2 * np.sin(2 * np.pi * t))
    
    # Sphere SDF: distance to center - radius
    dist = np.sqrt(
        (X - center[0])**2 + 
        (Y - center[1])**2 + 
        (Z - center[2])**2
    )
    sdf = dist - r
    
    return sdf.astype(np.float32), coords.astype(np.float32)


def create_falling_cloth_sdf(
    grid_size: int = 64,
    t: float = 0.0,
    bounds: Tuple[float, float] = (-1.0, 1.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create SDF for a cloth that falls and drapes over time.
    
    Simulates a cloth starting flat at the top, then falling and wrinkling.
    
    Args:
        grid_size: Resolution of the SDF volume
        t: Time parameter [0, 1] where 0=start, 1=fully draped
        bounds: Coordinate bounds
    
    Returns:
        sdf_volume: SDF values
        grid_coords: Coordinate arrays
    """
    coords = np.linspace(bounds[0], bounds[1], grid_size)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    
    # Interpolate between flat and wavy/draped state
    t_clamped = np.clip(t, 0.0, 1.0)
    
    # Flat cloth at z=0.8 (top)
    flat_z = 0.8
    
    # Draped cloth with wrinkles
    wrinkle_amp = 0.2 * t_clamped
    drape_depth = 0.6 * t_clamped
    draped_z = flat_z - drape_depth + \
               wrinkle_amp * np.sin(4 * X) * np.sin(4 * Y) - \
               0.2 * t_clamped * (X**2 + Y**2)
    
    # Interpolate surface height
    surface_z = (1 - t_clamped) * flat_z + t_clamped * draped_z
    
    # Cloth thickness
    thickness = 0.02
    
    # SDF
    distance_to_surface = np.abs(Z - surface_z)
    sdf = distance_to_surface - thickness
    
    return sdf.astype(np.float32), coords.astype(np.float32)


def save_sdf_to_hdf5(
    sdf: np.ndarray,
    coords: np.ndarray,
    filepath: str,
    time: float,
    bounds: Tuple[float, float] = (-1.0, 1.0),
    metadata: Optional[dict] = None
):
    """
    Save SDF volume to HDF5 file.
    
    Args:
        sdf: SDF volume array
        coords: Coordinate grid
        filepath: Output file path
        time: Time value for this frame
        bounds: Coordinate bounds
        metadata: Optional metadata dictionary
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('sdf', data=sdf, compression='gzip')
        f.create_dataset('coords', data=coords)
        f.attrs['time'] = time
        f.attrs['bounds_min'] = bounds[0]
        f.attrs['bounds_max'] = bounds[1]
        f.attrs['grid_size'] = sdf.shape[0]
        
        if metadata:
            for key, value in metadata.items():
                f.attrs[key] = value


def load_sdf_from_hdf5(filepath: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load SDF volume from HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
    
    Returns:
        sdf: SDF volume array
        coords: Coordinate grid
        attrs: Dictionary of file attributes
    """
    with h5py.File(filepath, 'r') as f:
        sdf = f['sdf'][:]
        coords = f['coords'][:]
        attrs = dict(f.attrs)
    return sdf, coords, attrs


def generate_dataset(
    output_dir: str,
    num_frames: int = 10,
    grid_size: int = 64,
    data_type: str = 'wavy',
    bounds: Tuple[float, float] = (-1.0, 1.0)
) -> list:
    """
    Generate a complete dataset of SDF frames.
    
    Args:
        output_dir: Output directory for HDF5 files
        num_frames: Number of time frames to generate
        grid_size: Resolution of each SDF volume
        data_type: Type of data ('wavy', 'sphere', 'falling')
        bounds: Coordinate bounds
    
    Returns:
        List of generated file paths
    """
    output_path = Path(output_dir)
    sdf_dir = output_path / 'sdf'
    sdf_dir.mkdir(parents=True, exist_ok=True)
    
    files = []
    times = np.linspace(0, 1, num_frames)
    
    print(f"Generating {num_frames} SDF frames ({data_type})...")
    print(f"Grid size: {grid_size}³")
    print(f"Output directory: {sdf_dir}")
    
    for i, t in enumerate(times):
        # Generate SDF based on type
        if data_type == 'wavy':
            sdf, coords = create_wavy_cloth_sdf(
                grid_size=grid_size, t=t, bounds=bounds
            )
        elif data_type == 'sphere':
            sdf, coords = create_sphere_sdf(
                grid_size=grid_size, t=t, bounds=bounds
            )
        elif data_type == 'falling':
            sdf, coords = create_falling_cloth_sdf(
                grid_size=grid_size, t=t, bounds=bounds
            )
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        # Save to HDF5
        filepath = sdf_dir / f'frame_{i:04d}.h5'
        save_sdf_to_hdf5(
            sdf=sdf,
            coords=coords,
            filepath=str(filepath),
            time=t,
            bounds=bounds,
            metadata={'frame_index': i, 'data_type': data_type}
        )
        files.append(str(filepath))
        
        # Progress
        print(f"  Frame {i+1}/{num_frames}: t={t:.3f}, "
              f"SDF range=[{sdf.min():.3f}, {sdf.max():.3f}]")
    
    # Save metadata
    metadata_path = output_path / 'metadata.txt'
    with open(metadata_path, 'w') as f:
        f.write(f"Dataset: NIF-Cloth4D Synthetic Data\n")
        f.write(f"Type: {data_type}\n")
        f.write(f"Frames: {num_frames}\n")
        f.write(f"Grid size: {grid_size}\n")
        f.write(f"Bounds: [{bounds[0]}, {bounds[1]}]\n")
        f.write(f"Files:\n")
        for fp in files:
            f.write(f"  {fp}\n")
    
    print(f"\nDataset generated: {len(files)} files")
    print(f"Metadata saved to: {metadata_path}")
    
    return files


def sample_points_from_sdf(
    sdf: np.ndarray,
    coords: np.ndarray,
    time: float,
    n_samples: int = 10000,
    near_surface_ratio: float = 0.7,
    surface_band: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample training points from an SDF volume.
    
    Uses a mixed sampling strategy:
    - near_surface_ratio of points are sampled near the surface (|SDF| < band)
    - Remaining points are uniformly sampled in the volume
    
    Args:
        sdf: SDF volume array
        coords: Coordinate grid
        time: Time value for this frame
        n_samples: Total number of samples
        near_surface_ratio: Fraction of samples near surface
        surface_band: SDF threshold for "near surface"
    
    Returns:
        points: Sample coordinates (N, 4) with (x, y, z, t)
        sdf_values: SDF values at sample points (N, 1)
    """
    grid_size = len(coords)
    bounds = (coords[0], coords[-1])
    
    # Number of samples for each strategy
    n_near = int(n_samples * near_surface_ratio)
    n_uniform = n_samples - n_near
    
    # Find near-surface voxels
    near_surface_mask = np.abs(sdf) < surface_band
    near_indices = np.argwhere(near_surface_mask)
    
    # Sample near-surface points
    if len(near_indices) > 0 and n_near > 0:
        # Random selection of near-surface voxels
        idx = np.random.choice(len(near_indices), size=min(n_near, len(near_indices)), replace=True)
        near_voxels = near_indices[idx]
        
        # Convert to world coordinates with small random offset
        near_points = np.zeros((len(near_voxels), 4))
        for i, (ix, iy, iz) in enumerate(near_voxels):
            # Add small random offset within voxel
            dx = np.random.uniform(-0.5, 0.5) * (bounds[1] - bounds[0]) / grid_size
            dy = np.random.uniform(-0.5, 0.5) * (bounds[1] - bounds[0]) / grid_size
            dz = np.random.uniform(-0.5, 0.5) * (bounds[1] - bounds[0]) / grid_size
            near_points[i] = [coords[ix] + dx, coords[iy] + dy, coords[iz] + dz, time]
        
        # Get SDF values (interpolate at offset positions - approximate with voxel value)
        near_sdf = sdf[near_voxels[:, 0], near_voxels[:, 1], near_voxels[:, 2]]
    else:
        near_points = np.zeros((0, 4))
        near_sdf = np.zeros(0)
    
    # Sample uniform points
    uniform_points = np.random.uniform(bounds[0], bounds[1], size=(n_uniform, 3))
    uniform_points = np.hstack([uniform_points, np.full((n_uniform, 1), time)])
    
    # Get SDF values for uniform points (nearest neighbor interpolation)
    # Convert to grid indices
    scale = (grid_size - 1) / (bounds[1] - bounds[0])
    indices = ((uniform_points[:, :3] - bounds[0]) * scale).astype(int)
    indices = np.clip(indices, 0, grid_size - 1)
    uniform_sdf = sdf[indices[:, 0], indices[:, 1], indices[:, 2]]
    
    # Combine
    points = np.vstack([near_points, uniform_points]) if len(near_points) > 0 else uniform_points
    sdf_values = np.concatenate([near_sdf, uniform_sdf]) if len(near_sdf) > 0 else uniform_sdf
    
    # Shuffle
    perm = np.random.permutation(len(points))
    points = points[perm]
    sdf_values = sdf_values[perm]
    
    return points.astype(np.float32), sdf_values.reshape(-1, 1).astype(np.float32)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic SDF data")
    parser.add_argument('--output_dir', type=str, default='/tmp/cloth_test_data',
                        help='Output directory')
    parser.add_argument('--num_frames', type=int, default=10,
                        help='Number of frames to generate')
    parser.add_argument('--grid_size', type=int, default=64,
                        help='Grid resolution')
    parser.add_argument('--data_type', type=str, default='falling',
                        choices=['wavy', 'sphere', 'falling'],
                        help='Type of synthetic data')
    args = parser.parse_args()
    
    files = generate_dataset(
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        grid_size=args.grid_size,
        data_type=args.data_type
    )
    
    # Test loading
    print("\nTesting data loading...")
    sdf, coords, attrs = load_sdf_from_hdf5(files[0])
    print(f"Loaded SDF shape: {sdf.shape}")
    print(f"Attributes: {attrs}")
    
    # Test sampling
    print("\nTesting point sampling...")
    points, sdf_vals = sample_points_from_sdf(sdf, coords, attrs['time'])
    print(f"Sampled {len(points)} points")
    print(f"Points shape: {points.shape}")
    print(f"SDF values shape: {sdf_vals.shape}")
