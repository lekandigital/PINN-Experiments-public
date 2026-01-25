"""
Synthetic Cloth Data Generator

Generates synthetic cloth simulation data for testing WITHOUT Blender.
Creates planar cloth meshes with sinusoidal deformations, multi-level
graph representations, and SDF volumes.

Usage:
    python -m src.data.synthetic_data --num_samples 100 --output data/test_data.h5
"""

import numpy as np
import h5py
import argparse
from pathlib import Path
from typing import Tuple, List, Optional
from tqdm import tqdm


def create_cloth_mesh(
    resolution: int = 20,
    size: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a regular grid cloth mesh in the XY plane.
    
    Args:
        resolution: Number of vertices per side (total = resolution^2)
        size: Physical size of the cloth
        
    Returns:
        vertices: (N, 3) vertex positions
        edges: (E, 2) edge indices
    """
    # Create grid vertices
    x = np.linspace(-size/2, size/2, resolution)
    y = np.linspace(-size/2, size/2, resolution)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)
    
    vertices = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=-1)
    
    # Create edges (4-connected grid)
    edges = []
    for i in range(resolution):
        for j in range(resolution):
            idx = i * resolution + j
            # Right neighbor
            if j < resolution - 1:
                edges.append([idx, idx + 1])
            # Bottom neighbor
            if i < resolution - 1:
                edges.append([idx, idx + resolution])
                
    edges = np.array(edges, dtype=np.int32)
    
    return vertices.astype(np.float32), edges


def apply_cloth_deformation(
    vertices: np.ndarray,
    time: float,
    amplitude: float = 0.2,
    frequency_space: float = 2.0,
    frequency_time: float = 1.0,
    gravity: float = 0.1,
    noise_scale: float = 0.02
) -> np.ndarray:
    """
    Apply sinusoidal deformation to simulate cloth dynamics.
    
    Deformation: z = A * sin(ωt) * sin(kx) * sin(ky) - gravity * y + noise
    
    Args:
        vertices: (N, 3) original positions
        time: Simulation time [0, 1]
        amplitude: Deformation amplitude
        frequency_space: Spatial frequency
        frequency_time: Temporal frequency
        gravity: Gravity-induced droop factor
        noise_scale: Random noise scale
        
    Returns:
        deformed: (N, 3) deformed positions
    """
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    
    # Sinusoidal wave deformation
    wave = amplitude * np.sin(2 * np.pi * frequency_time * time)
    wave *= np.sin(frequency_space * np.pi * x) * np.sin(frequency_space * np.pi * y)
    
    # Gravity-induced droop (more droop at bottom of cloth)
    droop = -gravity * (y + 0.5) ** 2  # Parabolic droop
    
    # Random noise for realism
    noise = np.random.randn(len(vertices)) * noise_scale
    
    # Apply deformation to z-axis
    z_new = z + wave + droop + noise
    
    deformed = np.stack([x, y, z_new], axis=-1).astype(np.float32)
    return deformed


def decimate_mesh(
    vertices: np.ndarray,
    edges: np.ndarray,
    stride: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a coarse mesh by sampling every stride-th vertex.
    
    Args:
        vertices: (N, 3) fine mesh vertices
        edges: (E, 2) fine mesh edges
        stride: Decimation stride (keep 1/stride^2 of vertices for grid)
        
    Returns:
        coarse_vertices: (N_coarse, 3) coarse positions
        coarse_edges: (E_coarse, 2) coarse edges
        vertex_map: Mapping from fine to coarse indices
    """
    # Assume vertices are arranged in a grid
    resolution = int(np.sqrt(len(vertices)))
    coarse_res = resolution // stride
    
    # Sample vertices
    coarse_indices = []
    for i in range(0, resolution, stride):
        for j in range(0, resolution, stride):
            if i < resolution and j < resolution:
                idx = i * resolution + j
                coarse_indices.append(idx)
                
    coarse_indices = np.array(coarse_indices)
    coarse_vertices = vertices[coarse_indices]
    
    # Create mapping from fine to coarse
    vertex_map = -np.ones(len(vertices), dtype=np.int32)
    for new_idx, old_idx in enumerate(coarse_indices):
        vertex_map[old_idx] = new_idx
        
    # Filter edges to only include coarse vertices
    coarse_edges = []
    for src, tgt in edges:
        if vertex_map[src] >= 0 and vertex_map[tgt] >= 0:
            coarse_edges.append([vertex_map[src], vertex_map[tgt]])
            
    if len(coarse_edges) == 0:
        # If no edges survived, create edges for coarse grid
        for i in range(coarse_res):
            for j in range(coarse_res):
                idx = i * coarse_res + j
                if j < coarse_res - 1:
                    coarse_edges.append([idx, idx + 1])
                if i < coarse_res - 1:
                    coarse_edges.append([idx, idx + coarse_res])
                    
    coarse_edges = np.array(coarse_edges, dtype=np.int32)
    
    return coarse_vertices, coarse_edges, vertex_map


def compute_sdf_volume(
    vertices: np.ndarray,
    grid_res: int = 32,
    padding: float = 0.2
) -> np.ndarray:
    """
    Compute signed distance field volume using nearest-vertex approximation.
    
    This is a simplified SDF that uses distance to nearest vertex.
    Positive = outside cloth plane, negative = inside (below cloth).
    
    Args:
        vertices: (N, 3) mesh vertices
        grid_res: SDF grid resolution per axis
        padding: Padding around mesh bounding box
        
    Returns:
        sdf: (grid_res, grid_res, grid_res) SDF volume
    """
    # Compute bounding box
    min_bound = vertices.min(axis=0) - padding
    max_bound = vertices.max(axis=0) + padding
    
    # Create 3D grid
    x = np.linspace(min_bound[0], max_bound[0], grid_res)
    y = np.linspace(min_bound[1], max_bound[1], grid_res)
    z = np.linspace(min_bound[2], max_bound[2], grid_res)
    
    # Grid points: (grid_res^3, 3)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    
    # Compute distance to nearest vertex (brute force for simplicity)
    # For large meshes, use KD-tree
    sdf = np.zeros(grid_res ** 3, dtype=np.float32)
    
    for i, point in enumerate(grid_points):
        dists = np.linalg.norm(vertices - point, axis=1)
        min_dist = dists.min()
        
        # Determine sign based on position relative to cloth
        # Simplified: positive if above average z, negative if below
        nearest_idx = dists.argmin()
        nearest_z = vertices[nearest_idx, 2]
        
        if point[2] > nearest_z:
            sdf[i] = min_dist  # Above cloth
        else:
            sdf[i] = -min_dist  # Below cloth
            
    return sdf.reshape(grid_res, grid_res, grid_res)


def compute_sdf_volume_fast(
    vertices: np.ndarray,
    grid_res: int = 32,
    padding: float = 0.2
) -> np.ndarray:
    """
    Fast SDF computation using vectorized operations.
    
    Uses chunked distance computation to avoid memory issues.
    """
    # Compute bounding box
    min_bound = vertices.min(axis=0) - padding
    max_bound = vertices.max(axis=0) + padding
    
    # Create 3D grid
    x = np.linspace(min_bound[0], max_bound[0], grid_res)
    y = np.linspace(min_bound[1], max_bound[1], grid_res)
    z = np.linspace(min_bound[2], max_bound[2], grid_res)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.stack([xx, yy, zz], axis=-1)  # (R, R, R, 3)
    
    # Compute SDF for each grid point
    sdf = np.zeros((grid_res, grid_res, grid_res), dtype=np.float32)
    
    for i in range(grid_res):
        for j in range(grid_res):
            # Process one row at a time
            points = grid_points[i, j]  # (R, 3)
            
            # Distance to all vertices: (R, N)
            dists = np.linalg.norm(
                points[:, np.newaxis, :] - vertices[np.newaxis, :, :],
                axis=-1
            )
            
            min_dists = dists.min(axis=1)  # (R,)
            nearest_idx = dists.argmin(axis=1)  # (R,)
            
            # Sign based on z-position relative to nearest vertex
            nearest_z = vertices[nearest_idx, 2]
            signs = np.sign(points[:, 2] - nearest_z)
            signs[signs == 0] = 1  # Tie-break: positive
            
            sdf[i, j] = signs * min_dists
            
    return sdf


def generate_synthetic_cloth_data(
    num_samples: int = 100,
    fine_resolution: int = 20,
    coarse_stride: int = 2,
    grid_res: int = 32,
    cloth_size: float = 1.0,
    output_path: Optional[str] = None,
    seed: int = 42
) -> dict:
    """
    Generate complete synthetic cloth simulation dataset.
    
    Args:
        num_samples: Number of simulation frames to generate
        fine_resolution: Vertices per side for fine mesh
        coarse_stride: Decimation stride for coarse mesh
        grid_res: SDF grid resolution
        cloth_size: Physical size of cloth
        output_path: Optional path to save HDF5 file
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing all data arrays:
        - fine_positions: (num_samples, N_fine, 3)
        - fine_edges: (E_fine, 2)
        - coarse_positions: (num_samples, N_coarse, 3)
        - coarse_edges: (E_coarse, 2)
        - sdf_volumes: (num_samples, grid_res, grid_res, grid_res)
    """
    np.random.seed(seed)
    
    print(f"Generating synthetic cloth data...")
    print(f"  Fine resolution: {fine_resolution}x{fine_resolution} = {fine_resolution**2} vertices")
    print(f"  Coarse stride: {coarse_stride}")
    print(f"  SDF grid: {grid_res}³")
    print(f"  Samples: {num_samples}")
    
    # Create base mesh
    base_vertices, fine_edges = create_cloth_mesh(fine_resolution, cloth_size)
    
    # Create coarse mesh structure
    _, coarse_edges, _ = decimate_mesh(base_vertices, fine_edges, coarse_stride)
    
    # Storage arrays
    num_fine = fine_resolution ** 2
    num_coarse = (fine_resolution // coarse_stride) ** 2
    
    fine_positions = np.zeros((num_samples, num_fine, 3), dtype=np.float32)
    coarse_positions = np.zeros((num_samples, num_coarse, 3), dtype=np.float32)
    sdf_volumes = np.zeros((num_samples, grid_res, grid_res, grid_res), dtype=np.float32)
    
    # Generate samples with varying deformations
    for i in tqdm(range(num_samples), desc="Generating samples"):
        # Time progresses through simulation
        t = i / max(num_samples - 1, 1)
        
        # Randomize deformation parameters slightly
        amplitude = 0.15 + np.random.rand() * 0.1
        freq_space = 1.5 + np.random.rand() * 1.0
        gravity = 0.05 + np.random.rand() * 0.1
        
        # Apply deformation
        deformed = apply_cloth_deformation(
            base_vertices,
            time=t,
            amplitude=amplitude,
            frequency_space=freq_space,
            gravity=gravity
        )
        
        fine_positions[i] = deformed
        
        # Decimate for coarse mesh
        coarse_verts, _, _ = decimate_mesh(deformed, fine_edges, coarse_stride)
        coarse_positions[i] = coarse_verts
        
        # Compute SDF volume
        sdf = compute_sdf_volume_fast(deformed, grid_res)
        sdf_volumes[i] = sdf
        
    # Compile data dictionary
    data = {
        'fine_positions': fine_positions,
        'fine_edges': fine_edges,
        'coarse_positions': coarse_positions,
        'coarse_edges': coarse_edges,
        'sdf_volumes': sdf_volumes,
        'metadata': {
            'num_samples': num_samples,
            'fine_resolution': fine_resolution,
            'coarse_stride': coarse_stride,
            'grid_res': grid_res,
            'cloth_size': cloth_size
        }
    }
    
    # Save to HDF5 if path provided
    if output_path:
        save_to_hdf5(data, output_path)
        
    return data


def save_to_hdf5(data: dict, output_path: str) -> None:
    """Save data dictionary to HDF5 file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Save arrays
        f.create_dataset('fine_positions', data=data['fine_positions'], compression='gzip')
        f.create_dataset('fine_edges', data=data['fine_edges'], compression='gzip')
        f.create_dataset('coarse_positions', data=data['coarse_positions'], compression='gzip')
        f.create_dataset('coarse_edges', data=data['coarse_edges'], compression='gzip')
        f.create_dataset('sdf_volumes', data=data['sdf_volumes'], compression='gzip')
        
        # Save metadata as attributes
        for key, value in data['metadata'].items():
            f.attrs[key] = value
            
    print(f"Saved data to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")


def load_from_hdf5(input_path: str) -> dict:
    """Load data dictionary from HDF5 file."""
    with h5py.File(input_path, 'r') as f:
        data = {
            'fine_positions': f['fine_positions'][:],
            'fine_edges': f['fine_edges'][:],
            'coarse_positions': f['coarse_positions'][:],
            'coarse_edges': f['coarse_edges'][:],
            'sdf_volumes': f['sdf_volumes'][:],
            'metadata': dict(f.attrs)
        }
    return data


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic cloth data')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--fine_resolution', type=int, default=20, help='Fine mesh resolution')
    parser.add_argument('--coarse_stride', type=int, default=2, help='Coarse mesh stride')
    parser.add_argument('--grid_res', type=int, default=32, help='SDF grid resolution')
    parser.add_argument('--output', type=str, default='data/test_data.h5', help='Output path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    generate_synthetic_cloth_data(
        num_samples=args.num_samples,
        fine_resolution=args.fine_resolution,
        coarse_stride=args.coarse_stride,
        grid_res=args.grid_res,
        output_path=args.output,
        seed=args.seed
    )
    
    print("\n✓ Data generation complete!")


if __name__ == '__main__':
    main()
