"""
PyFlex-based cloth simulation data generator.

Generates training data for NIF-Cloth4D-Temporal by simulating cloth with
NVIDIA PyFlex and computing volumetric SDF supervision signals.

PyFlex is NVIDIA's position-based dynamics library for cloth simulation.
If PyFlex is not available, a synthetic fallback generator is provided.

Requirements:
    - PyFlex (https://github.com/YunzhuLi/PyFlex)
    - PyVista for SDF computation
"""

import os
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Tuple, Dict
import warnings

# Optional imports
try:
    import pyflex
    PYFLEX_AVAILABLE = True
except ImportError:
    PYFLEX_AVAILABLE = False
    warnings.warn("PyFlex not available. Use generate_synthetic_dataset() instead.")

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False


def generate_cloth_dataset(
    output_path: str = 'data/processed/cloth_sdf_dataset.h5',
    cloth_res: int = 32,
    num_frames: int = 120,
    fps: float = 30.0,
    cloth_size: float = 1.0,
    gravity: Tuple[float, float, float] = (0.0, -9.8, 0.0),
    wind_magnitude: float = 5.0,
    wind_start_frame: int = 10,
    wind_end_frame: int = 70,
    sdf_resolution: int = 64,
    sdf_margin: float = 0.1,
    fix_top_edge: bool = True,
) -> str:
    """
    Generate cloth simulation dataset with volumetric SDF supervision.
    
    Simulates a cloth grid using PyFlex with gravity and periodic wind,
    then computes volumetric SDF for each frame.
    
    Args:
        output_path: Path to save HDF5 dataset
        cloth_res: Cloth grid resolution (cloth_res x cloth_res vertices)
        num_frames: Number of simulation frames (4 seconds at 30fps = 120 frames)
        fps: Simulation framerate
        cloth_size: Physical size of cloth (meters)
        gravity: Gravity vector (x, y, z)
        wind_magnitude: Maximum wind force magnitude
        wind_start_frame: Frame when wind starts
        wind_end_frame: Frame when wind ends
        sdf_resolution: Volumetric SDF grid resolution (64³)
        sdf_margin: Margin around cloth for SDF grid
        fix_top_edge: Whether to fix the top edge of the cloth
        
    Returns:
        Path to saved dataset
    """
    if not PYFLEX_AVAILABLE:
        raise RuntimeError(
            "PyFlex not available. Please install from: "
            "https://github.com/YunzhuLi/PyFlex\n"
            "Or use generate_synthetic_dataset() for testing."
        )
    
    if not PYVISTA_AVAILABLE:
        raise RuntimeError("PyVista required for SDF computation. Install with: pip install pyvista")
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating cloth dataset: {cloth_res}x{cloth_res} cloth, {num_frames} frames")
    
    # Initialize PyFlex
    pyflex.init()
    
    # Set up cloth simulation
    # PyFlex cloth scene parameters
    cloth_params = {
        'cloth_pos': [0.0, 1.5, 0.0],  # Starting position (center, height, center)
        'cloth_size': [cloth_size, cloth_size],
        'cloth_res': [cloth_res, cloth_res],
        'stretch_stiffness': 0.9,
        'bend_stiffness': 0.5,
        'shear_stiffness': 0.9,
        'mass': 0.5 / (cloth_res * cloth_res),  # Total mass distributed
    }
    
    # Create cloth scene
    pyflex.set_scene(
        0,  # Cloth scene ID
        cloth_params,
    )
    
    # Get initial state
    num_particles = pyflex.get_n_particles()
    num_vertices = cloth_res * cloth_res
    
    # Fix top edge vertices
    if fix_top_edge:
        inv_mass = pyflex.get_inv_mass()
        for j in range(cloth_res):
            idx = j  # Top row indices
            inv_mass[idx] = 0.0  # Zero inverse mass = fixed
        pyflex.set_inv_mass(inv_mass)
    
    # Storage
    all_vertices = []
    all_sdf_vols = []
    timestamps = []
    
    dt = 1.0 / fps
    
    # Simulation loop
    for frame in range(num_frames):
        t = frame * dt
        timestamps.append(t)
        
        # Apply wind force (sinusoidal)
        if wind_start_frame <= frame <= wind_end_frame:
            wind_x = wind_magnitude * np.sin(frame / 10.0)
            wind_force = [wind_x, 0.0, 0.0]
        else:
            wind_force = [0.0, 0.0, 0.0]
        
        # Apply gravity + wind
        pyflex.set_gravity([
            gravity[0] + wind_force[0],
            gravity[1] + wind_force[1],
            gravity[2] + wind_force[2],
        ])
        
        # Step simulation
        pyflex.step()
        
        # Get vertex positions
        positions = pyflex.get_positions().reshape(-1, 4)[:num_vertices, :3]
        all_vertices.append(positions.copy())
        
        # Compute volumetric SDF
        sdf_vol = compute_mesh_sdf(
            vertices=positions,
            faces=get_cloth_faces(cloth_res),
            resolution=sdf_resolution,
            margin=sdf_margin,
        )
        all_sdf_vols.append(sdf_vol)
        
        if (frame + 1) % 10 == 0:
            print(f"  Frame {frame + 1}/{num_frames}")
    
    # Clean up PyFlex
    pyflex.clean()
    
    # Save to HDF5
    print(f"Saving dataset to {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        # Metadata
        f.attrs['cloth_res'] = cloth_res
        f.attrs['num_frames'] = num_frames
        f.attrs['fps'] = fps
        f.attrs['sdf_resolution'] = sdf_resolution
        f.attrs['cloth_size'] = cloth_size
        
        # SDF volumes: (num_frames, res, res, res)
        f.create_dataset(
            'sdf_vol',
            data=np.stack(all_sdf_vols, axis=0),
            dtype='float32',
            compression='gzip',
            compression_opts=4,
        )
        
        # Timestamps
        f.create_dataset('time', data=np.array(timestamps), dtype='float32')
        
        # Per-frame vertices
        vertices_grp = f.create_group('vertices')
        for i, verts in enumerate(all_vertices):
            vertices_grp.create_dataset(f'frame_{i}', data=verts, dtype='float32')
        
        # Faces (same for all frames)
        faces = get_cloth_faces(cloth_res)
        f.create_dataset('faces', data=faces, dtype='int32')
    
    print(f"Dataset saved: {output_path}")
    print(f"  - SDF volumes: ({num_frames}, {sdf_resolution}, {sdf_resolution}, {sdf_resolution})")
    print(f"  - Vertices per frame: {num_vertices}")
    
    return str(output_path)


def get_cloth_faces(cloth_res: int) -> np.ndarray:
    """Generate triangle faces for a regular grid cloth mesh."""
    faces = []
    for i in range(cloth_res - 1):
        for j in range(cloth_res - 1):
            v00 = i * cloth_res + j
            v01 = v00 + 1
            v10 = v00 + cloth_res
            v11 = v10 + 1
            
            # Two triangles per quad
            faces.append([v00, v10, v01])
            faces.append([v01, v10, v11])
    
    return np.array(faces, dtype=np.int32)


def compute_mesh_sdf(
    vertices: np.ndarray,
    faces: np.ndarray,
    resolution: int = 64,
    margin: float = 0.1,
) -> np.ndarray:
    """
    Compute volumetric SDF for a mesh using PyVista.
    
    Args:
        vertices: Mesh vertices (N, 3)
        faces: Triangle indices (F, 3)
        resolution: Output grid resolution
        margin: Margin around mesh bounding box
        
    Returns:
        SDF volume (resolution, resolution, resolution)
    """
    if not PYVISTA_AVAILABLE:
        # Fallback: return approximate SDF based on distance to mesh center
        return np.zeros((resolution, resolution, resolution), dtype=np.float32)
    
    # Create PyVista mesh
    # Convert faces to VTK format (prepend face size)
    vtk_faces = np.hstack([
        np.full((faces.shape[0], 1), 3, dtype=np.int32),
        faces
    ]).flatten()
    
    mesh = pv.PolyData(vertices, vtk_faces)
    
    # Compute bounding box with margin
    bounds = mesh.bounds
    x_min, x_max = bounds[0] - margin, bounds[1] + margin
    y_min, y_max = bounds[2] - margin, bounds[3] + margin
    z_min, z_max = bounds[4] - margin, bounds[5] + margin
    
    # Create uniform grid
    grid = pv.UniformGrid(
        dimensions=(resolution, resolution, resolution),
        spacing=(
            (x_max - x_min) / (resolution - 1),
            (y_max - y_min) / (resolution - 1),
            (z_max - z_min) / (resolution - 1),
        ),
        origin=(x_min, y_min, z_min),
    )
    
    # Compute signed distance
    # Using implicit distance (negative inside, positive outside)
    distance = grid.compute_implicit_distance(mesh)
    sdf_vol = distance['implicit_distance'].reshape(resolution, resolution, resolution)
    
    return sdf_vol.astype(np.float32)


def generate_synthetic_dataset(
    output_path: str = 'data/processed/synthetic_cloth_dataset.h5',
    cloth_res: int = 32,
    num_frames: int = 120,
    fps: float = 30.0,
    cloth_size: float = 1.0,
    num_samples_per_frame: int = 4096,
    seed: int = 42,
) -> str:
    """
    Generate synthetic cloth-like dataset for testing without PyFlex.
    
    Creates a procedurally animated cloth surface using sine waves,
    useful for testing the training pipeline.
    
    Args:
        output_path: Path to save HDF5 dataset
        cloth_res: Cloth grid resolution
        num_frames: Number of frames
        fps: Framerate
        cloth_size: Physical size of cloth
        num_samples_per_frame: Number of SDF sample points per frame
        seed: Random seed for reproducibility
        
    Returns:
        Path to saved dataset
    """
    np.random.seed(seed)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating synthetic cloth dataset (no PyFlex required)")
    
    num_vertices = cloth_res * cloth_res
    dt = 1.0 / fps
    
    # Create rest-state grid
    x = np.linspace(-cloth_size / 2, cloth_size / 2, cloth_res)
    z = np.linspace(-cloth_size / 2, cloth_size / 2, cloth_res)
    xx, zz = np.meshgrid(x, z)
    rest_x = xx.flatten()
    rest_z = zz.flatten()
    rest_y = np.ones_like(rest_x) * 1.0  # Start at y=1
    
    all_vertices = []
    all_sdf_samples = []
    timestamps = []
    
    for frame in range(num_frames):
        t = frame * dt
        timestamps.append(t)
        
        # Procedural animation: gravity + wave motion
        # Gravity effect (top edge fixed, bottom falls)
        row_idx = np.arange(num_vertices) // cloth_res
        gravity_factor = row_idx / (cloth_res - 1)  # 0 at top, 1 at bottom
        
        # Wave motion
        wave_freq = 2.0 * np.pi
        wave_amp = 0.1
        wave = wave_amp * np.sin(wave_freq * t + rest_x * 3.0)
        wave *= gravity_factor  # More motion at bottom
        
        # Compute deformed positions
        y_offset = -0.3 * gravity_factor * min(t, 1.0)  # Gravity over first second
        
        curr_x = rest_x + wave * 0.2
        curr_y = rest_y + y_offset + wave * 0.5
        curr_z = rest_z + wave * 0.1
        
        vertices = np.stack([curr_x, curr_y, curr_z], axis=1).astype(np.float32)
        all_vertices.append(vertices)
        
        # Generate SDF samples around the surface
        # Sample points near vertices with noise
        sample_points = []
        sample_sdf = []
        
        for _ in range(num_samples_per_frame):
            # Random vertex
            v_idx = np.random.randint(num_vertices)
            v_pos = vertices[v_idx]
            
            # Random offset
            offset = np.random.randn(3) * 0.05
            point = v_pos + offset
            
            # Approximate SDF as distance to nearest vertex
            dists = np.linalg.norm(vertices - point, axis=1)
            min_dist = dists.min()
            
            # Sign: positive if "outside" (simplified heuristic)
            sign = 1.0 if offset[1] > 0 else -1.0
            sdf_value = sign * min_dist
            
            sample_points.append(point)
            sample_sdf.append(sdf_value)
        
        # Combine xyz and sdf
        sdf_samples = np.concatenate([
            np.array(sample_points, dtype=np.float32),
            np.array(sample_sdf, dtype=np.float32).reshape(-1, 1),
        ], axis=1)  # (num_samples, 4)
        
        all_sdf_samples.append(sdf_samples)
        
        if (frame + 1) % 20 == 0:
            print(f"  Frame {frame + 1}/{num_frames}")
    
    # Save to HDF5
    print(f"Saving synthetic dataset to {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        # Metadata
        f.attrs['cloth_res'] = cloth_res
        f.attrs['num_frames'] = num_frames
        f.attrs['fps'] = fps
        f.attrs['num_samples_per_frame'] = num_samples_per_frame
        f.attrs['cloth_size'] = cloth_size
        f.attrs['synthetic'] = True
        
        # SDF samples: (num_frames, num_samples, 4)
        f.create_dataset(
            'sdf_samples',
            data=np.stack(all_sdf_samples, axis=0),
            dtype='float32',
            compression='gzip',
        )
        
        # Timestamps
        f.create_dataset('time', data=np.array(timestamps), dtype='float32')
        
        # Vertices per frame
        vertices_grp = f.create_group('vertices')
        for i, verts in enumerate(all_vertices):
            vertices_grp.create_dataset(f'frame_{i}', data=verts, dtype='float32')
        
        # Faces
        faces = get_cloth_faces(cloth_res)
        f.create_dataset('faces', data=faces, dtype='int32')
    
    print(f"Synthetic dataset saved: {output_path}")
    print(f"  - Frames: {num_frames}")
    print(f"  - Samples per frame: {num_samples_per_frame}")
    
    return str(output_path)


if __name__ == '__main__':
    # Generate test dataset
    if PYFLEX_AVAILABLE:
        generate_cloth_dataset()
    else:
        print("PyFlex not available, generating synthetic dataset...")
        generate_synthetic_dataset()
