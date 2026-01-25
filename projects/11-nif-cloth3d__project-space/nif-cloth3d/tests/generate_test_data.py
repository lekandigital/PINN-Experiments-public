"""
NIF-Cloth3D-Interactive: Generate Test Data

Generate synthetic cloth mesh data for training and testing.
"""

import sys
from pathlib import Path
import numpy as np
import torch


def generate_cloth_mesh(
    resolution: int = 50,
    size: float = 2.0,
    center: tuple = (0.0, 0.0, 0.0)
):
    """
    Generate a simple plane mesh for cloth simulation.
    
    Args:
        resolution: Grid resolution (resolution x resolution vertices)
        size: Physical size of cloth
        center: Center position of cloth
    
    Returns:
        vertices: (N, 3) vertex positions
        edges: (E, 2) edge indices
        faces: (F, 3) triangle face indices
    """
    # Create grid of vertices
    x = np.linspace(-size/2, size/2, resolution) + center[0]
    y = np.linspace(-size/2, size/2, resolution) + center[1]
    z = np.zeros((resolution, resolution)) + center[2]
    
    xx, yy = np.meshgrid(x, y)
    vertices = np.stack([xx.flatten(), yy.flatten(), z.flatten()], axis=1)
    
    # Create edges (horizontal, vertical, and diagonal)
    edges = []
    
    # Horizontal edges
    for i in range(resolution):
        for j in range(resolution - 1):
            idx = i * resolution + j
            edges.append([idx, idx + 1])
    
    # Vertical edges
    for i in range(resolution - 1):
        for j in range(resolution):
            idx = i * resolution + j
            edges.append([idx, idx + resolution])
    
    # Diagonal edges (for shear stiffness)
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx = i * resolution + j
            edges.append([idx, idx + resolution + 1])  # Down-right
            edges.append([idx + 1, idx + resolution])  # Down-left
    
    edges = np.array(edges, dtype=np.int64)
    
    # Create triangle faces
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx = i * resolution + j
            # Two triangles per quad
            faces.append([idx, idx + 1, idx + resolution + 1])
            faces.append([idx, idx + resolution + 1, idx + resolution])
    
    faces = np.array(faces, dtype=np.int64)
    
    return vertices, edges, faces


def generate_draped_cloth(
    resolution: int = 50,
    size: float = 2.0,
    drape_amount: float = 0.3
):
    """
    Generate a cloth with some initial drape (non-flat).
    
    Args:
        resolution: Grid resolution
        size: Physical size
        drape_amount: Amount of drape deformation
    
    Returns:
        vertices, edges, faces
    """
    vertices, edges, faces = generate_cloth_mesh(resolution, size)
    
    # Add some drape (parabolic sag in the middle)
    N = vertices.shape[0]
    for i in range(N):
        x, y = vertices[i, 0], vertices[i, 1]
        # Distance from center
        r = np.sqrt(x**2 + y**2)
        # Parabolic drape
        vertices[i, 2] = -drape_amount * (1 - (r / (size/2))**2)
    
    return vertices, edges, faces


def generate_pinned_cloth_data(
    resolution: int = 50,
    size: float = 2.0,
    n_samples: int = 100,
    force_range: tuple = (0.0, 1.0)
):
    """
    Generate dataset with pinned corners and varying forces.
    
    Returns dictionary ready for torch.save()
    """
    vertices, edges, faces = generate_cloth_mesh(resolution, size)
    
    # Identify pinned vertices (top row)
    pinned_indices = list(range(resolution))  # First row
    
    # Rest lengths for all edges
    rest_lengths = np.linalg.norm(
        vertices[edges[:, 0]] - vertices[edges[:, 1]], 
        axis=1
    )
    
    # Generate random force samples
    forces = np.random.uniform(
        force_range[0], force_range[1], 
        size=(n_samples, 3)
    ).astype(np.float32)
    
    # Generate random time values
    times = np.random.uniform(0, 1, size=(n_samples,)).astype(np.float32)
    
    # Generate random material IDs
    material_ids = np.random.randint(0, 5, size=(n_samples,))
    
    return {
        'vertices': torch.tensor(vertices, dtype=torch.float32),
        'edges': torch.tensor(edges, dtype=torch.long),
        'faces': torch.tensor(faces, dtype=torch.long),
        'rest_lengths': torch.tensor(rest_lengths, dtype=torch.float32),
        'pinned_indices': pinned_indices,
        'forces': torch.tensor(forces, dtype=torch.float32),
        'times': torch.tensor(times, dtype=torch.float32),
        'material_ids': torch.tensor(material_ids, dtype=torch.long),
        'resolution': resolution,
        'size': size
    }


def main():
    """Generate and save test data."""
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate basic cloth
    print("Generating cloth mesh...")
    vertices, edges, faces = generate_cloth_mesh(resolution=50)
    
    basic_data = {
        'vertices': torch.tensor(vertices, dtype=torch.float32),
        'edges': torch.tensor(edges, dtype=torch.long),
        'faces': torch.tensor(faces, dtype=torch.long)
    }
    
    basic_path = output_dir / 'test_cloth.pt'
    torch.save(basic_data, basic_path)
    print(f"Saved basic cloth: {basic_path}")
    print(f"  Vertices: {vertices.shape[0]}")
    print(f"  Edges: {edges.shape[0]}")
    print(f"  Faces: {faces.shape[0]}")
    
    # Generate training dataset
    print("\nGenerating training dataset...")
    train_data = generate_pinned_cloth_data(
        resolution=50,
        size=2.0,
        n_samples=1000,
        force_range=(0.0, 1.0)
    )
    
    train_path = output_dir / 'train_cloth.pt'
    torch.save(train_data, train_path)
    print(f"Saved training data: {train_path}")
    print(f"  Samples: {len(train_data['forces'])}")
    
    # Generate high-res cloth for L40S
    print("\nGenerating high-res cloth (L40S)...")
    vertices_hr, edges_hr, faces_hr = generate_cloth_mesh(resolution=100)
    
    hr_data = {
        'vertices': torch.tensor(vertices_hr, dtype=torch.float32),
        'edges': torch.tensor(edges_hr, dtype=torch.long),
        'faces': torch.tensor(faces_hr, dtype=torch.long)
    }
    
    hr_path = output_dir / 'cloth_highres.pt'
    torch.save(hr_data, hr_path)
    print(f"Saved high-res cloth: {hr_path}")
    print(f"  Vertices: {vertices_hr.shape[0]}")
    print(f"  Edges: {edges_hr.shape[0]}")
    
    print("\n✅ Test data generation complete!")


if __name__ == '__main__':
    main()
