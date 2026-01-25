"""
Generate minimal synthetic test data for ClothGNN.
Creates a small square cloth mesh with simple dynamics.
"""
import torch
import numpy as np
import h5py
import os


def create_square_mesh(resolution=10):
    """
    Create a square cloth mesh with specified resolution.
    Returns vertices (positions) and face connectivity.
    
    Args:
        resolution: Number of vertices per side
        
    Returns:
        vertices: (N, 3) tensor of vertex positions
        faces: (T, 3) tensor of triangle face indices
    """
    # Generate grid vertices
    x = np.linspace(-0.5, 0.5, resolution)
    y = np.linspace(0, 1, resolution)  # Cloth hangs from y=1
    xx, yy = np.meshgrid(x, y)
    vertices = np.stack([xx.flatten(), yy.flatten(), np.zeros_like(xx.flatten())], axis=1)
    
    # Generate triangular faces
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            v0 = i * resolution + j
            v1 = v0 + 1
            v2 = v0 + resolution
            v3 = v2 + 1
            # Two triangles per quad
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    
    return torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.long)


def create_edges_from_faces(faces, num_vertices):
    """
    Create edge_index from triangle faces.
    
    Args:
        faces: (T, 3) triangle indices
        num_vertices: Total number of vertices
        
    Returns:
        edge_index: (2, E) edge connectivity (undirected)
    """
    edges = set()
    for face in faces.numpy():
        # Add edges for each triangle
        for i in range(3):
            v0 = face[i]
            v1 = face[(i + 1) % 3]
            # Ensure consistent ordering
            edge = tuple(sorted([v0, v1]))
            edges.add(edge)
    
    # Convert to bidirectional edge_index
    edge_list = []
    for v0, v1 in edges:
        edge_list.append([v0, v1])
        edge_list.append([v1, v0])  # Undirected
    
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()


def simulate_simple_gravity(vertices, faces, num_frames=60, dt=1/60):
    """
    Simple gravity simulation for test data.
    Top row is pinned, rest falls under gravity.
    
    Args:
        vertices: (N, 3) initial positions
        faces: (T, 3) triangle faces
        num_frames: Number of frames to simulate
        dt: Time step
        
    Returns:
        trajectory: (num_frames+1, N, 3) tensor of positions over time
    """
    num_vertices = vertices.shape[0]
    resolution = int(np.sqrt(num_vertices))
    
    # Identify pinned vertices (top row)
    pinned_mask = torch.zeros(num_vertices, dtype=torch.bool)
    pinned_mask[-resolution:] = True  # Top row
    
    trajectory = [vertices.clone()]
    velocities = torch.zeros_like(vertices)
    gravity = torch.tensor([0.0, -9.8, 0.0])
    
    # Simple spring constants
    damping = 0.98
    
    for frame in range(num_frames):
        # Apply gravity to non-pinned vertices
        velocities[~pinned_mask] += gravity * dt
        
        # Apply damping
        velocities *= damping
        
        # Update positions
        new_pos = trajectory[-1].clone()
        new_pos[~pinned_mask] += velocities[~pinned_mask] * dt
        
        # Simple ground collision (y >= -0.5)
        collision_mask = new_pos[:, 1] < -0.5
        new_pos[collision_mask, 1] = -0.5
        velocities[collision_mask, 1] *= -0.3  # bounce with damping
        
        trajectory.append(new_pos)
    
    return torch.stack(trajectory)


def simulate_wind(vertices, faces, num_frames=60, dt=1/60):
    """
    Simulation with oscillating wind force.
    
    Args:
        vertices: (N, 3) initial positions
        faces: (T, 3) triangle faces
        num_frames: Number of frames
        dt: Time step
        
    Returns:
        trajectory: (num_frames+1, N, 3) positions over time
    """
    num_vertices = vertices.shape[0]
    resolution = int(np.sqrt(num_vertices))
    
    # Identify pinned vertices (top row)
    pinned_mask = torch.zeros(num_vertices, dtype=torch.bool)
    pinned_mask[-resolution:] = True
    
    trajectory = [vertices.clone()]
    velocities = torch.zeros_like(vertices)
    gravity = torch.tensor([0.0, -9.8, 0.0])
    damping = 0.95
    
    for frame in range(num_frames):
        t = frame * dt
        
        # Oscillating wind in Z direction
        wind_strength = 2.0 * np.sin(2 * np.pi * t * 0.5)
        wind = torch.tensor([0.0, 0.0, wind_strength])
        
        # Apply forces
        velocities[~pinned_mask] += (gravity + wind) * dt
        velocities *= damping
        
        # Update positions
        new_pos = trajectory[-1].clone()
        new_pos[~pinned_mask] += velocities[~pinned_mask] * dt
        
        trajectory.append(new_pos)
    
    return torch.stack(trajectory)


def create_pyg_dataset(vertices, faces, trajectory, save_path="test_cloth_data.h5"):
    """
    Convert mesh and trajectory to PyG-compatible format and save as HDF5.
    
    Args:
        vertices: (N, 3) rest positions
        faces: (T, 3) triangle faces
        trajectory: (F, N, 3) position trajectory
        save_path: Output file path
        
    Returns:
        save_path: Path to saved file
    """
    edge_index = create_edges_from_faces(faces, len(vertices))
    
    # Compute displacements
    displacements = trajectory[1:] - trajectory[:-1]  # (F-1, N, 3)
    
    # Save dataset
    with h5py.File(save_path, "w") as f:
        f.create_dataset("trajectory", data=trajectory.numpy())
        f.create_dataset("displacements", data=displacements.numpy())
        f.create_dataset("edge_index", data=edge_index.numpy())
        f.create_dataset("faces", data=faces.numpy())
        f.create_dataset("rest_positions", data=vertices.numpy())
        
        # Metadata
        f.attrs["num_vertices"] = len(vertices)
        f.attrs["num_edges"] = edge_index.shape[1]
        f.attrs["num_faces"] = len(faces)
        f.attrs["num_frames"] = len(trajectory)
    
    print(f"Saved test dataset to: {save_path}")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Edges: {edge_index.shape[1]}")
    print(f"  Faces: {len(faces)}")
    print(f"  Frames: {len(trajectory)}")
    
    return save_path


def generate_all_test_data(output_dir=None):
    """Generate all test datasets."""
    if output_dir is None:
        # Use directory relative to this script
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    # Small mesh (100 vertices) - gravity
    print("\n=== Generating gravity simulation (10x10 mesh) ===")
    vertices, faces = create_square_mesh(resolution=10)
    trajectory = simulate_simple_gravity(vertices, faces, num_frames=60)
    create_pyg_dataset(vertices, faces, trajectory, 
                       os.path.join(output_dir, "gravity_10x10.h5"))
    
    # Small mesh - wind
    print("\n=== Generating wind simulation (10x10 mesh) ===")
    trajectory_wind = simulate_wind(vertices, faces, num_frames=60)
    create_pyg_dataset(vertices, faces, trajectory_wind,
                       os.path.join(output_dir, "wind_10x10.h5"))
    
    # Larger mesh (1024 vertices) for benchmark
    print("\n=== Generating benchmark mesh (32x32) ===")
    vertices_large, faces_large = create_square_mesh(resolution=32)
    trajectory_large = simulate_simple_gravity(vertices_large, faces_large, num_frames=30)
    create_pyg_dataset(vertices_large, faces_large, trajectory_large,
                       os.path.join(output_dir, "gravity_32x32.h5"))
    
    print("\n=== All test data generated ===")


if __name__ == "__main__":
    generate_all_test_data()
