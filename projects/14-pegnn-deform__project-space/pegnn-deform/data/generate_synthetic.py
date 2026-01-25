"""
PEGNN-Deform: Synthetic Data Generator

Generates training/validation/test datasets for deformable mesh simulation.
Uses spring-mass physics with gravity and damping to create ground truth.

Supports:
- Synthetic grid/sphere meshes
- Loading real meshes from OBJ/STL files

Author: PEGNN-Deform Team
"""

import argparse
import math
from pathlib import Path
from typing import Tuple, Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm


def load_mesh_from_file(
    filepath: Union[str, Path],
    scale: float = 1.0,
    center: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load mesh from OBJ/STL/PLY file using trimesh.

    Args:
        filepath: Path to mesh file (OBJ, STL, PLY, etc.)
        scale: Scale factor for mesh
        center: Center mesh at origin

    Returns:
        pos: Vertex positions [N, 3]
        edge_index: Edge indices [2, E]
        faces: Triangle faces [F, 3]

    Raises:
        ImportError: If trimesh is not installed
        FileNotFoundError: If mesh file doesn't exist
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError(
            "trimesh is required for mesh loading. "
            "Install with: pip install trimesh"
        )

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Mesh file not found: {filepath}")

    # Load mesh
    mesh = trimesh.load(filepath, force='mesh')

    # Get vertices
    pos = torch.tensor(mesh.vertices, dtype=torch.float32)

    # Scale and center
    if center:
        pos -= pos.mean(dim=0, keepdim=True)
    pos *= scale

    # Get faces
    faces = torch.tensor(mesh.faces, dtype=torch.long)

    # Get edges (unique edges from mesh)
    edges = mesh.edges_unique
    edge_list = []
    for edge in edges:
        edge_list.append([edge[0], edge[1]])
        edge_list.append([edge[1], edge[0]])  # Bidirectional

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    print(f"Loaded mesh from {filepath}:")
    print(f"  Vertices: {pos.size(0)}")
    print(f"  Faces: {faces.size(0)}")
    print(f"  Edges: {edge_index.size(1) // 2}")

    return pos, edge_index, faces


def load_mesh_sequence(
    directory: Union[str, Path],
    pattern: str = "*.obj",
    scale: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load a sequence of meshes from a directory.

    Args:
        directory: Directory containing mesh files
        pattern: Glob pattern for mesh files
        scale: Scale factor

    Returns:
        pos_sequence: Positions over time [T, N, 3]
        edge_index: Edge indices [2, E] (from first frame)
        faces: Faces [F, 3] (from first frame)
    """
    directory = Path(directory)
    mesh_files = sorted(directory.glob(pattern))

    if len(mesh_files) == 0:
        raise FileNotFoundError(f"No mesh files found in {directory} matching {pattern}")

    pos_list = []
    edge_index = None
    faces = None

    for i, mesh_file in enumerate(tqdm(mesh_files, desc="Loading meshes")):
        pos, ei, f = load_mesh_from_file(mesh_file, scale=scale, center=(i == 0))

        if i == 0:
            edge_index = ei
            faces = f
            centroid = pos.mean(dim=0)
        else:
            # Apply same centering as first frame
            pos -= pos.mean(dim=0, keepdim=True)
            pos += centroid

        pos_list.append(pos)

    pos_sequence = torch.stack(pos_list, dim=0)
    print(f"Loaded {len(mesh_files)} frames, shape: {pos_sequence.shape}")

    return pos_sequence, edge_index, faces


def create_grid_mesh(
    grid_size: int = 20,
    spacing: float = 0.05
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a 2D grid mesh.
    
    Args:
        grid_size: Number of vertices per side
        spacing: Distance between vertices
        
    Returns:
        pos: Vertex positions [N, 3]
        edge_index: Edge indices [2, E]
        faces: Triangle faces [F, 3]
    """
    N = grid_size * grid_size
    
    # Create grid positions
    x = torch.arange(grid_size, dtype=torch.float32) * spacing
    y = torch.arange(grid_size, dtype=torch.float32) * spacing
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    pos = torch.stack([
        xx.flatten(),
        yy.flatten(),
        torch.zeros(N)
    ], dim=1)
    
    # Center the mesh
    pos -= pos.mean(dim=0, keepdim=True)
    
    # Build edges (grid connectivity + diagonals for stability)
    edge_list = []
    face_list = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            
            # Right neighbor
            if j < grid_size - 1:
                edge_list.append([idx, idx + 1])
                edge_list.append([idx + 1, idx])
            
            # Bottom neighbor
            if i < grid_size - 1:
                edge_list.append([idx, idx + grid_size])
                edge_list.append([idx + grid_size, idx])
            
            # Diagonal (for structural stability)
            if i < grid_size - 1 and j < grid_size - 1:
                edge_list.append([idx, idx + grid_size + 1])
                edge_list.append([idx + grid_size + 1, idx])
                
                # Create triangles
                face_list.append([idx, idx + 1, idx + grid_size + 1])
                face_list.append([idx, idx + grid_size + 1, idx + grid_size])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    faces = torch.tensor(face_list, dtype=torch.long)
    
    return pos, edge_index, faces


def create_sphere_mesh(
    subdivisions: int = 3,
    radius: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create an icosphere mesh.
    
    Args:
        subdivisions: Number of subdivision iterations
        radius: Sphere radius
        
    Returns:
        pos, edge_index, faces
    """
    # Golden ratio
    phi = (1 + math.sqrt(5)) / 2
    
    # Initial icosahedron vertices
    vertices = [
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ]
    
    # Initial faces
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide
    for _ in range(subdivisions):
        new_faces = []
        edge_cache = {}
        
        def get_middle_point(p1, p2):
            key = (min(p1, p2), max(p1, p2))
            if key in edge_cache:
                return edge_cache[key]
            
            mid = [
                (vertices[p1][0] + vertices[p2][0]) / 2,
                (vertices[p1][1] + vertices[p2][1]) / 2,
                (vertices[p1][2] + vertices[p2][2]) / 2
            ]
            vertices.append(mid)
            idx = len(vertices) - 1
            edge_cache[key] = idx
            return idx
        
        for tri in faces:
            a, b, c = tri
            ab = get_middle_point(a, b)
            bc = get_middle_point(b, c)
            ca = get_middle_point(c, a)
            
            new_faces.append([a, ab, ca])
            new_faces.append([b, bc, ab])
            new_faces.append([c, ca, bc])
            new_faces.append([ab, bc, ca])
        
        faces = new_faces
    
    # Normalize to sphere and scale
    pos = torch.tensor(vertices, dtype=torch.float32)
    pos = pos / pos.norm(dim=1, keepdim=True) * radius
    
    faces = torch.tensor(faces, dtype=torch.long)
    
    # Build edges from faces
    edge_set = set()
    for face in faces:
        for i in range(3):
            a, b = face[i].item(), face[(i + 1) % 3].item()
            edge_set.add((min(a, b), max(a, b)))
    
    edge_list = []
    for a, b in edge_set:
        edge_list.append([a, b])
        edge_list.append([b, a])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    return pos, edge_index, faces


def compute_edge_attributes(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    stiffness: float = 10.0
) -> torch.Tensor:
    """
    Compute spring parameters for edges.
    
    Args:
        pos: Vertex positions [N, 3]
        edge_index: Edge indices [2, E]
        stiffness: Spring stiffness constant
        
    Returns:
        edge_attr: [E, 2] - (stiffness, rest_length)
    """
    src, dst = edge_index
    rest_length = torch.norm(pos[src] - pos[dst], dim=1)
    k = torch.full_like(rest_length, stiffness)
    return torch.stack([k, rest_length], dim=1)


def simulate_step(
    pos: torch.Tensor,
    vel: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    fixed_mask: torch.Tensor,
    dt: float = 0.01,
    gravity: float = -9.81,
    damping: float = 0.1,
    mass: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate one time step using explicit Euler with spring forces.
    
    This is the ground truth physics simulation.
    """
    N = pos.size(0)
    forces = torch.zeros_like(pos)
    
    # Spring forces (Hooke's law)
    src, dst = edge_index
    diff = pos[src] - pos[dst]
    dist = torch.norm(diff, dim=1, keepdim=True)
    direction = diff / (dist + 1e-8)
    
    k = edge_attr[:, 0:1]
    L0 = edge_attr[:, 1:2]
    
    spring_force = -k * (dist - L0) * direction
    
    # Scatter forces to nodes
    forces.scatter_add_(0, dst.unsqueeze(1).expand(-1, 3), spring_force)
    
    # Gravity
    forces[:, 2] += gravity * mass
    
    # Damping
    forces -= damping * vel
    
    # Integration
    acc = forces / mass
    vel_new = vel + dt * acc
    pos_new = pos + dt * vel_new
    
    # Apply boundary conditions (fixed nodes)
    pos_new[fixed_mask] = pos[fixed_mask]
    vel_new[fixed_mask] = 0
    
    return pos_new, vel_new


def generate_dataset(
    output_dir: Path,
    split: str,
    num_sequences: int,
    steps_per_sequence: int,
    mesh_type: str = 'grid',
    grid_size: int = 20,
    stiffness: float = 10.0,
    dt: float = 0.01,
    gravity: float = -2.0,
    damping: float = 0.5
):
    """
    Generate a dataset of deformable mesh simulations.
    
    Args:
        output_dir: Output directory
        split: Dataset split ('train', 'val', 'test')
        num_sequences: Number of simulation sequences
        steps_per_sequence: Steps per sequence
        mesh_type: 'grid' or 'sphere'
        grid_size: Size of grid mesh
        stiffness: Spring stiffness
        dt: Time step
        gravity: Gravity acceleration
        damping: Velocity damping
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sample_idx = 0
    
    for seq_idx in tqdm(range(num_sequences), desc=f"Generating {split}"):
        # Create mesh
        if mesh_type == 'grid':
            pos_rest, edge_index, faces = create_grid_mesh(grid_size=grid_size)
            # Fix top row
            fixed_mask = torch.zeros(pos_rest.size(0), dtype=torch.bool)
            top_row = torch.arange(grid_size) * grid_size + (grid_size - 1)
            fixed_mask[top_row] = True
        else:
            pos_rest, edge_index, faces = create_sphere_mesh(subdivisions=3)
            # Fix top quarter
            fixed_mask = pos_rest[:, 2] > pos_rest[:, 2].max() * 0.7
        
        # Compute edge attributes
        edge_attr = compute_edge_attributes(pos_rest, edge_index, stiffness)
        
        # Random initial perturbation
        pos = pos_rest.clone()
        
        # Add sinusoidal perturbation
        freq = torch.rand(1).item() * 2 + 1  # Random frequency
        amplitude = torch.rand(1).item() * 0.1 + 0.05
        
        if mesh_type == 'grid':
            pos[:, 2] += amplitude * torch.sin(pos[:, 0] * freq * math.pi) * torch.sin(pos[:, 1] * freq * math.pi)
        else:
            # Radial perturbation for sphere
            radial = pos / (pos.norm(dim=1, keepdim=True) + 1e-8)
            pos += amplitude * torch.sin(pos[:, 2] * freq * math.pi).unsqueeze(1) * radial
        
        # Apply fixed constraints to initial position
        pos[fixed_mask] = pos_rest[fixed_mask]
        
        # Initial velocity (small random)
        vel = torch.randn_like(pos) * 0.01
        vel[fixed_mask] = 0
        
        # Simulate and save each step
        for step in range(steps_per_sequence):
            # Current state
            pos_current = pos.clone()
            vel_current = vel.clone()
            
            # Simulate one step
            pos_next, vel_next = simulate_step(
                pos, vel, edge_index, edge_attr, fixed_mask,
                dt=dt, gravity=gravity, damping=damping
            )
            
            # Save sample
            data = Data(
                pos=pos_current,
                vel=vel_current,
                pos_next=pos_next,
                vel_next=vel_next,
                pos_rest=pos_rest,
                edge_index=edge_index,
                edge_attr=edge_attr,
                fixed_mask=fixed_mask
            )
            
            torch.save(data, output_dir / f"{split}_{sample_idx:06d}.pt")
            sample_idx += 1
            
            # Update state
            pos = pos_next
            vel = vel_next
    
    print(f"Generated {sample_idx} samples for {split}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic data')
    
    parser.add_argument('--output-dir', type=str, default='data/synthetic',
                        help='Output directory')
    parser.add_argument('--mesh-type', type=str, default='grid',
                        choices=['grid', 'sphere'],
                        help='Type of mesh to generate')
    parser.add_argument('--grid-size', type=int, default=20,
                        help='Grid size (for grid mesh)')
    parser.add_argument('--num-train', type=int, default=100,
                        help='Number of training sequences')
    parser.add_argument('--num-val', type=int, default=20,
                        help='Number of validation sequences')
    parser.add_argument('--num-test', type=int, default=20,
                        help='Number of test sequences')
    parser.add_argument('--steps', type=int, default=50,
                        help='Steps per sequence')
    parser.add_argument('--stiffness', type=float, default=10.0,
                        help='Spring stiffness')
    parser.add_argument('--gravity', type=float, default=-2.0,
                        help='Gravity acceleration')
    parser.add_argument('--damping', type=float, default=0.5,
                        help='Velocity damping')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    
    print(f"Generating {args.mesh_type} mesh data...")
    print(f"Grid size: {args.grid_size} ({args.grid_size**2} nodes)")
    print(f"Output directory: {output_dir}")
    
    # Generate datasets
    generate_dataset(
        output_dir, 'train',
        num_sequences=args.num_train,
        steps_per_sequence=args.steps,
        mesh_type=args.mesh_type,
        grid_size=args.grid_size,
        stiffness=args.stiffness,
        gravity=args.gravity,
        damping=args.damping
    )
    
    generate_dataset(
        output_dir, 'val',
        num_sequences=args.num_val,
        steps_per_sequence=args.steps,
        mesh_type=args.mesh_type,
        grid_size=args.grid_size,
        stiffness=args.stiffness,
        gravity=args.gravity,
        damping=args.damping
    )
    
    generate_dataset(
        output_dir, 'test',
        num_sequences=args.num_test,
        steps_per_sequence=args.steps,
        mesh_type=args.mesh_type,
        grid_size=args.grid_size,
        stiffness=args.stiffness,
        gravity=args.gravity,
        damping=args.damping
    )
    
    print("\nData generation complete!")
    print(f"Total samples: {(args.num_train + args.num_val + args.num_test) * args.steps}")


if __name__ == '__main__':
    main()
