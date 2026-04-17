"""
Cloth Dynamics Dataset Generator.

Generates training data for ClothGNN using physics simulation.
This creates ground truth trajectories that can be used for:
1. Training without a teacher (baseline)
2. Providing hard labels alongside soft distillation targets

Physics Model:
- Position-Based Dynamics (PBD) for cloth simulation
- Constraint-based edge length preservation
- Simple gravity + damping

Output Format (HDF5):
- sequences/{i}/positions: [T, N, 3] trajectory
- sequences/{i}/velocities: [T, N, 3] velocities  
- sequences/{i}/edge_index: [2, E] connectivity
- sequences/{i}/rest_lengths: [E] rest edge lengths
- sequences/{i}/fixed_mask: [N] boolean mask for fixed vertices
"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, List
from tqdm import tqdm
import h5py


class PositionBasedDynamics:
    """
    Position-Based Dynamics (PBD) solver for cloth simulation.
    
    This is a stable, fast simulation method suitable for generating
    training data. It uses constraint projection rather than explicit
    force computation.
    """
    
    def __init__(
        self,
        stiffness: float = 1.0,
        damping: float = 0.99,
        gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0),
        num_constraint_iters: int = 10,
    ):
        self.stiffness = stiffness
        self.damping = damping
        self.gravity = torch.tensor(gravity)
        self.num_constraint_iters = num_constraint_iters
    
    def step(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        edge_index: torch.Tensor,
        rest_lengths: torch.Tensor,
        fixed_mask: torch.Tensor,
        dt: float = 1/60,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one simulation step.
        
        Args:
            positions: [N, 3] current positions
            velocities: [N, 3] current velocities
            edge_index: [2, E] edge connectivity
            rest_lengths: [E] rest edge lengths
            fixed_mask: [N] boolean mask (True = fixed vertex)
            dt: Time step
            
        Returns:
            new_positions: [N, 3]
            new_velocities: [N, 3]
        """
        device = positions.device
        self.gravity = self.gravity.to(device)
        
        # Predict positions (explicit Euler with gravity)
        predicted = positions + velocities * dt
        predicted = predicted + 0.5 * self.gravity * dt * dt
        
        # Apply distance constraints iteratively
        for _ in range(self.num_constraint_iters):
            predicted = self._project_distance_constraints(
                predicted, edge_index, rest_lengths, fixed_mask
            )
        
        # Update velocities
        new_velocities = (predicted - positions) / dt
        new_velocities = new_velocities * self.damping
        
        # Enforce fixed vertices
        predicted = torch.where(
            fixed_mask.unsqueeze(-1),
            positions,
            predicted,
        )
        new_velocities = torch.where(
            fixed_mask.unsqueeze(-1),
            torch.zeros_like(new_velocities),
            new_velocities,
        )
        
        return predicted, new_velocities
    
    def _project_distance_constraints(
        self,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        rest_lengths: torch.Tensor,
        fixed_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Project positions to satisfy distance constraints."""
        src, dst = edge_index
        
        # Current edge vectors and lengths
        diff = positions[dst] - positions[src]
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-8)
        direction = diff / dist
        
        # Compute constraint violation
        stretch = dist.squeeze(-1) - rest_lengths
        
        # Compute corrections (split equally between vertices)
        # If one vertex is fixed, the free vertex takes full correction
        correction = 0.5 * self.stiffness * stretch.unsqueeze(-1) * direction
        
        # Weight by inverse mass (fixed = infinite mass)
        w_src = (~fixed_mask[src]).float().unsqueeze(-1)
        w_dst = (~fixed_mask[dst]).float().unsqueeze(-1)
        w_total = w_src + w_dst + 1e-8
        
        correction_src = -correction * w_src / w_total
        correction_dst = correction * w_dst / w_total
        
        # Apply corrections
        new_positions = positions.clone()
        new_positions.scatter_add_(0, src.unsqueeze(-1).expand(-1, 3), correction_src)
        new_positions.scatter_add_(0, dst.unsqueeze(-1).expand(-1, 3), correction_dst)
        
        return new_positions
    
    def simulate(
        self,
        initial_positions: torch.Tensor,
        edge_index: torch.Tensor,
        rest_lengths: torch.Tensor,
        fixed_mask: torch.Tensor,
        n_steps: int,
        dt: float = 1/60,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run simulation for multiple steps.
        
        Returns:
            positions: [T, N, 3] position trajectory
            velocities: [T, N, 3] velocity trajectory
        """
        positions_list = [initial_positions]
        velocities_list = [torch.zeros_like(initial_positions)]
        
        pos = initial_positions.clone()
        vel = torch.zeros_like(pos)
        
        for _ in range(n_steps - 1):
            pos, vel = self.step(pos, vel, edge_index, rest_lengths, fixed_mask, dt)
            positions_list.append(pos)
            velocities_list.append(vel)
        
        return torch.stack(positions_list), torch.stack(velocities_list)


def create_cloth_mesh(
    size: int,
    scale: float = 2.0,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a square cloth mesh.
    
    Args:
        size: Grid resolution (size x size vertices)
        scale: Physical size
        center: Center position
        
    Returns:
        positions: [N, 3] vertex positions
        edge_index: [2, E] edge connectivity (structural + shear + bend)
        fixed_mask: [N] boolean mask (top row fixed)
    """
    n_nodes = size * size
    
    # Create grid positions
    x = torch.linspace(-scale/2, scale/2, size) + center[0]
    z = torch.linspace(-scale/2, scale/2, size) + center[2]
    xx, zz = torch.meshgrid(x, z, indexing='ij')
    
    positions = torch.stack([
        xx.flatten(),
        torch.full((n_nodes,), center[1]),
        zz.flatten(),
    ], dim=-1).float()
    
    # Create edges
    edges = []
    
    for i in range(size):
        for j in range(size):
            node = i * size + j
            
            # Structural edges (right, down)
            if j < size - 1:
                edges.append([node, node + 1])
            if i < size - 1:
                edges.append([node, node + size])
            
            # Shear edges (diagonals)
            if i < size - 1 and j < size - 1:
                edges.append([node, node + size + 1])
            if i < size - 1 and j > 0:
                edges.append([node, node + size - 1])
            
            # Bend edges (skip one)
            if j < size - 2:
                edges.append([node, node + 2])
            if i < size - 2:
                edges.append([node, node + 2 * size])
    
    # Make undirected
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Remove duplicates
    edge_index = torch.unique(edge_index, dim=1)
    
    # Fixed mask (top row)
    fixed_mask = torch.zeros(n_nodes, dtype=torch.bool)
    fixed_mask[:size] = True
    
    return positions, edge_index, fixed_mask


def add_random_perturbation(
    positions: torch.Tensor,
    fixed_mask: torch.Tensor,
    magnitude: float = 0.1,
) -> torch.Tensor:
    """Add random perturbation to non-fixed vertices."""
    perturbation = torch.randn_like(positions) * magnitude
    perturbation = torch.where(fixed_mask.unsqueeze(-1), torch.zeros_like(perturbation), perturbation)
    return positions + perturbation


def generate_dataset(
    output_path: str,
    n_sequences: int = 1000,
    n_steps: int = 100,
    mesh_sizes: Tuple[int, ...] = (20, 25, 30, 32),
    dt: float = 1/60,
    perturbation_magnitude: float = 0.1,
    seed: int = 42,
    device: str = "cpu",
):
    """
    Generate cloth dynamics dataset.
    
    Args:
        output_path: Path to output HDF5 file
        n_sequences: Number of sequences to generate
        n_steps: Steps per sequence
        mesh_sizes: Cloth grid sizes to sample from
        dt: Time step
        perturbation_magnitude: Initial perturbation magnitude
        seed: Random seed
        device: Compute device
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize simulator
    simulator = PositionBasedDynamics(
        stiffness=0.9,
        damping=0.98,
        gravity=(0.0, -9.81, 0.0),
        num_constraint_iters=15,
    )
    
    with h5py.File(output_path, 'w') as f:
        # Store metadata
        f.attrs['n_sequences'] = n_sequences
        f.attrs['n_steps'] = n_steps
        f.attrs['dt'] = dt
        f.attrs['mesh_sizes'] = list(mesh_sizes)
        f.attrs['perturbation_magnitude'] = perturbation_magnitude
        f.attrs['seed'] = seed
        
        sequences_grp = f.create_group('sequences')
        
        for seq_idx in tqdm(range(n_sequences), desc="Generating sequences"):
            # Random mesh size
            size = int(np.random.choice(mesh_sizes))
            
            # Create cloth mesh
            positions, edge_index, fixed_mask = create_cloth_mesh(size)
            
            # Compute rest lengths
            src, dst = edge_index
            rest_lengths = torch.norm(positions[dst] - positions[src], dim=-1)
            
            # Add random initial perturbation
            positions = add_random_perturbation(positions, fixed_mask, perturbation_magnitude)
            
            # Move to device
            positions = positions.to(device)
            edge_index = edge_index.to(device)
            rest_lengths = rest_lengths.to(device)
            fixed_mask = fixed_mask.to(device)
            
            # Run simulation
            pos_traj, vel_traj = simulator.simulate(
                positions, edge_index, rest_lengths, fixed_mask,
                n_steps=n_steps, dt=dt,
            )
            
            # Save sequence
            seq_grp = sequences_grp.create_group(str(seq_idx))
            seq_grp.create_dataset('positions', data=pos_traj.cpu().numpy(), compression='gzip')
            seq_grp.create_dataset('velocities', data=vel_traj.cpu().numpy(), compression='gzip')
            seq_grp.create_dataset('edge_index', data=edge_index.cpu().numpy())
            seq_grp.create_dataset('rest_lengths', data=rest_lengths.cpu().numpy())
            seq_grp.create_dataset('fixed_mask', data=fixed_mask.cpu().numpy())
            seq_grp.attrs['mesh_size'] = size
    
    print(f"Dataset saved to {output_path}")
    print(f"Total sequences: {n_sequences}")
    print(f"Steps per sequence: {n_steps}")
    
    # Print statistics
    with h5py.File(output_path, 'r') as f:
        total_frames = sum(
            f['sequences'][str(i)]['positions'].shape[0]
            for i in range(n_sequences)
        )
        total_vertices = sum(
            f['sequences'][str(i)]['positions'].shape[1]
            for i in range(n_sequences)
        )
        print(f"Total frames: {total_frames}")
        print(f"Average vertices per sequence: {total_vertices / n_sequences:.0f}")


def split_dataset(
    h5_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    output_dir: Optional[str] = None,
    seed: int = 42,
):
    """
    Split dataset into train/val/test sets.
    
    Creates index files rather than duplicating data.
    """
    np.random.seed(seed)
    
    with h5py.File(h5_path, 'r') as f:
        n_sequences = f.attrs['n_sequences']
    
    indices = np.arange(n_sequences)
    np.random.shuffle(indices)
    
    n_train = int(n_sequences * train_ratio)
    n_val = int(n_sequences * val_ratio)
    
    splits = {
        'train': indices[:n_train].tolist(),
        'val': indices[n_train:n_train + n_val].tolist(),
        'test': indices[n_train + n_val:].tolist(),
    }
    
    if output_dir is None:
        output_dir = Path(h5_path).parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(output_dir / 'splits.json', 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"Split indices saved to {output_dir / 'splits.json'}")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate cloth dynamics dataset")
    parser.add_argument("--output", type=str, default="data/cloth_dynamics.h5",
                        help="Output HDF5 file path")
    parser.add_argument("--n-sequences", type=int, default=1000,
                        help="Number of sequences")
    parser.add_argument("--n-steps", type=int, default=100,
                        help="Steps per sequence")
    parser.add_argument("--mesh-sizes", type=int, nargs="+", default=[20, 25, 30, 32],
                        help="Mesh sizes to sample from")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda)")
    parser.add_argument("--split", action="store_true",
                        help="Also create train/val/test splits")
    
    args = parser.parse_args()
    
    generate_dataset(
        output_path=args.output,
        n_sequences=args.n_sequences,
        n_steps=args.n_steps,
        mesh_sizes=tuple(args.mesh_sizes),
        seed=args.seed,
        device=args.device,
    )
    
    if args.split:
        split_dataset(args.output)
