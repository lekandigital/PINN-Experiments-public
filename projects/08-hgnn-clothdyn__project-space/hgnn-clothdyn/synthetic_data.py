"""
Synthetic Cloth Simulation Data Generator

Generates ground-truth cloth simulation data for training and testing.
Uses a simple mass-spring system with gravity and ground collision.

Author: HGNN-ClothDyn
"""

import numpy as np
import h5py
import argparse
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClothSimulator:
    """
    Simple mass-spring cloth simulator for generating ground truth data.
    
    Uses Verlet integration with:
    - Structural springs (horizontal/vertical edges)
    - Shear springs (diagonal edges)
    - Bend springs (skip connections)
    - Gravity and ground collision
    """
    
    def __init__(
        self,
        mesh_size: int = 20,
        spacing: float = 0.05,
        mass: float = 0.01,
        stiffness: float = 500.0,
        damping: float = 0.99,
        gravity: float = -9.81,
        ground_height: float = -1.0,
        dt: float = 0.001,
        substeps: int = 10
    ):
        """
        Args:
            mesh_size: Grid size (NxN vertices)
            spacing: Distance between adjacent vertices
            mass: Mass per vertex
            stiffness: Spring stiffness coefficient
            damping: Velocity damping factor
            gravity: Gravitational acceleration
            ground_height: Ground plane y-coordinate
            dt: Time step per substep
            substeps: Number of substeps per frame
        """
        self.mesh_size = mesh_size
        self.spacing = spacing
        self.mass = mass
        self.stiffness = stiffness
        self.damping = damping
        self.gravity = gravity
        self.ground_height = ground_height
        self.dt = dt
        self.substeps = substeps
        
        # Initialize mesh
        self._init_mesh()
        self._init_springs()
        
        logger.info(f"ClothSimulator initialized: {mesh_size}x{mesh_size} mesh, "
                    f"{len(self.edges)} springs")
    
    def _init_mesh(self):
        """Initialize vertex positions and velocities."""
        n = self.mesh_size
        
        # Create grid positions (cloth starts horizontal at y=0.5)
        x = np.linspace(-self.spacing * (n - 1) / 2, self.spacing * (n - 1) / 2, n)
        z = np.linspace(-self.spacing * (n - 1) / 2, self.spacing * (n - 1) / 2, n)
        xx, zz = np.meshgrid(x, z)
        
        self.positions = np.zeros((n * n, 3), dtype=np.float32)
        self.positions[:, 0] = xx.flatten()
        self.positions[:, 1] = 0.5  # Starting height
        self.positions[:, 2] = zz.flatten()
        
        self.velocities = np.zeros((n * n, 3), dtype=np.float32)
        
        # Previous positions for Verlet integration
        self.prev_positions = self.positions.copy()
        
        # Fixed corners (top two corners pinned)
        self.fixed_mask = np.zeros(n * n, dtype=bool)
        self.fixed_mask[0] = True  # Top-left
        self.fixed_mask[n - 1] = True  # Top-right
    
    def _init_springs(self):
        """Initialize spring connections."""
        n = self.mesh_size
        edges = []
        
        # Helper to add edge
        def add_edge(i, j, spring_type):
            if 0 <= i < n * n and 0 <= j < n * n:
                edges.append((min(i, j), max(i, j), spring_type))
        
        for row in range(n):
            for col in range(n):
                idx = row * n + col
                
                # Structural springs (horizontal and vertical)
                if col < n - 1:
                    add_edge(idx, idx + 1, 'structural')
                if row < n - 1:
                    add_edge(idx, idx + n, 'structural')
                
                # Shear springs (diagonals)
                if col < n - 1 and row < n - 1:
                    add_edge(idx, idx + n + 1, 'shear')
                    add_edge(idx + 1, idx + n, 'shear')
                
                # Bend springs (skip one)
                if col < n - 2:
                    add_edge(idx, idx + 2, 'bend')
                if row < n - 2:
                    add_edge(idx, idx + 2 * n, 'bend')
        
        # Remove duplicates and create arrays
        edges = list(set(edges))
        self.edges = np.array([(e[0], e[1]) for e in edges], dtype=np.int64)
        self.spring_types = [e[2] for e in edges]
        
        # Compute rest lengths
        self.rest_lengths = self._compute_edge_lengths(self.positions)
        
        # Spring stiffness by type
        self.spring_stiffness = np.array([
            self.stiffness if t == 'structural' else 
            self.stiffness * 0.5 if t == 'shear' else
            self.stiffness * 0.25
            for t in self.spring_types
        ], dtype=np.float32)
    
    def _compute_edge_lengths(self, positions: np.ndarray) -> np.ndarray:
        """Compute current edge lengths."""
        v0 = positions[self.edges[:, 0]]
        v1 = positions[self.edges[:, 1]]
        return np.linalg.norm(v1 - v0, axis=1).astype(np.float32)
    
    def _apply_spring_forces(self, positions: np.ndarray) -> np.ndarray:
        """Compute spring forces for all vertices."""
        forces = np.zeros_like(positions)
        
        v0 = positions[self.edges[:, 0]]
        v1 = positions[self.edges[:, 1]]
        
        diff = v1 - v0
        lengths = np.linalg.norm(diff, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-8)
        
        # Normalized direction
        direction = diff / lengths
        
        # Spring force (Hooke's law)
        stretch = lengths.flatten() - self.rest_lengths
        force_magnitude = self.spring_stiffness * stretch
        
        # Apply forces to both ends
        spring_forces = direction * force_magnitude[:, np.newaxis]
        
        np.add.at(forces, self.edges[:, 0], spring_forces)
        np.add.at(forces, self.edges[:, 1], -spring_forces)
        
        return forces
    
    def step(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform one simulation frame (multiple substeps).
        
        Returns:
            positions: Current vertex positions
            velocities: Current vertex velocities
            collision_mask: Boolean mask of vertices in collision
        """
        for _ in range(self.substeps):
            self._substep()
        
        # Detect collisions
        collision_mask = self.positions[:, 1] <= self.ground_height + 0.01
        
        return self.positions.copy(), self.velocities.copy(), collision_mask
    
    def _substep(self):
        """Perform single integration substep using Verlet integration."""
        # Compute forces
        forces = self._apply_spring_forces(self.positions)
        
        # Add gravity
        forces[:, 1] += self.gravity * self.mass
        
        # Verlet integration
        acceleration = forces / self.mass
        
        # New position
        new_positions = (
            2 * self.positions - 
            self.prev_positions + 
            acceleration * self.dt * self.dt
        )
        
        # Update velocities (for output)
        self.velocities = (new_positions - self.positions) / self.dt * self.damping
        
        # Apply fixed constraints
        new_positions[self.fixed_mask] = self.positions[self.fixed_mask]
        self.velocities[self.fixed_mask] = 0
        
        # Ground collision
        below_ground = new_positions[:, 1] < self.ground_height
        new_positions[below_ground, 1] = self.ground_height
        self.velocities[below_ground, 1] = np.abs(self.velocities[below_ground, 1]) * 0.3
        
        # Update state
        self.prev_positions = self.positions.copy()
        self.positions = new_positions
    
    def get_mesh_data(self) -> Dict[str, np.ndarray]:
        """Get current mesh data for export."""
        return {
            'positions': self.positions.copy(),
            'velocities': self.velocities.copy(),
            'edges': self.edges.copy(),
            'rest_lengths': self.rest_lengths.copy(),
            'fixed_mask': self.fixed_mask.copy()
        }
    
    def reset(self):
        """Reset simulation to initial state."""
        self._init_mesh()


def generate_cloth_sequence(
    num_frames: int = 100,
    mesh_size: int = 20,
    output_path: Optional[str] = None,
    **sim_kwargs
) -> Dict[str, np.ndarray]:
    """
    Generate a cloth simulation sequence.
    
    Args:
        num_frames: Number of frames to simulate
        mesh_size: Grid size (NxN vertices)
        output_path: Path to save HDF5 file (optional)
        **sim_kwargs: Additional arguments for ClothSimulator
        
    Returns:
        Dictionary containing simulation data
    """
    logger.info(f"Generating {num_frames} frames for {mesh_size}x{mesh_size} mesh...")
    
    sim = ClothSimulator(mesh_size=mesh_size, **sim_kwargs)
    
    # Storage arrays
    all_positions = np.zeros((num_frames, mesh_size * mesh_size, 3), dtype=np.float32)
    all_velocities = np.zeros((num_frames, mesh_size * mesh_size, 3), dtype=np.float32)
    all_collisions = np.zeros((num_frames, mesh_size * mesh_size), dtype=bool)
    
    # Initial state
    mesh_data = sim.get_mesh_data()
    all_positions[0] = mesh_data['positions']
    all_velocities[0] = mesh_data['velocities']
    
    # Simulate
    for frame in range(1, num_frames):
        pos, vel, coll = sim.step()
        all_positions[frame] = pos
        all_velocities[frame] = vel
        all_collisions[frame] = coll
        
        if frame % 20 == 0:
            logger.info(f"  Frame {frame}/{num_frames}")
    
    # Compile data
    data = {
        'positions': all_positions,
        'velocities': all_velocities,
        'collisions': all_collisions,
        'edges': mesh_data['edges'],
        'rest_lengths': mesh_data['rest_lengths'],
        'fixed_mask': mesh_data['fixed_mask'],
        'mesh_size': np.array([mesh_size]),
        'num_frames': np.array([num_frames])
    }
    
    # Save to HDF5
    if output_path:
        save_to_hdf5(data, output_path)
    
    logger.info(f"Generated {num_frames} frames successfully")
    
    return data


def save_to_hdf5(data: Dict[str, np.ndarray], filepath: str):
    """Save simulation data to HDF5 file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        # Metadata
        f.attrs['created'] = datetime.now().isoformat()
        f.attrs['version'] = '1.0'
        
        # Store arrays
        for key, value in data.items():
            f.create_dataset(key, data=value, compression='gzip')
    
    logger.info(f"Saved to {filepath}")


def load_from_hdf5(filepath: str) -> Dict[str, np.ndarray]:
    """Load simulation data from HDF5 file."""
    data = {}
    
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            data[key] = f[key][:]
    
    logger.info(f"Loaded from {filepath}: {list(data.keys())}")
    
    return data


def generate_multiple_sequences(
    num_sequences: int = 5,
    num_frames: int = 100,
    mesh_size: int = 20,
    output_dir: str = 'data',
    vary_params: bool = True
) -> None:
    """
    Generate multiple sequences with varied parameters.
    
    Args:
        num_sequences: Number of sequences to generate
        num_frames: Frames per sequence
        mesh_size: Mesh size
        output_dir: Output directory
        vary_params: Whether to vary simulation parameters
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_sequences):
        # Vary parameters for diversity
        if vary_params:
            stiffness = 300 + np.random.rand() * 400  # 300-700
            damping = 0.95 + np.random.rand() * 0.04  # 0.95-0.99
            gravity = -9.81 * (0.8 + np.random.rand() * 0.4)  # 0.8x-1.2x
        else:
            stiffness = 500
            damping = 0.99
            gravity = -9.81
        
        output_path = output_dir / f'cloth_sequence_{i:03d}.h5'
        
        generate_cloth_sequence(
            num_frames=num_frames,
            mesh_size=mesh_size,
            output_path=str(output_path),
            stiffness=stiffness,
            damping=damping,
            gravity=gravity
        )
        
        logger.info(f"Generated sequence {i+1}/{num_sequences}")


def visualize_sequence(data: Dict[str, np.ndarray], frame_step: int = 10):
    """
    Visualize simulation sequence using matplotlib.
    
    Args:
        data: Simulation data dictionary
        frame_step: Step between plotted frames
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        logger.warning("matplotlib not available for visualization")
        return
    
    positions = data['positions']
    num_frames = len(positions)
    
    # Select frames to plot
    frames_to_plot = list(range(0, num_frames, frame_step))
    n_plots = len(frames_to_plot)
    
    fig = plt.figure(figsize=(4 * min(n_plots, 5), 4 * ((n_plots - 1) // 5 + 1)))
    
    for idx, frame in enumerate(frames_to_plot):
        ax = fig.add_subplot((n_plots - 1) // 5 + 1, min(n_plots, 5), idx + 1, 
                             projection='3d')
        
        pos = positions[frame]
        ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1], c='blue', s=1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_title(f'Frame {frame}')
        
        # Set consistent axis limits
        ax.set_xlim([-0.6, 0.6])
        ax.set_ylim([-0.6, 0.6])
        ax.set_zlim([-1.2, 0.6])
    
    plt.tight_layout()
    plt.savefig('simulation_frames.png', dpi=150)
    plt.close()
    
    logger.info("Saved visualization to simulation_frames.png")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic cloth simulation data'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/synthetic_cloth.h5',
        help='Output HDF5 file path'
    )
    parser.add_argument(
        '--frames', '-f',
        type=int,
        default=100,
        help='Number of frames to simulate'
    )
    parser.add_argument(
        '--mesh-size', '-m',
        type=int,
        default=20,
        help='Mesh grid size (NxN vertices)'
    )
    parser.add_argument(
        '--stiffness', '-k',
        type=float,
        default=500.0,
        help='Spring stiffness coefficient'
    )
    parser.add_argument(
        '--gravity', '-g',
        type=float,
        default=-9.81,
        help='Gravitational acceleration'
    )
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Generate visualization of simulation'
    )
    parser.add_argument(
        '--multi', '-M',
        type=int,
        default=0,
        help='Generate multiple sequences with varied parameters'
    )
    
    args = parser.parse_args()
    
    if args.multi > 0:
        generate_multiple_sequences(
            num_sequences=args.multi,
            num_frames=args.frames,
            mesh_size=args.mesh_size,
            output_dir=Path(args.output).parent
        )
    else:
        data = generate_cloth_sequence(
            num_frames=args.frames,
            mesh_size=args.mesh_size,
            output_path=args.output,
            stiffness=args.stiffness,
            gravity=args.gravity
        )
        
        if args.visualize:
            visualize_sequence(data)
    
    print(f"\n✓ Data generation complete: {args.output}")


if __name__ == '__main__':
    main()
